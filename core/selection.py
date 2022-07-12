import  torch 
import torch.nn as nn
from torch import autograd
from core.utils import grad_with_none
import core.utils as utils
from core.utils import Config

import jax ## need to import jax before torchopt otherwise get an error
import TorchOpt
import core
from itertools import cycle
import copy
from functools import partial


from functorch import make_functional_with_buffers, jacfwd, jacrev, jvp

import os
import importlib

import nvidia_smi
import torch.optim as optim

def get_gpu_usage(device):
	nvidia_smi.nvmlInit()

	deviceCount = nvidia_smi.nvmlDeviceGetCount()
	handle = nvidia_smi.nvmlDeviceGetHandleByIndex(device)
	info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
	print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(device, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used))

	nvidia_smi.nvmlShutdown()
	

def config_to_instance(config_module_name="name",**config):
	module_name = config.pop(config_module_name)
	try:
		attr = import_module(module_name)
		if config:
			attr = attr(**config)
	except:
		attr = eval(module_name)(**config)
	return attr


def config_to_instance(config_module_name="name",**config):
	module_name = config.pop(config_module_name)
	attr = import_module(module_name)
	if config:
		attr = attr(**config)
	return attr
def import_module(module_name):
    module, attr = os.path.splitext(module_name)
    try:
        module = importlib.import_module(module)
        return getattr(module, attr[1:])
    except:
        try:
            module = import_module(module)
            return getattr(module, attr[1:])
        except:
            return eval(module+attr[1:])

class Functional(object):
	def __init__(self,module):
		self.module= module
		module.train(True)
		self.func, self.weights, self.buffers = make_functional_with_buffers(module)
		module.train(False)
		self.eval_func, _,_ = make_functional_with_buffers(module)
	def eval(self,inputs,upper_var,lower_var,train_mode=True,**kwargs):
		params = upper_var + lower_var 
		if train_mode:
			return self.func(params,self.buffers,inputs,**kwargs)
		else:
			return self.eval_func(params,self.buffers,inputs,**kwargs)
	def __call__(self,inputs,upper_var,lower_var,train_mode=True,**kwargs):
		return self.eval(inputs,upper_var,lower_var,train_mode=train_mode,**kwargs)



class Selection(nn.Module):
	def __init__(self,
				func,
				init_lower_var,
				generator,
				options,
				device,
				dtype):
		super(Selection,self).__init__()
		self.func = func
		### Convert data loader to a ring generator
		self.generator = RingGenerator(generator)
		
		self.lower_var = tuple(init_lower_var)
		self.options = options

		self.linear_solver = LinearSolver(self.func,self.generator,options.linear_solver,
									device, dtype)
		if isinstance(self.linear_solver.residual_op,FiniteDiffResidual):
			self.track_grad_for_backward=False
		else:
			self.track_grad_for_backward=True


		self.optimizer = DiffOpt(self.func,self.generator,
								init_lower_var,options.optimizer,
								options.warm_start_iter,options.unrolled_iter,
									device, dtype,
									track_grad_for_backward=self.track_grad_for_backward)


		self.use_scheduler = options.scheduler.pop("use_scheduler", None)
		if self.use_scheduler:
			dummy_opt = optim.SGD(self.lower_var, lr = self.optimizer.lr)
			self.scheduler = config_to_instance(**options.scheduler, optimizer = dummy_opt)

		if options.correction:
			self.dual_var = [torch.zeros_like(p) for p in init_lower_var]
		else:
			self.dual_var = None
	def update_lr(self):
		if self.use_scheduler:
			self.scheduler.step()
			self.optimizer.update_lr(self.scheduler.get_last_lr()[0])
	
	def update_dual(self,dual_var):
		if self.options.dual_var_warm_start:
			self.dual_var = dual_var
			# norm = 0.
			# for i,var in enumerate(dual_var):
			# 	norm += torch.mean(var**2)
			# norm = norm/i 
			# print(norm)

	def update_var(self,opt_lower_var):
		for p,new_p in zip(self.lower_var,opt_lower_var):
			p.data.copy_(new_p.data)

	def forward(self,*all_params):
		len_lower = len(self.lower_var)
		with  torch.enable_grad():
			opt_lower_var =  ArgMinOp.apply(self,len_lower,*all_params)
		
		return  opt_lower_var

class ArgMinOp(torch.autograd.Function):

	@staticmethod
	def forward(ctx,selection, len_lower,*all_params):
		lower_var = all_params[:len_lower]
		upper_var = all_params[len_lower:]
		ctx.selection = selection
		ctx.len_lower = len_lower
		with  torch.enable_grad():
			iterates, val,grad, inputs = selection.optimizer.run(upper_var,lower_var)
		ctx.iterates = iterates
		ctx.grad = grad
		ctx.inputs = inputs
		ctx.save_for_backward(*all_params)
		
		return tuple( p.detach() for p in iterates[-1])

	@staticmethod
	def backward(ctx, *grad_output):
		selection = ctx.selection
		iterates = ctx.iterates
		len_lower = ctx.len_lower
		upper_var = ctx.saved_tensors[len_lower:]
		lower_var = ctx.saved_tensors[:len_lower]
		grad = ctx.grad
		inputs = ctx.inputs
		with  torch.enable_grad():
			if len(iterates)>1:
				
				val = [ torch.einsum('...i,...i->',y,g) for y,g in zip(  iterates[-1], grad_output)]
				val = torch.sum(torch.stack(val))
				all_params = iterates[0]+upper_var
				len_lower = len(lower_var)
				grad_selection = autograd.grad(
									outputs=val, 
									inputs=all_params, 
									grad_outputs=None, 
									retain_graph=selection.track_grad_for_backward,
									create_graph=False, 
									only_inputs=True,
									allow_unused=True)
				grad_selection_lower =  grad_selection[:len_lower]
				grad_selection_upper =  grad_selection[len_lower:]
				del iterates
			else:
				grad_selection_lower =  grad_output
				grad_selection_upper =  tuple([torch.zeros_like(var) for var in upper_var])
		## Solve a system Ax+b=0, 
		# A: the hessian of the lower objective, 
		# b:   grad_selection_lower
		if selection.options.correction:
			correction, dual_var = selection.linear_solver.run(
								upper_var,
								lower_var,
								selection.dual_var,
								grad_selection_lower,grad,inputs)		
			
			## summing the contributions of partial_x phi and correction term
			for g_upper,g_lin in zip(grad_selection_upper,correction):
				g_upper.data.add_(g_lin)
			
			## update the dual variable for next iteration 
			selection.update_dual(dual_var)
		#print(hello)

		return (None,)*(len(lower_var)+2) + grad_selection_upper



class RingGenerator:
	def __init__(self, init_generator):
		self.init_generator = init_generator
		self.generator = None
	# def make_generator(self):
	# 	return iter(self.init_generator)

	def make_generator(self):
		return (data for data in self.init_generator)
	def __next__(self, *args):
		try:
			return next(self.generator)
		except:
			self.generator = self.make_generator()
			return next(self.generator)
	def __iter__(self):
		return self.make_generator()

	def __getstate__(self):
		return {'init_generator': self.init_generator,
				'generator': None}
	def __setstate__(self, d ):
		self.init_generator = d['init_generator']
		self.generator = None


class DiffOpt(object):
	def __init__(self,func,generator,params,config_dict, 
						warm_start_iter,unrolled_iter,
						device, dtype,
						track_grad_for_backward=False):
		self.func = func
		self.generator = generator

		#scheduler = config_to_instance(**config_dict.scheduler)
		self.lr = config_dict.pop("lr", None)
		self.config_opt = config_dict
		self.optimizer = config_to_instance(**self.config_opt, lr = self.lr)
		self.opt_state = self.optimizer.init(params)		
		self.unrolled_iter = unrolled_iter
		self.warm_start_iter = warm_start_iter
		self.device = device 
		self.dtype= dtype
		self.track_grad_for_backward = track_grad_for_backward

		assert (self.warm_start_iter + self.unrolled_iter >0) 
	
	def update_lr(self,lr):
		self.optimizer = config_to_instance(**self.config_opt,lr=lr)
		
	def init_state_opt(self):
		# Detach all tensors to avoid backbropagating twice. 
		self.opt_state = tuple([utils.detach_states(state) for state in self.opt_state])


		
	def run(self,upper_var,lower_var):
		avg_val = 0.		
		cur_lower_var = lower_var
		all_lower_var = [cur_lower_var]
		track_grad = False
		total_iter = self.warm_start_iter + self.unrolled_iter
		
		self.init_state_opt()		

		for i in range(total_iter):
			inputs = next(self.generator)
			inputs = utils.to_device(inputs,self.device,self.dtype) 
			value = self.func(inputs,upper_var,cur_lower_var)
			if i>=self.warm_start_iter:
				track_grad = True

			if i == total_iter-1:
				all_grad = torch.autograd.grad(
											outputs=value, 
											inputs=upper_var+cur_lower_var, 
											retain_graph=self.track_grad_for_backward,
											create_graph=self.track_grad_for_backward,
											only_inputs=True,
											allow_unused=True)
				grad = all_grad[len(upper_var):]
			else:
				grad = torch.autograd.grad(
							outputs=value, 
							inputs=cur_lower_var, 
							retain_graph=track_grad,
							create_graph=track_grad,
							only_inputs=True,
							allow_unused=True)


			if  i<self.warm_start_iter:
				track_grad=False

			updates, self.opt_state = self.optimizer.update(grad, self.opt_state, inplace=not track_grad)
			cur_lower_var = TorchOpt.apply_updates(cur_lower_var, updates, inplace=not track_grad)
			
			if i>=self.warm_start_iter:
				all_lower_var.append(cur_lower_var)
			
			avg_val +=value.detach()

		avg_val = avg_val/total_iter

		return all_lower_var, avg_val,all_grad,inputs


class LinearSolver(object):
	def __init__(self,func,generator, config_dict, device,dtype):
		self.func = func
		self.generator = generator
		self.cur_generator = generator
		self.stochastic = config_dict.pop("stochastic", None)
		self.linear_solver_alg =config_to_instance(**config_dict.algorithm)
		self.residual_op = config_to_instance(**config_dict.residual_op, 
												hvp = self.hvp)
		self.device = device
		self.dtype = dtype
		
	def stochastic_mode(self):
		if self.stochastic:
			self.cur_generator = self.generator
		else:
			inputs = next(self.generator)
			self.cur_generator = cycle([inputs])

	def jvp(self,upper_var, lower_var,retain_graph=True, create_graph=True):

		inputs = next(self.cur_generator)
		inputs = utils.to_device(inputs,self.device,self.dtype)
		val = self.func(inputs,upper_var,lower_var)
		
		jac = autograd.grad(outputs=val, 
							inputs=lower_var, 
							grad_outputs=None, 
							retain_graph=retain_graph,
							create_graph=create_graph, 
							only_inputs=True,
							allow_unused=True)
		return jac
	def hvp(self,jac, diff_params, iterate, retain_graph=True):
		# if diff_params is None:
		# 	diff_params=lower_var 
		#jac = self.jvp(upper_var,lower_var)
		vhp = utils.grad_with_none(outputs=jac, 
			inputs=diff_params, 
			grad_outputs=tuple(iterate), 
			retain_graph=retain_graph,
			create_graph=False, 
			only_inputs=True,
			allow_unused=True)
		# if with_jac:
		# 	return vhp,jac
		return vhp
		
	def run(self,
			upper_var,lower_var,
			init,
			b_vector,jac,inputs):
		self.stochastic_mode()
		#res_op = partialmethod(self.residual_op.eval,upper_var=upper_var,lower_var=lower_var, b_vector=b_vector)
		with  torch.enable_grad():
			#jac = self.jvp(upper_var, lower_var)
			if isinstance(self.residual_op,FiniteDiffResidual):
				def res_op(iterate):
					return self.residual_op.eval(jac,self.func,inputs, upper_var,lower_var,b_vector, iterate)
			else:
				def res_op(iterate):
					return self.residual_op.eval(jac, upper_var,lower_var,b_vector, iterate)
			sol,out = self.linear_solver_alg(res_op,init)
		
		return out, sol

class ResidualOp(object):
	def __init__(self,hvp):
		self.hvp = hvp
	def eval(self,jac, upper_var,lower_var,b_vector, iterate):
		raise NotImplementedError


class FiniteDiffResidual(ResidualOp):
	def __init__(self,hvp,epsilon=0.01):
		super(FiniteDiffResidual,self).__init__(hvp)	
		self.epsilon = epsilon
	def eval(self,jac,func, inputs,upper_var,lower_var,b_vector, iterate):
		params = upper_var+lower_var
		norm = torch.cat([w.view(-1) for w in iterate]).norm()
		eps = self.epsilon / (norm.detach()+self.epsilon)

		## y + epsilon d
		lower_var_plus = tuple([p+eps* d if d is not None else p  for p,d in zip(lower_var,iterate)])

		val_plus =func(inputs,upper_var,lower_var_plus)
		grad_plus = torch.autograd.grad(
									outputs=val_plus, 
									inputs=params, 
									retain_graph=False,
									create_graph=False,
									only_inputs=True,
									allow_unused=True)

		# ## y - epsilon d


		grad_minus = jac
		hvp  = [(p-m)/(eps) for p,m in zip(grad_plus,grad_minus)]
		residual = hvp[len(upper_var):]
		out = hvp[:len(upper_var)]
		residual= [g+b if g is not None else b for g,b in zip(residual,b_vector)]
		return residual, out


class Residual(ResidualOp):
	# residual of the form res= Ax+b.
	def __init__(self,hvp):
		super(Residual,self).__init__(hvp)
	def eval(self,jac, upper_var,lower_var,b_vector, iterate):
		params = upper_var+lower_var
		lower_jac = jac[len(upper_var):]

		hvp  = self.hvp(lower_jac, params,iterate)
		residual = hvp[len(upper_var):]
		
		out = hvp[:len(upper_var)]
		residual= [g+b if g is not None else b for g,b in zip(residual,b_vector)]
		
		return residual, out

class NormalResidual(ResidualOp):
	# residual of the form res= A(Ax+b).
	def __init__(self,hvp):
		super(NormalResidual,self).__init__(hvp)
	def eval(self,jac,upper_var, lower_var,b_vector, iterate):
		params = upper_var+lower_var
		lower_jac = jac[len(upper_var):]
		hvp = self.hvp(lower_jac, params,iterate,retain_graph=True)
		residual = hvp[len(upper_var):]
		out = hvp[:len(upper_var)]
		residual = tuple([g+b if g is not None else b for g,b in zip(residual,b_vector)])
		residual = utils.grad_with_none(outputs=lower_jac, 
			inputs=lower_var, 
			grad_outputs=residual, 
			retain_graph=True,
			create_graph=False, 
			only_inputs=True,
			allow_unused=True)
		return residual,out

