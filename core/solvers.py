import  torch 
from torch.autograd import Variable
import torch.nn as nn
from torch import autograd
from core.utils import grad_with_none, grad_unused_zero
import core.utils as utils
from core.utils import Config
import higher
from torch.autograd import grad as torch_grad
import copy
import numpy as np

import time
from functools import partial
import functorch



class ForwardSolver(object):
	def __init__(self,optimizer=None,**ctx):
		self.ctx = Config(ctx)
		self.optimizer = optimizer
		self.counter = 0
		self.counter_grad = 0
	def run(self,func,generator,inner_params):
		n_iter = self.update_alg_params()
		avg_val = 0.
		if not self.ctx.warm_start:
			for p in inner_params:
				p.data = torch.zeros_like(p.data)
		with  torch.enable_grad():
			for i in range(n_iter):
				self.optimizer.zero_grad()
				inputs = next(generator)
				inputs = utils.to_device(inputs,generator.device,generator.dtype) 
				
				val = func(inputs)
				torch.autograd.backward(val, 
										retain_graph=False, 
										create_graph=False)
				self.counter_grad +=1
				self.optimizer.step()
				avg_val +=val.detach()
			#print(inner_params[0])
			avg_val = avg_val/n_iter
		return inner_params, None, avg_val
	
	def update_alg_params(self):
		self.counter = self.counter+1
		if 'increase_n_iter' in self.ctx and self.ctx.increase_n_iter:
			n_iter = max(self.ctx.n_iter, self.ctx.fac_increase*int(np.log(self.counter)))
		else:
			n_iter = self.ctx.n_iter
		return n_iter



class ForwardSolverSGD(ForwardSolver):
	def __init__(self,optimizer=None,**ctx):
		super(ForwardSolverSGD,self).__init__(optimizer=optimizer,**ctx)
		self.lr = self.optimizer.param_groups[0]['lr']
	def run(self,func,generator,inner_params):
		lr = self.lr
		n_iter = self.update_alg_params()
		avg_val = 0.
		if not self.ctx.warm_start:
			for p in inner_params:
				p.data = torch.zeros_like(p.data)
		with  torch.enable_grad():
			#start_time = time.time()
			for i in range(n_iter):
				
				for p in inner_params:
					if p.grad is not None:
						p.grad.zero_()
				
				#start_time = time.time()
				inputs = next(generator)
				#print('time iter: '+ str(time.time()-start_time))
				inputs = utils.to_device(inputs,generator.device,generator.dtype) 
				
				val = func(inputs)
				torch.autograd.backward(val, 
										retain_graph=False, 
										create_graph=False)
				avg_val +=val.detach()
				self.counter_grad +=1
				for p	in inner_params:
					if p.grad is not None:
						p.data = p.data -lr*p.grad
			#print(inner_params[0])
			avg_val = avg_val/n_iter
		return inner_params, None,avg_val


class UnrolledForwardSGD(ForwardSolver):
	def __init__(self, optimizer=None,**ctx):
		super(UnrolledForwardSGD,self).__init__(optimizer=optimizer, **ctx)
	def run(self,func,generator,inner_params):
		lr = self.optimizer.param_groups[0]['lr']
		n_iter = self.update_alg_params()
		avg_val = 0.
		outer_params = func.outer_params
		if not self.ctx.warm_start:
			for p in inner_params:
				p.data = torch.zeros_like(p.data)

		with  torch.enable_grad():
			fmodel = higher.patch.monkeypatch(func, device=func.device, copy_initial_weights=False, track_higher_grads=True)
			cur_inner_params = inner_params
			ys = [copy.deepcopy(inner_params)]
			for i in range(n_iter):
				inputs = next(generator)
				inputs = utils.to_device(inputs,generator.device,generator.dtype) 
				all_params = outer_params + cur_inner_params
				val = fmodel(inputs,params=all_params)
				jac = autograd.grad(outputs=val, 
									inputs=cur_inner_params, 
									grad_outputs=None, 
									retain_graph=True,
									create_graph=True, 
									only_inputs=True,
									allow_unused=True)
				avg_val +=val.detach()
				self.counter_grad +=1
				cur_inner_params = tuple([pp-lr*jj for pp, jj in zip(cur_inner_params,jac)])
				ys.append(cur_inner_params)
			avg_val = avg_val/n_iter
			for p, q	in zip(inner_params, cur_inner_params):
				p.data = 1.*q.data
		return cur_inner_params, ys, avg_val


class UnrolledForwardSolver(ForwardSolver):
	def __init__(self,optimizer=None,**ctx ):
		super(UnrolledForwardSolver,self).__init__(optimizer=optimizer, **ctx)

	def run(self,func,generator,inner_params):
		#lr = self.optimizer.param_groups[0]['lr']
		n_iter = self.update_alg_params()
		avg_val = 0.
		outer_params = func.outer_params
		if not self.ctx.warm_start:
			for p in inner_params:
				p.data = torch.zeros_like(p.data)

		with  torch.enable_grad():
			
			cur_inner_params = inner_params
			ys = [cur_inner_params]
			with higher.innerloop_ctx(func, self.optimizer) as (fmodel,diffopt):
				for i in range(n_iter):
					inputs = next(generator)
					inputs = utils.to_device(inputs,generator.device,generator.dtype) 
					all_params = outer_params + cur_inner_params
					
					val = fmodel(inputs,params=all_params)
					params = diffopt.step(val)
					avg_val +=val.detach()
					self.counter_grad +=1
					cur_inner_params = params[len(outer_params):]
					ys.append(cur_inner_params)
			avg_val = avg_val/n_iter
			for p, q	in zip(inner_params, cur_inner_params):
				p.data = 1.*q.data
		return cur_inner_params, ys, avg_val




class BackwardSolver(object):
	def __init__(self,**ctx):
		self.ctx = Config(ctx)
		self.counter=0
		self.counter_grad = 0
		self.counter_jac = 0
		self.counter_hess = 0

	def run(self,
			func,
			generator,
			outer_params,
			inner_params,
			iterates,
			amortized_grad,
			grad_output):
		return NotImplementedError('Not implemented')
	def compute_grads(self,
						func,
						generator,
						inner_params
						):
		with  torch.enable_grad():
			inputs = next(generator)
			inputs = utils.to_device(inputs,generator.device,generator.dtype) 
			val = func(inputs)
			
			jac = autograd.grad(outputs=val, 
								inputs=inner_params, 
								grad_outputs=None, 
								retain_graph=True,
								create_graph=True, 
								only_inputs=True,
								allow_unused=True)
			self.counter_grad +=1
			
		return jac


	def compute_second_order(self,jac, params, out_grad, param_type='inner',retain_graph=False):
		vhp = grad_with_none(outputs=jac, 
			inputs=params, 
			grad_outputs=out_grad, 
			retain_graph=retain_graph,
			create_graph=False, 
			only_inputs=True,
			allow_unused=True)
		if param_type=='inner':
			self.counter_hess+=1
		else:
			self.counter_jac+=1
		return vhp

	def update_alg_params(self):
		self.counter = self.counter+1
		if 'increase_n_iter' in self.ctx and self.ctx.increase_n_iter:
			n_iter = max(self.ctx.n_iter, self.ctx.fac_increase*int(np.log(self.counter)))
		else:
			n_iter = self.ctx.n_iter
		if 'decrease_lr' in self.ctx and self.ctx.decrease_lr:
			lr = self.ctx.lr*min(1., self.ctx.fac_decrease_lr/ np.power(self.counter, self.ctx.exp_decrease_lr)   )
		else:
			lr = self.ctx.lr
		return lr, n_iter



class BackwardSolverSGD(BackwardSolver):
	def __init__(self,**ctx):
		super(BackwardSolverSGD,self).__init__(**ctx)
	def hessian(self,func,generator,params, outer_params):
		inputs = next(generator)
		loss = func(inputs)
		return utils.hessian(loss,params[0])
		
	def run(self,
			func,
			generator,
			outer_params,
			inner_params,
			iterates,
			amortized_grad,
			grad_output):
		lr, n_iter = self.update_alg_params()
		
		if amortized_grad is None:
			out_amortized_grad = [torch.zeros_like(p) for p in inner_params]
		else:
			out_amortized_grad = [1.*g.data for g in amortized_grad]


		out = list(torch.zeros_like(g) for g in  outer_params)
		all_params = inner_params +outer_params
		for i in range(n_iter):
			with  torch.enable_grad():
				jac = self.compute_grads(func,
											generator,
											inner_params)
				vhp = self.compute_second_order(jac,inner_params,tuple(out_amortized_grad), param_type='inner')

			out_amortized_grad = [ ag - lr*(g+o) if g is not None else 1.*ag for ag,g,o in zip(out_amortized_grad,vhp,grad_output)]
				
		with  torch.enable_grad():
			jac = self.compute_grads(func,
										generator,
										inner_params)
			beta = self.compute_second_order(jac,outer_params,tuple(out_amortized_grad), param_type='outer')



		out = [o +b if b is not None else o for o,b in zip(out,beta) ]
		if self.ctx.warm_start:
			return tuple(out),tuple(out_amortized_grad)
		else:
			return tuple(out),amortized_grad 



class BackwardSolverSGD(BackwardSolver):
	def __init__(self,**ctx):
		super(BackwardSolverSGD,self).__init__(**ctx)
	def hessian(self,func,generator,params, outer_params):
		inputs = next(generator)
		loss = func(inputs)
		return utils.hessian(loss,params[0])
		
	def run(self,
			module,
			generator,
			outer_params,
			inner_params,
			iterates,
			amortized_grad,
			grad_output):
		lr, n_iter = self.update_alg_params()
		func, params, buffers = functorch.make_functional_with_buffers(module)
		def func_unfolded_params(inner_params, outer_params, buffers, inputs,func):
			params = outer_params + inner_params 
			return func(params,buffers,inputs)
		if amortized_grad is None:
			out_amortized_grad = [torch.zeros_like(p) for p in inner_params]
		else:
			out_amortized_grad = [1.*g.data for g in amortized_grad]
		out = list(torch.zeros_like(g) for g in  outer_params)
		
		all_params = inner_params +outer_params
		for i in range(n_iter):
			
			with  torch.enable_grad():
				inputs = next(generator)
				inputs = utils.to_device(inputs,generator.device,generator.dtype)
				#pfunc = partial(func_unfolded_params,outer_params=outer_params,inputs=inputs, functional=functional)
				
				# hvp = vjp_fn(*(tuple(out_amortized_grad),))

				# hvp = functorch.jvp(gfunc, (inner_params,), (tuple(out_amortized_grad),))
				#partial_func = partial(func_unfolded_params,outer_params=outer_params,inputs=inputs, buffers=buffers, func=func)
				#jac = functorch.grad(partial_func)(inner_params)
				#_, vjp_fn = vjp(gfunc, *(inner_params,))
				#hvp = vjp_fn(*(tuple(out_amortized_grad),))
				#vhp = utils.hvp_revrev(partial_func,(inner_params,),(tuple(out_amortized_grad),))	
				
				# vhp = utils.hvp(partial(partial_func,outer_params=(outer_params,),inputs=(inputs,), functional=functional),inner_params,tuple(out_amortized_grad))	
				jac = self.compute_grads(module,
											 generator,
											 inner_params)
				vhp = self.compute_second_order(jac,inner_params,tuple(out_amortized_grad), param_type='inner')

			out_amortized_grad = [ ag - lr*(g+o) if g is not None else 1.*ag for ag,g,o in zip(out_amortized_grad,vhp,grad_output)]
				
		with  torch.enable_grad():

			# jac = self.compute_grads(module,
			# 							generator,
			# 							inner_params)
			# beta = self.compute_second_order(jac,outer_params,tuple(out_amortized_grad), param_type='outer')
			inputs = next(generator)
			inputs = utils.to_device(inputs,generator.device,generator.dtype)
			partial_func = partial(func_unfolded_params,outer_params=outer_params,inputs=inputs, buffers=buffers, func=func)
			jac = functorch.grad(partial_func)(inner_params)

			# jac = self.compute_grads(module,
			# 							generator,
			# 							inner_params)
			beta = self.compute_second_order(jac,outer_params,tuple(out_amortized_grad), param_type='outer')




		out = [o +b if b is not None else o for o,b in zip(out,beta) ]
		if self.ctx.warm_start:
			return tuple(out),tuple(out_amortized_grad)
		else:
			return tuple(out),amortized_grad 

def func_unfolded_params(inner_params, outer_params, buffers, inputs,func):
	params = outer_params + inner_params 
	return func(params,buffers,inputs)

class BackwardSolverSGD(BackwardSolver):
	def __init__(self,module = None,**ctx):
		super(BackwardSolverSGD,self).__init__(**ctx)

		self.func, self.params, self.buffers = functorch.make_functional_with_buffers(module)

	def hessian(self,func,generator,params, outer_params):
		inputs = next(generator)
		loss = func(inputs)
		return utils.hessian(loss,params[0])
		
	def run(self,
			module,
			generator,
			outer_params,
			inner_params,
			iterates,
			amortized_grad,
			grad_output):
		lr, n_iter = self.update_alg_params()

		if amortized_grad is None:
			out_amortized_grad = [torch.zeros_like(p) for p in inner_params]
		else:
			out_amortized_grad = [1.*g.data for g in amortized_grad]
		out = list(torch.zeros_like(g) for g in  outer_params)
		
		all_params = inner_params +outer_params
		for i in range(n_iter):
			
			with  torch.enable_grad():
				# inputs = next(generator)
				# inputs = utils.to_device(inputs,generator.device,generator.dtype)
				# partial_func = partial(func_unfolded_params,outer_params=outer_params,inputs=inputs, buffers=self.buffers, func=self.func)
				# jac = functorch.grad(partial_func)(inner_params)
				
				jac = self.compute_grads(module,
							 generator,
							 inner_params)


				vhp = self.compute_second_order(jac,inner_params,tuple(out_amortized_grad), param_type='inner', retain_graph=True )
				
				## rev-forward hvp does not work yet with current beta version of functorch
				#vhp = functorch.jvp(functorch.grad(partial_func), (inner_params,), (tuple(out_amortized_grad),))[1]

				grad = tuple([g+o if g is not None else None for g, o in zip(vhp, grad_output) ])

				grad = self.compute_second_order(jac,inner_params,grad, param_type='inner', retain_graph=False)


			out_amortized_grad = [ ag - lr*g if g is not None else 1.*ag for ag,g in zip(out_amortized_grad,grad)]


		with  torch.enable_grad():


			# inputs = next(generator)
			# inputs = utils.to_device(inputs,generator.device,generator.dtype)
			# partial_func = partial(func_unfolded_params,outer_params=outer_params,inputs=inputs, buffers=self.buffers, func=self.func)
			# jac = functorch.grad(partial_func)(inner_params)
			jac = self.compute_grads(module,
										generator,
										inner_params)


			beta = self.compute_second_order(jac,outer_params,tuple(out_amortized_grad), param_type='outer')


		out = [o +b if b is not None else o for o,b in zip(out,beta) ]
		if self.ctx.warm_start:
			return tuple(out),tuple(out_amortized_grad)
		else:
			return tuple(out),amortized_grad 


class BackwardSolverApproxInv(BackwardSolver):
	def __init__(self,**ctx):
		super(BackwardSolverApproxInv,self).__init__(**ctx)

	def fp_map(self,func,generator,inner_params):
		jac = self.compute_grads(func,
								generator,
								inner_params)
		fmap = [p-self.ctx.lr*j if j is not None else p for p,j in zip(inner_params,jac)] 
		return fmap
	def run(self,
			func,
			generator,
			outer_params,
			inner_params,
			iterates,
			amortized_grad,
			grad_output):
		lr, n_iter = self.update_alg_params()
		out_amortized_grad = list(g.data for g in grad_output)
		out = list(torch.zeros_like(g) for g in  outer_params)
		all_params = inner_params +outer_params
		rd_n_inter = np.random.randint(0,n_iter)
		all_grads = [out_amortized_grad]
		for i in range(rd_n_inter):
			with  torch.enable_grad():
				# eval vhp
				jac = tuple(self.fp_map(func,generator,	inner_params))
				vhp = self.compute_second_order(jac,inner_params,tuple(out_amortized_grad), param_type='inner')

			out_amortized_grad = [g for g in vhp]

		with  torch.enable_grad():
			jac = tuple(self.fp_map(func,generator,	inner_params))
			beta = self.compute_second_order(jac,outer_params,tuple(out_amortized_grad), param_type='outer')
		out = [ o if b is None else o+n_iter*b for o, b in zip(out, beta) ]

		return tuple(out),None 

class BackwardSolverNeumann(BackwardSolver):
	def __init__(self,**ctx):
		super(BackwardSolverNeumann,self).__init__(**ctx)

	def fp_map(self,func,generator,inner_params,lr):
		jac = self.compute_grads(func,
								generator,
								inner_params)
		fmap = [p-lr*j if j is not None else p for p,j in zip(inner_params,jac)] 
		return fmap
	# def hessian(self,func,generator,params, outer_params):
	# 	fmodel = higher.patch.monkeypatch(func, device=func.device, copy_initial_weights=False, track_higher_grads=True)
	# 	def func_aux(params):
	# 		inputs = next(generator)
	# 		all_params = outer_params + tuple([params])
	# 		return fmodel(inputs,params=all_params)
	# 	return torch.autograd.functional.hessian(func_aux,params)

	def run(self,
			func,
			generator,
			outer_params,
			inner_params,
			iterates,
			amortized_grad,
			grad_output):
		lr, n_iter = self.update_alg_params()
		out_amortized_grad = [1.*g.data for g in grad_output]
		all_params = inner_params +outer_params
		all_grads = [ [1.*p] for p in out_amortized_grad]
		
		
		for i in range(n_iter):
			with  torch.enable_grad():
				# eval vhp
				jac = tuple(self.fp_map(func,generator,	inner_params,lr))
				vhp = self.compute_second_order(jac,inner_params,tuple(out_amortized_grad), param_type='inner')
			out_amortized_grad = [1.*g for g in vhp]
			# print('iter: '+str(i) +', err: '+ str(torch.norm(out_amortized_grad[0]).item()))
			# HH = self.hessian(func,generator,inner_params,outer_params)
			# U,S,V = torch.svd(HH[0][0])
			for p,o in zip(all_grads,out_amortized_grad):
				p.append(1.*o)
			#err = torch.stack([torch.norm(g) for g in vhp]).sum()
			#if err< self.ctx.tol:
			#	break
		
		all_grads = [torch.stack(p).sum(0) for p in all_grads]
		#print(all_grads)
		with  torch.enable_grad():
			jac = tuple(self.fp_map(func,generator,	inner_params,lr))
			beta = self.compute_second_order(jac,outer_params,tuple(all_grads), param_type='outer')
		out = list(torch.zeros_like(g) for g in  outer_params)
		out = [ o+b if b is not None else o for  b,o in zip(beta, out) ]
		return tuple(out),None 



class UnrolledBackwardSolver(BackwardSolver):
	def __init__(self,**ctx):
		super(UnrolledBackwardSolver,self).__init__(**ctx)



	def run(self,
			func,
			generator,
			outer_params,
			inner_params,
			iterates,
			amortized_grad,
			grad_output):
		#lr, n_iter = self.update_alg_params()
		with  torch.enable_grad():
			val = [ torch.einsum('...i,...i->',y,g) for y,g in zip(  iterates[-1], grad_output)]
			val = torch.sum(torch.stack(val))
			out = autograd.grad(outputs=val, 
								inputs=outer_params, 
								grad_outputs=None, 
								retain_graph=False,
								create_graph=False, 
								only_inputs=True,
								allow_unused=True)
			self.counter_hess += len(iterates)-2
			self.counter_jac += len(iterates)-1
		return tuple(out), None


class BackwardSolverFP(BackwardSolver):
	def __init__(self,**ctx):
		super(BackwardSolverFP,self).__init__(**ctx)
		
	def run(self,
			func,
			generator,
			outer_params,
			inner_params,
			iterates,
			amortized_grad,
			grad_output):
		lr, n_iter = self.update_alg_params()
		stochastic, tol = self.ctx.stochastic, self.ctx.tol
		with  torch.enable_grad():
			out = self.fixed_point(inner_params,
									outer_params, 
									n_iter, 
									func,
									generator, 
									grad_output, 
									tol=tol, 
									stochastic=stochastic)
		return tuple(out),None
	def fp_map(self,func,generator,inner_params):
		jac = self.compute_grads(func,
								generator,
								inner_params)
		fmap = [p-self.ctx.lr*j if j is not None else p for p,j in zip(inner_params,jac)] 
		return fmap
		
	# adapted from https://github.com/JunjieYang97/stocBiO/blob/master/Hyperparameter-optimization/hypergrad/hypergradients.py
	def fixed_point(self,
					params,
					hparams,
					K,
					func,
					generator,
					grad_out,
					tol=1e-10,
					set_grad=True,
					stochastic=False):
		"""
		Computes the hypergradient by applying K steps of the fixed point method (it can end earlier when tol is reached).

		Args:
				params: the output of the inner solver procedure.
				hparams: the outer variables (or hyperparameters), each element needs requires_grad=True
				K: the maximum number of fixed point iterations
				fp_map: the fixed point map which defines the inner problem 
				outer_loss: computes the outer objective taking parameters and hyperparameters as inputs
				tol: end the method earlier when  the normed difference between two iterates is less than tol
				set_grad: if True set t.grad to the hypergradient for every t in hparams
				stochastic: set this to True when fp_map is not a deterministic function of its inputs

		Returns:
				the list of hypergradients for each element in hparams
		"""

		#params = [w.detach().requires_grad_(True) for w in params]
		#o_loss = outer_loss(params, hparams)
		#grad_outer_w, grad_outer_hparams = get_outer_gradients(o_loss, params, hparams)

		if not stochastic:
				w_mapped = self.fp_map(func,generator,params)

		vs = [torch.zeros_like(w) for w in params]
		vs_vec = cat_list_to_tensor(vs)
		for k in range(K):
				vs_prev_vec = vs_vec

				if stochastic:
						w_mapped = self.fp_map(func,generator,params)
						vs = torch_grad(w_mapped, params, grad_outputs=vs, retain_graph=False)
				else:
						vs = torch_grad(w_mapped, params, grad_outputs=vs, retain_graph=True)
				self.counter_hess +=1
				vs = [v + gow for v, gow in zip(vs, grad_out)]
				vs_vec = cat_list_to_tensor(vs)
				if float(torch.norm(vs_vec - vs_prev_vec)) < tol:
						break

		if stochastic:
				w_mapped = self.fp_map(func,generator,params)

		grads = torch_grad(w_mapped, hparams, grad_outputs=vs, allow_unused=True)
		self.counter_jac +=1
		grads = [g if g is not None else torch.zeros_like(p) for p,g in zip(hparams,grads)]

		#grads = [g + v if g is not None else v for g, v in zip(grads, grad_outer_hparams)]

		#if set_grad:
		#    update_tensor_grads(hparams, grads)

		return grads


class BackwardSolverCG(BackwardSolver):
	def __init__(self,**ctx):
		super(BackwardSolverCG,self).__init__(**ctx)

	def run(self,
			func,
			generator,
			outer_params,
			inner_params,
			iterates,
			amortized_grad,
			grad_output):
		lr, n_iter = self.update_alg_params()
		stochastic, tol = self.ctx.stochastic, self.ctx.tol
		with  torch.enable_grad():
			return self.CG(inner_params,
							outer_params, 
							n_iter, 
							func,
							generator, 
							grad_output,
							amortized_grad,
							tol=tol,
							stochastic=stochastic)
	def fp_map(self,func,generator,inner_params):
		jac = self.compute_grads(func,generator,inner_params)
		fmap = [p-self.ctx.lr*j if j is not None else p for p,j in zip(inner_params,jac)] 
		return fmap

	# adapted from https://github.com/JunjieYang97/stocBiO/blob/master/Hyperparameter-optimization/hypergrad/hypergradients.py
	def CG(self,
			params,
			hparams,
			K,
			func,
			generator,
			grad_out,
			amortized_grad,
			tol=1e-10,
			set_grad=True,
			stochastic=False):
			"""
			 Computes the hypergradient by applying K steps of the conjugate gradient method (CG).
			 It can end earlier when tol is reached.

			 Args:
					 params: the output of the inner solver procedure.
					 hparams: the outer variables (or hyperparameters), each element needs requires_grad=True
					 K: the maximum number of conjugate gradient iterations
					 fp_map: the fixed point map which defines the inner problem
					 outer_loss: computes the outer objective taking parameters and hyperparameters as inputs
					 tol: end the method earlier when the norm of the residual is less than tol
					 set_grad: if True set t.grad to the hypergradient for every t in hparams
					 stochastic: set this to True when fp_map is not a deterministic function of its inputs

			 Returns:
					 the list of hypergradients for each element in hparams
			 """
			#params = [w.detach().requires_grad_(True) for w in params]
			if amortized_grad is None:
				out_amortized_grad = list(torch.zeros_like(p) for p in grad_out)
			else:
				out_amortized_grad = list(g.data for g in amortized_grad)


			if not stochastic:
					w_mapped = self.fp_map(func,generator,params)

			def dfp_map_dw(xs):
					if stochastic:
							w_mapped_in = self.fp_map(func,generator,params)
							Jfp_mapTv = torch_grad(w_mapped_in, params, grad_outputs=xs, retain_graph=False)
					else:
							Jfp_mapTv = torch_grad(w_mapped, params, grad_outputs=xs, retain_graph=True)
					return [v - j for v, j in zip(xs, Jfp_mapTv)]

			if self.ctx.warm_start:
				x = [o for o in out_amortized_grad]
			else:
				x = [torch.zeros_like(bb) for bb in grad_out]

			vs, counter_hess = cg(dfp_map_dw, grad_out, x, max_iter=K, epsilon=tol)  # K steps of conjugate gradient
			self.counter_hess += counter_hess
			out_amortized_grad = vs
			if stochastic:
					w_mapped = self.fp_map(func,generator,params)

			grads = torch_grad(w_mapped, hparams, grad_outputs=vs, allow_unused=True)
			grads = [g if g is not None else torch.zeros_like(p) for p,g in zip(hparams,grads)]
			self.counter_jac +=1
			if self.ctx.warm_start:
				return tuple(grads),tuple(out_amortized_grad)
			else:
				return tuple(grads),amortized_grad 



# adapted from https://github.com/JunjieYang97/stocBiO/blob/master/Hyperparameter-optimization/hypergrad/CG_torch.py
def cg(Ax, b, x, max_iter=100, epsilon=1.0e-5):
		""" Conjugate Gradient
			Args:
				Ax: function, takes list of tensors as input
				b: list of tensors
			Returns:
				x_star: list of tensors
		"""

		x_last = x
		init_Ax = Ax(x_last)
		r_last = [bb-ax for bb,ax in zip(b,init_Ax)]
		p_last = [torch.zeros_like(rr).copy_(rr) for rr in r_last]
		counter_hess = 1
		for ii in range(max_iter):
			Ap = Ax(p_last)
			counter_hess +=1
			Ap_vec = cat_list_to_tensor(Ap)
			p_last_vec = cat_list_to_tensor(p_last)
			r_last_vec = cat_list_to_tensor(r_last)
			rTr = torch.sum(r_last_vec * r_last_vec)
			pAp = torch.sum(p_last_vec * Ap_vec)
			alpha = rTr / pAp

			x = [xx + alpha * pp for xx, pp in zip(x_last, p_last)]
			r = [rr - alpha * pp for rr, pp in zip(r_last, Ap)]
			r_vec = cat_list_to_tensor(r)

			if float(torch.norm(r_vec)) < epsilon:
				break

			beta = torch.sum(r_vec * r_vec) / rTr
			p = [rr + beta * pp for rr, pp in zip(r, p_last)]

			x_last = x
			p_last = p
			r_last = r

		return x_last,counter_hess


def cat_list_to_tensor(list_tx):
		return torch.cat([xx.view([-1]) for xx in list_tx])

