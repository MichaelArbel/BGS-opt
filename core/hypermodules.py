import  torch 
from torch.autograd import Variable
import torch.nn as nn
from torch import autograd
import pdb

from core import utils


class HyperLoss(nn.Module):
	def __init__(self, 
				outer_loss,
				inner_loss,
				inner_loader,
				forward_solver,
				backward_solver 
				):
		super(HyperLoss,self).__init__()
		self.func = outer_loss
		self.argmin = ArgMin(inner_loss, 
							inner_loader, 
							forward_solver, 
							backward_solver)
		self.inner_params = outer_loss.inner_params
		self.outer_params = outer_loss.outer_params
		self.counter_outer_grad = 0
	def get_grad_counts(self):
		counter_inner_grad = self.argmin.forward_solver.counter_grad + self.argmin.backward_solver.counter_grad
		counts = {'inner_grad':  counter_inner_grad,
					'inner_hess': self.argmin.backward_solver.counter_hess,
					'inner_jac': self.argmin.backward_solver.counter_jac,
					'outer_grad': self.counter_outer_grad}
		return counts

	def eval_inner_loss(self, generator,total=False, max_iter=10):
		return self.argmin.eval_loss(generator,total=total, max_iter=max_iter)
	def forward_opt(self,inputs):
		with  torch.enable_grad():
			inner_params_out = self.argmin(*self.outer_params)
			loss = self.func(inputs)
			#if loss<0.:
			#	print('bug')
			#	loss = self.func(inputs)
		return loss,inner_params_out
	def backward_opt(self,
					loss,
					inputs,
					grad_output,
					inner_params_out):
		with  torch.enable_grad():
			
			for p in self.outer_params:
				if p.grad is not None:
					p.grad.zero_()
			for p in self.inner_params:
				if p.grad is not None:
					p.grad.zero_()
			# input_grad = autograd.grad(outputs=loss, 
			# 						inputs=inputs, 
			# 						grad_outputs=grad_output, 
			# 						retain_graph=True,
			# 						create_graph=True, 
			# 						only_inputs=True)[0]

			# input_grad = autograd.grad(outputs=loss, 
			# 						inputs=inputs, 
			# 						grad_outputs=grad_output, 
			# 						retain_graph=True,
			# 						create_graph=True, 
			# 						only_inputs=True)[0]

			#input_grad = autograd.grad(outputs=loss, inputs=self.inner_params+self.outer_params, grad_outputs=grad_output, retain_graph=True,create_graph=False, only_inputs=True)


			torch.autograd.backward(loss, 
									retain_graph=True, 
									create_graph=False, inputs=self.inner_params+self.outer_params)
			self.counter_outer_grad +=1
			for p in self.inner_params:
				if p.grad is None:
						p.grad = torch.zeros_like(p)

			inner_grads = [p.grad for p in self.inner_params]

			torch.autograd.backward(inner_params_out, grad_tensors=inner_grads, retain_graph=False, create_graph=False,
									inputs = self.inner_params+self.outer_params)
		
		#out = inputss + 2 
		#if inputs.requires_grad:
		#	return inputs.grad
		#else:
		#	return None
	def forward(self,inputs):
		#y_star =  self.argmin(*self.outer_params)
		#out = self.func.func(y_star[0], self.outer_params[0])
		with  torch.enable_grad():
			out = HyperLossOp.apply(inputs,self.inner_params,self.outer_params,self)
		#if not out.requires_grad:
		#	out.requires_grad=True
		return out

	

class HyperLossOp(torch.autograd.Function):
	@staticmethod
	def forward(ctx,inputs,inner_params,outer_params,hyperloss):
		ctx.hyperloss= hyperloss
		loss,inner_params_out = hyperloss.forward_opt(inputs)
		ctx.save_for_backward(loss,inputs, *inner_params_out)
		val_loss = loss.detach()
		return val_loss

	@staticmethod
	def backward(ctx, grad_output):
		hyperloss = ctx.hyperloss
		loss = ctx.saved_tensors[0]
		inputs = ctx.saved_tensors[1]
		inner_params_out = ctx.saved_tensors[2:]
		hyperloss.backward_opt(loss,
								inputs,
								grad_output,
								inner_params_out)
		
		if isinstance(inputs,tuple):
			grads = tuple([p.grad for p in inputs])
		elif isinstance(inputs,torch.Tensor):
			grads = inputs.grad
		else:
			grads = None
		return grads,None,None,None

class ArgMin(nn.Module):
	def __init__(self, 
				inner_loss, 
				generator, 
				forward_solver, 
				backward_solver):
		super(ArgMin,self).__init__()
		self.func = inner_loss
		self.inner_params = inner_loss.inner_params
		self.outer_params = inner_loss.outer_params
		self.generator = generator
		self.forward_solver = forward_solver
		self.backward_solver = backward_solver
		self.amortized_grad = None
	def eval_loss(self,generator, total=False, max_iter=10):
		if total:
			loss = 0.
			count = 0
			for index, inputs in enumerate(generator):
				if index>max_iter and max_iter>0:
					break
				inputs = utils.to_device(inputs,generator.device,generator.dtype)
				loss= loss + self.func(inputs)
				count+=1
			return loss/count
		else:
			inputs = next(generator)
			inputs = utils.to_device(inputs,generator.device,generator.dtype)
			return self.func(inputs)
	def forward_opt(self,inner_params):
		out, iterates = self.forward_solver.run(self.func,
									self.generator,
									inner_params)
		return out, iterates

	def backward_opt(self,
					outer_params,
					inner_params,
					iterates,
					grad_output):
		out, self.amortized_grad =  self.backward_solver.run(self.func,
								self.generator,
								outer_params,
								inner_params,
								iterates,
								self.amortized_grad,
								grad_output)
		return out

	def forward(self,*outer_params):
		all_params = self.inner_params+outer_params
		len_inner = len(self.inner_params)
		inner_params_out =  ArgMinOp.apply(self,len_inner,*all_params)
		
		return  inner_params_out


class ArgMinOp(torch.autograd.Function):

	@staticmethod
	def forward(ctx,propagator, len_inner,*all_params):
		ctx.propagator = propagator
		ctx.len_inner = len_inner
		inner_params = all_params[:len_inner]
		outer_params = all_params[len_inner:]
		inner_params_out, iterates = propagator.forward_opt(inner_params)
		ctx.iterates = iterates
		ctx.save_for_backward(*all_params)
		
		return tuple( p.detach() for p in inner_params_out)

	@staticmethod
	def backward(ctx, *grad_output):
		propagator = ctx.propagator
		len_inner = ctx.len_inner
		iterates =ctx.iterates
		#inner_params_out =  ctx.saved_tensors[:len_inner]
		outer_params = ctx.saved_tensors[len_inner:]
		inner_params = ctx.saved_tensors[:len_inner]
		
		grad_inner_params = propagator.backward_opt(outer_params,
													inner_params,
													iterates,
													grad_output)

		return (None,)*(len(inner_params)+2) + grad_inner_params



