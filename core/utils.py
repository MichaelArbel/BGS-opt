import  torch 
from torch.autograd import Variable
import torch.nn as nn
from torch import autograd
import time
import numpy as np
import omegaconf

from functorch import jvp, grad, vjp




class AttrList(object):
	def __init__(self,module,attr_list,tag):
		self.module = module
		self.tag = tag
		for i,w in enumerate(attr_list):
			setattr(module,tag+str(i),w)
		self.num_attributes = len(attr_list)
	def __getitem__(self,key):
		return getattr(self.module,self.tag+str(key))
	def __iter__(self):
		return iter(self.__getitem__(i) for i in range(self.num_attributes))




def detach_states(state_tuples):
	fields = state_tuples._fields
	key_values = {}
	for field in fields:
		state_tuple = eval('state_tuples.'+field)
		try:
			if type(state_tuple[0])==torch.Tensor:
				new_state_tuple = tuple([state.data for state in state_tuple])
				key_values[field] = new_state_tuple
		except:
			pass
	return state_tuples._replace(**key_values)
			
def hvp(f, primals, tangents):
  return jvp(grad(f), primals, tangents)[1]

def hvp_revrev(f, primals, tangents):
	_, vjp_fn = vjp(grad(f), *primals)
	return vjp_fn(*tangents)[0]

# class Config(dict):
#     def __getattr__(self, name):
#         return self[name]
class Config(dict):
	def __init__(self, *args, **kwargs):
		super(Config, self).__init__(*args, **kwargs)
		self.__dict__ = self

def config_to_dict(config):
	done = False
	out_dict = {}
	for key, value in config.items():
		if isinstance(value, omegaconf.dictconfig.DictConfig):
			out_dict[key] = config_to_dict(value)
		else:
			out_dict[key] = value
	return Config(out_dict)


def jacobian(y, x, create_graph=False):                                                               
	jac = []                                                                                          
	flat_y = y.reshape(-1)                                                                            
	grad_y = torch.zeros_like(flat_y)                                                                 
	for i in range(len(flat_y)):                                                                      
		grad_y[i] = 1.                                                                                
		grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
		jac.append(grad_x.reshape(x.shape))                                                           
		grad_y[i] = 0.                                                                                
	return torch.stack(jac).reshape(y.shape + x.shape)                                                
																									  
def hessian(y, x):                                                                                    
	return jacobian(jacobian(y, x, create_graph=True), x) 

def cond_loss(loss,dataloader,params):
	inputs = next(dataloader)
	loss = loss(inputs)
	u,v = params.shape
	H = hessian(loss,params)

	H = H.reshape(u*v, u*v).detach()
	cond = torch.linalg.cond(H,p=2)
	return cond

def one_hot(indices, depth):
	"""
	Returns a one-hot tensor.
	This is a PyTorch equivalent of Tensorflow's tf.one_hot.
		
	Parameters:
	  indices:  a (n_batch, m) Tensor or (m) Tensor.
	  depth: a scalar. Represents the depth of the one hot dimension.
	Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
	"""

	encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
	index = indices.view(indices.size()+torch.Size([1]))
	encoded_indicies = encoded_indicies.scatter_(1,index,1)
	
	return encoded_indicies

def grad_with_none(outputs,inputs, grad_outputs=None,retain_graph=False,create_graph=False, only_inputs=False,allow_unused=False):
	# Inspired from _autograd_grad in https://pytorch.org/docs/stable/_modules/torch/autograd/functional.html#vhp
	assert isinstance(outputs, tuple)
	if grad_outputs is None:
		grad_outputs = (None,) * len(outputs)
	assert isinstance(grad_outputs, tuple)
	assert len(outputs) == len(grad_outputs)
	new_outputs: Tuple[torch.Tensor, ...] = tuple()
	new_grad_outputs: Tuple[torch.Tensor, ...] = tuple()
	for out, grad_out in zip(outputs,grad_outputs):
		if out is not None and out.requires_grad:
			new_outputs +=(out,)
			new_grad_outputs +=(grad_out,)
	if len(new_outputs) == 0:
		return (None,) * len(inputs)
	else:
		return  autograd.grad(outputs=new_outputs, inputs=inputs, grad_outputs=new_grad_outputs, retain_graph=retain_graph,create_graph=create_graph, only_inputs=only_inputs,allow_unused=allow_unused) 


def grad_unused_zero(output, inputs, grad_outputs=None, retain_graph=False, create_graph=False):
	grads = torch.autograd.grad(output, inputs, grad_outputs=grad_outputs, allow_unused=True,
								retain_graph=retain_graph, create_graph=create_graph)

	def grad_or_zeros(grad, var):
		return torch.zeros_like(var) if grad is None else grad

	return tuple(grad_or_zeros(g, v) for g, v in zip(grads, inputs))

# def to_device(data,device, dtype):
# 	if isinstance(data,list):
# 		data = tuple(data)
# 	if type(data) is tuple:

# 		data = tuple([ to_type(d.to(device),dtype) if isinstance(d,torch.Tensor) else d for d in data ])
# 		#for d in data:
# 		#	if not d.is_sparse:
# 		#		d.requires_grad = True
# 		if len(data)==1:
# 			data = data[0]
# 	elif isinstance(data,torch.Tensor):
# 		data = to_type(data.to(device),dtype)
# 		#if not data.is_sparse:
# 		#	data.requires_grad = True
# 	else:
# 		data = to_type(data,dtype)
# 		#if not data.is_sparse:
# 		#	data.requires_grad = True
# 	return data

def to_device(data,device, dtype):
	if isinstance(data,list):
		data = tuple(data)
	if type(data) is tuple:
		data = tuple([ to_device(d,device,dtype) for d in data ])
		return data
	elif isinstance(data,torch.Tensor):
		return to_type(data.to(device),dtype)
	elif isinstance(data,int):
		return torch.tensor(data).to(device)
	else:
		raise NotImplementedError('unknown type')


def to_type(data,dtype):
	if dtype==torch.double:
		return data.double()
	else:
		return data.float()

def zero_grad(params):
	for p in params:
		if p.grad is not None:
			p.grad.zero_()




def add_prefix_to_keys_dict(dico,prefix):
	keys = list(dico.keys())
	for key in keys:
		dico[prefix+key] = dico.pop(key)


def eval_avg_time_grads(upper_loader,
						lower_loader, 
						upper_loss,
						lower_loss,
						device,
						dtype):
	out_dict = { 'cost_upper_grad': 0,
				'cost_lower_grad': 0,
				'cost_lower_hess': 0,
				'cost_lower_jac': 0 }
	K = 10
	lower_params = lower_loss.lower_params
	upper_params = upper_loss.upper_params
	for k in range(K):
		data = next(upper_loader)
		data = to_device(data,device,dtype)
		
		time_1 = time.time()
		loss = upper_loss(data)
		grad_upper = autograd.grad(outputs=loss,inputs=lower_params+upper_params,allow_unused=True)

		time_11 = time.time()
		data = next(lower_loader)
		data = to_device(data,device,dtype)
		
		time_2 = time.time()
		loss = lower_loss(data)
		grad_lower = autograd.grad(outputs=loss, 
							inputs=lower_params,
							grad_outputs=None, 
							retain_graph=True,
							create_graph=True, 
							only_inputs=True,
							allow_unused=True)


		#grad_lower = autograd.grad(outputs=loss,inputs=lower_params, create_graph=True)

		time_3 = time.time()
		hess = grad_with_none(outputs=grad_lower, 
			inputs=lower_params, 
			grad_outputs=grad_upper[:len(lower_params)], 
			retain_graph=True,
			create_graph=False, 
			only_inputs=True,
			allow_unused=True)


		time_4 = time.time()
		hess = grad_with_none(outputs=grad_lower, 
			inputs=upper_params, 
			grad_outputs=grad_upper[:len(lower_params)], 
			retain_graph=False,
			create_graph=False, 
			only_inputs=True,
			allow_unused=True)

		time_5 = time.time()

		out_dict['cost_upper_grad'] += (time_11-time_1)/K
		out_dict['cost_lower_grad'] += (time_3-time_2)/K
		out_dict['cost_lower_hess'] += (time_4-time_3)/K
		out_dict['cost_lower_jac'] += (time_5-time_4)/K
	return out_dict

def accuracy(predictions, targets):
	predictions = predictions.argmax(dim=1).view(targets.shape)
	return 100.*(predictions == targets).sum().float() / targets.size(0)


def fast_adapt(train_batch,
				val_batch,
				learner,
				features,
				loss,
				reg_lambda,
				adaptation_steps,
				shots,
				ways,
				device=None):

	data, labels, index = train_batch
	data, labels = data.to(device), labels.to(device)
	labels = labels.long()
	data = features(data)

	for step in range(adaptation_steps):
		l2_reg = 0
		for p in learner.parameters():
			l2_reg += p.norm(2)
		predictions = learner(data)
		train_error = loss(predictions, labels) + reg_lambda*l2_reg 
	
		learner.adapt(train_error)
	train_accuracy = accuracy(predictions, labels)
	data, labels, index = val_batch
	labels = labels.long()
	data = features(data)
	predictions = learner(data)
	valid_error = loss(predictions, labels)
	valid_accuracy = accuracy(predictions, labels)
	return valid_error, valid_accuracy, train_error, train_accuracy

class Lambda(nn.Module):

	def __init__(self, fn):
		super(Lambda, self).__init__()
		self.fn = fn

	def forward(self, x):
		return self.fn(x)


