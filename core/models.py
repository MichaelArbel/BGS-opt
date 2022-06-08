import  torch 
from torch.autograd import Variable
import torch.nn as nn
from torch import autograd
import pdb
import higher
import copy
import learn2learn as l2l


class Identity(nn.Module):
	def __init__(self,dim, init):
		super(Identity,self).__init__()
		self.param_x = torch.nn.parameter.Parameter(init*torch.ones([dim]))
	def forward(self,input):
		return inputs

class ModelDataset(nn.Module):
	def __init__(self,shape):
		super(ModelDataset,self).__init__()
		data = torch.distributions.normal.Normal(loc=0., scale=1.).sample(shape)
		self.x = torch.nn.parameter.Parameter(data)
		#self.x = torch.nn.parameter.Parameter(torch.zeros(shape))
		self.reg = torch.nn.parameter.Parameter(torch.zeros(shape[1:]).view(-1))
		num_label = shape[0]
		self.y = torch.nn.parameter.Parameter(torch.tensor(list(range(num_label))).float())
	def forward(self,input):
		return inputs

class Linear(nn.Module):
	def __init__(self,n_features,n_classes,with_bias=False):
		super(Linear,self).__init__()
		#data = torch.distributions.normal.Normal(loc=0., scale=1.).sample([n_features,n_classes])
		self.weight = torch.nn.parameter.Parameter(torch.zeros([n_features,n_classes]))
		if with_bias:
			self.bias = torch.nn.parameter.Parameter(torch.zeros([n_classes]))
		else:
			self. bias = 0.
	def forward(self,inputs):
		return inputs @ self.weight + self.bias



class MultiModel(nn.Module):
	def __init__(self,base_model,num_copies):
		super(MultiModel,self).__init__()
		base_model = base_model.to('cpu')
		params = tuple(base_model.parameters())
		#copy_params = [None]*num_copies
		copy_params = []
		#self.vv = [torch.nn.parameter.Parameter(torch.empty_like(p).copy_(p.data)) for p in params]

		for i in range(num_copies):
			new_params = []
			for j,p in enumerate(params):
				new_p = torch.nn.parameter.Parameter(torch.empty_like(p).copy_(p.data))
				self.register_parameter('cp_'+str(i)+'_idx_'+str(j),new_p)
				new_params.append(new_p)
			copy_params.append(tuple(new_params))

		self.params = copy_params
		self.fmodel = higher.patch.monkeypatch(base_model, device=params[0].device, copy_initial_weights=False, track_higher_grads=True)


		
	def forward(self,data):
		x,index = data
		#return x@self.params[index][0] + self.params[index][1]
		param = tuple([ p.to(x.device) for p in self.params[index]])
		return self.fmodel(x, params =  param)
	def norm_l2(self,index, device='cpu'):
		out = 0.
		param = tuple([ p.to(device) for p in self.params[index]])
		for p in param:
			out += p.norm(2)

		return out


class MultiModel(nn.Module):
	def __init__(self,base_model,num_copies):
		super(MultiModel,self).__init__()
		self.base_model = base_model.to('cpu')
		params = tuple(base_model.parameters())
		#copy_params = [None]*num_copies
		copy_model = []
		#self.vv = [torch.nn.parameter.Parameter(torch.empty_like(p).copy_(p.data)) for p in params]

		for i in range(num_copies):
			copy_model.append(copy.deepcopy(base_model))

		self.copy_model = copy_model
		#self.fmodel = higher.patch.monkeypatch(base_model, device=params[0].device, copy_initial_weights=False, track_higher_grads=True)


		
	def forward(self,data):
		x,index = data
		#return x@self.params[index][0] + self.params[index][1]
		#param = tuple([ p.to(x.device) for p in self.params[index]])
		model = self.copy_model[index].to(x.device)
		#model = self.copy_model[index]

		return model(x)
	def norm_l2(self,index, device='cpu'):
		model = self.copy_model[index].to(device)
		#model = self.copy_model[index]
		out = 0.
		param = list(model.parameters())
		for p in param:
			out += torch.sum(p**2)
		del model
		return out



class SingleModel(nn.Module):
	def __init__(self,base_model):
		super(SingleModel,self).__init__()
		self.base_model = base_model
		self.params = tuple(base_model.parameters())

	def forward(self,data):
		#self.model = copy.deepcopy(self.base_model)
		x,index = data
		#return x@self.params[index][0] + self.params[index][1]
		return self.base_model(x)

	def norm_l2(self,index,device='cpu'):
		#out = torch.sum(params[0]**2)
		out = 0.
		for p in self.params:
			out += torch.sum(p**2)
		return out

class Lambda(nn.Module):

    def __init__(self, fn):
        super(Lambda, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class ConvBase(nn.Module):
	def __init__(self,channels, out_dim, max_pool=True,n_tasks=20000):
		super(ConvBase,self).__init__()
		features = l2l.vision.models.ConvBase( channels=channels, max_pool=max_pool)
		self.features = torch.nn.Sequential(features, Lambda(lambda x: x.view(-1, out_dim)))
		#self.reg_params = -torch.ones([n_tasks])
		self.reg_params = torch.nn.parameter.Parameter(-1.*torch.ones([n_tasks]))
	def forward(self,data):
		return self.features(data)




