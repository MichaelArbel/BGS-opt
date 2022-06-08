import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import core.utils as utils

class Loss(nn.Module):
	def __init__(self,outer_model,inner_model,device=None):
		super(Loss,self).__init__()
		self.outer_model = outer_model
		self.inner_model = inner_model
		self.device = device
		self.recompute_features = True
		self.features = None
		if  isinstance(inner_model,nn.Module):
			self.inner_params = tuple(inner_model.parameters())
		elif isinstance(inner_model,list) and isinstance(inner_model[0],nn.Module):
			self.inner_params = [ tuple(m.parameters()) for m in inner_model ]
		else:
			self.inner_params = inner_params
		if  isinstance(outer_model,nn.Module):
			self.outer_params = tuple(outer_model.parameters())
		elif isinstance(outer_model,list) and isinstance(outer_model[0],nn.Module):
			self.outer_params = [ tuple(m.parameters()) for m in outer_params ]
		else:
			self.outer_params = outer_model
	def forward(self,inputs):
		raise NotImplementedError('Abstract class')  


# class MetaLoss(Loss):
# 	def __init__(self,outer_model,inner_model,num_tasks,device=None):
# 		super(MetaLoss,self).__init__(outer_model,inner_model,device=device)
# 		self.num_tasks = num_tasks
# 		inner_params = [None]*num_tasks
# 		for i in range(num_tasks):
# 			inner_params[i] = tuple([torch.nn.parameter.Parameter(p.clone()) for p in self.inner_params])    
			

class QuadyLinx(Loss):
	def __init__(self,outer_model,inner_model,device=None,cond=.1, swap = False,params_system=None):
		super(QuadyLinx,self).__init__(outer_model,inner_model, device=device)
		self.cond = cond 
		if swap:
			self.y = self.outer_params[0]
			self.x = self.inner_params[0]
		else:
			self.x = self.outer_params[0]
			self.y = self.inner_params[0]
		N = self.x.shape[0]
		d = self.y.shape[0]
		A = torch.randn(d,d, dtype=self.y.dtype, device=self.y.device)
		self.A = A@A.t()
		U,S,V = torch.svd(self.A)
		S[-1] = 0.



		lmbda = S[0]/(self.cond-1)
		
		#lmbda = max(0,lmbda)
		S = S+lmbda
		S = S/S[0]
		self.A = torch.einsum('ij,j,kj->ik',U,S,V)

		if swap:
			self.B = torch.zeros([d,N],dtype=self.y.dtype, device=self.y.device)
			self.C = torch.randn([N],dtype=self.y.dtype, device=self.y.device)
			if params_system is None: 
				self.x_star = None
				self.D = torch.zeros([1],dtype=self.y.dtype, device=self.y.device)
			else:
				self.A_y,self.B_y,_,_ = params_system
				v = self.C@torch.inverse(self.A_y)@self.B_y
				self.x_star = torch.inverse(self.A)@v
				self.D = 0.5*v.t()@torch.inverse(self.A)@v
		else:
			self.B = torch.randn([d,N],dtype=self.y.dtype, device=self.y.device)
			self.C = torch.zeros([N],dtype=self.y.dtype, device=self.y.device)
			self.D = torch.zeros([1],dtype=self.y.dtype, device=self.y.device)
			self.x_star = None


	def get_param_system(self):
		return self.A, self.B, self.C, self.D
	def forward(self,inputs):

		loss = 0.*torch.mean(inputs)  \
				+.5*self.y.t()@self.A@self.y \
				+ self.y.t()@self.B@self.x \
				+ self.C@self.x +self.D

		#loss = 0.*torch.mean(inputs) +.5*self.y.t()@self.A@self.y + self.y.t()@self.B@self.x + self.C@self.x +self.D

		return loss.mean()
	def func(self,x,y):
		if self.x_star is None:
			loss = .5*y.t()@self.A@y \
					+ y.t()@self.B@x \
					+ self.C@x +self.D
		else:
			vv = y-self.x_star
			loss = 0.5*vv.T@self.A@vv
			#print('x: '+str(y) + ', x_star: '+ str(self.x_star))
		return loss 



class QuadyLinxSto(Loss):
	def __init__(self,outer_model,inner_model,device=None,cond=.1, sigma=0.1, swap = False,params_system=None):
		super(QuadyLinxSto,self).__init__(outer_model,inner_model, device=device)
		self.cond = cond 
		self.sigma = sigma
		if swap:
			self.y = self.outer_params[0]
			self.x = self.inner_params[0]
			self.sigma_B = 0
			self.sigma_C = 1 
		else:
			self.x = self.outer_params[0]
			self.y = self.inner_params[0]
			self.sigma_C = 0
			self.sigma_B = 1 
		N = self.x.shape[0]
		d = self.y.shape[0]
		A = torch.randn(d,d, dtype=self.y.dtype, device=self.y.device)
		self.A = A@A.t()
		U,S,V = torch.svd(self.A)
		S[-1] = 0.



		lmbda = S[0]/(self.cond-1)
		
		#lmbda = max(0,lmbda)
		S = S+lmbda
		S = S/S[0]
		self.A = torch.einsum('ij,j,kj->ik',U,S,V)
		if swap:
			self.B = torch.zeros([d,N],dtype=self.y.dtype, device=self.y.device)
			self.C = torch.randn([N],dtype=self.y.dtype, device=self.y.device)
			if params_system is None: 
				self.x_star = None
				self.D = torch.zeros([1],dtype=self.y.dtype, device=self.y.device)
			else:
				self.A_y,self.B_y,_,_ = params_system
				v = self.C@torch.inverse(self.A_y)@self.B_y
				self.x_star = torch.inverse(self.A)@v
				self.D = 0.5*v.t()@torch.inverse(self.A)@v
		else:
			self.B = torch.randn([d,N],dtype=self.y.dtype, device=self.y.device)
			self.C = torch.zeros([N],dtype=self.y.dtype, device=self.y.device)
			self.D = torch.zeros([1],dtype=self.y.dtype, device=self.y.device)
			self.x_star = None

	def gene_noisy_param(self):
		d,N = self.B.shape
		A = self.A + self.sigma*torch.randn(d,d, dtype=self.y.dtype, device=self.y.device)
		B = self.B + self.sigma_B*self.sigma*torch.randn([d,N],dtype=self.y.dtype, device=self.y.device)
		C = self.C + self.sigma_C*self.sigma*torch.randn([N],dtype=self.y.dtype, device=self.y.device)
		return A,B,C

	def get_param_system(self):
		return self.A, self.B, self.C, self.D
	def forward(self,inputs):
		A,B,C = self.gene_noisy_param()
		loss = 0.*torch.mean(inputs)  \
				+.5*self.y.t()@A@self.y \
				+ self.y.t()@B@self.x \
				+ C@self.x +self.D

		#loss = 0.*torch.mean(inputs) +.5*self.y.t()@self.A@self.y + self.y.t()@self.B@self.x + self.C@self.x +self.D

		return loss.mean()
	def func(self,x,y):
		if self.x_star is None:
			loss = .5*y.t()@self.A@y \
					+ y.t()@self.B@x \
					+ self.C@x +self.D
		else:
			vv = y-self.x_star
			loss = 0.5*vv.T@self.A@vv
			#print('x: '+str(y) + ', x_star: '+ str(self.x_star))
		return loss 




class Logistic(Loss):
	def __init__(self,outer_model,inner_model,reg=False, device=None):
		super(Logistic,self).__init__(outer_model,inner_model,device=device)
		self.reg = reg
	def forward(self,data,with_acc=False):
		x,y = data
		y = y.long()

		out_x = self.inner_model(x)
		out = F.cross_entropy(out_x, y)
		if self.reg:
			out =  out +  self.reg_term()

		if with_acc:
			pred = out_x.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
			acc = pred.eq(y.view_as(pred)).sum().item() / len(y)
			return out,acc
		else:
			return out

	def reg_term(self):
		ones_dxc = torch.ones(self.inner_params[0].size()).to(self.inner_params[0].device)
		out =  0.5*((self.inner_params[0]**2)*torch.exp(self.outer_params[0].unsqueeze(1)*ones_dxc)).mean()
		if len(self.outer_params)>1:
			out = out+ (self.inner_params[0].abs() * torch.exp(self.outer_params[1].unsqueeze(1) * ones_dxc)).mean()
		return out

class LogisticDistill(Loss):
	def __init__(self,outer_model,inner_model,is_inner=False, reg= 1., device=None):
		super(LogisticDistill,self).__init__(outer_model,inner_model,device=device)
		self.reg = False
		self.reg_param=reg
		#self.dim ,self.n_classes  = self.inner_params[0].shape()
		self.is_inner = is_inner
		if self.is_inner:
			self.reg = True
	def forward(self,data,with_acc=False):
		
		if self.is_inner:
			x, y = self.outer_model.x, self.outer_model.y.data
		else:
			x,y = data
		#x,y = data
		y = y.long()
		out_x = self.inner_model(x.view(x.shape[0], -1))
		out = F.cross_entropy(out_x, y)
		if self.is_inner:
			out =  out +  self.reg_term()

		if with_acc:
			pred = out_x.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
			acc = pred.eq(y.view_as(pred)).sum().item() / len(y)
			return out,acc
		else:
			return out

	def reg_term(self):
		ones_dxc = torch.ones(self.inner_params[0].size()).to(self.inner_params[0].device)
		out =  0.5*((self.inner_params[0]**2)*torch.exp(self.outer_model.reg.unsqueeze(1)*ones_dxc)).mean()
		if len(self.outer_params)>1:
			out = out+ (self.inner_params[0].abs() * torch.exp(self.outer_model.reg.unsqueeze(1) * ones_dxc)).mean()
		return out




