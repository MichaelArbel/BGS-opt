import math

import torch
import torch.nn as nn

import numpy as np

import csv
import sys
import os
import time
from datetime import datetime
from core.hypermodules import HyperLoss
import learn2learn as l2l

import trainers.utils.helpers as hp   


from Experimentalist.experimentalist import Experimentalist
from core import utils
from core import models
import core
import inspect
from torch import autograd
import copy 
from trainers.utils.viz import make_and_save_grid_images



class Trainer(Experimentalist):
	def __init__(self, args):
		super(Trainer, self).__init__(args)
		self.args = utils.config_to_dict(args)
		#self.args = args
		#self.exp = Experiment(self.args)
		self.device = hp.assign_device(args.system.device)
		self.dtype = hp.get_dtype(args.system.dtype)
		self.build_model()



	def build_model(self):

		# create data loaders for lower and upper problems 
		self.loader = hp.config_to_instance(config_module_name="loader",
											num_workers=self.args.system.num_workers,
											dtype=self.dtype,
											device=self.device,
											**self.args.data)
		self.lower_loader = self.loader.data['lower_loader'] 
		self.upper_loader = self.loader.data['upper_loader']
		self.data_info = self.loader.meta_data
		self.update_args()
		# create either a pytorch Module or a list of parameters
		lower_model_path = self.args.loss.model.lower.pop("path", None)
		self.lower_model = hp.config_to_instance(**self.args.loss.model.lower)
		self.lower_model = hp.init_model(self.lower_model,lower_model_path,
										self.dtype, 
										self.device, 
										is_lower=True
										)
		upper_model_path = self.args.loss.model.upper.pop("path", None)
		self.upper_model = hp.config_to_instance(**self.args.loss.model.upper)
		self.upper_model = hp.init_model(self.upper_model,upper_model_path,
										self.dtype, 
										self.device, 
										is_lower=False
										)
		

		# create a pytorch Modules whose output is a scalar
		
		self.lower_loss = hp.config_to_instance(**self.args.loss.objective.lower,
											upper_model=self.upper_model, 
											lower_model=self.lower_model, 
											device=self.device)
		self.upper_loss = hp.config_to_instance(**self.args.loss.objective.upper,
											upper_model=self.upper_model, 
											lower_model=self.lower_model, 
											device=self.device)
       
		# Construct the approximate solution to lower problem
		self.lower_params = self.lower_loss.lower_params
		self.upper_params = self.lower_loss.upper_params

		self.upper_optimizer = hp.config_to_instance(params=self.upper_params, **self.args.training.optimizer.upper)
		self.lower_optimizer = hp.config_to_instance(params=self.lower_params, **self.args.training.optimizer.lower)

		self.use_upper_scheduler = self.args.training.scheduler.upper.pop("use_scheduler", None)
		self.use_lower_scheduler = self.args.training.scheduler.lower.pop("use_scheduler", None)
		self.lower_scheduler = hp.config_to_instance(optimizer=self.lower_optimizer, **self.args.training.scheduler.lower)
		self.upper_scheduler = hp.config_to_instance(optimizer=self.upper_optimizer, **self.args.training.scheduler.upper)


		self.forward_solver = hp.config_to_instance(optimizer=self.lower_optimizer,**self.args.method.forward)
		self.backward_solver = hp.config_to_instance(**self.args.method.backward)

		# Construct the hyper-loss
		self.hyperloss = HyperLoss(self.upper_loss,
									self.lower_loss, 
									self.lower_loader,
									self.forward_solver,
									self.backward_solver)

		self.mode = 'train'
		self.counter = 0
		self.avg_upper_loss = 0. 
		self.avg_lower_loss = 0.
		self.amortized_grad = None
		self.upper_grad = None
		self.alg_time = 0.
		self.count_max = self.args.training.upper_iterations

		dev_count = torch.cuda.device_count()    


	def update_args(self):
		if self.args.loss.model.lower.name=='core.models.Linear':
			self.args.loss.model.lower.n_features =  self.data_info['n_features']
			self.args.loss.model.lower.n_classes = self.data_info['n_classes']
		if self.args.loss.model.upper.name=='core.models.Identity':
			self.args.loss.model.upper.dim = self.data_info['n_features']


	def main(self):
		print(f'==> Mode: {self.mode}')
		if self.mode == 'train':
			self.train()


 
	def get_grad_counts(self):
		return self.hyperloss.get_grad_counts()
	def train(self):
		self.upper_optimizer.zero_grad()
		init_accum_dict = self.eval_losses()
		accum_dict = { key:[value]  for key,value in init_accum_dict.items()}
		while self.counter<=self.count_max:
			for batch_idx, data in enumerate(self.upper_loader):
				if self.counter>self.count_max:
					break

				if self.counter % self.args.metrics.disp_freq == 0:
					if self.counter==0:
						self.alg_time = 0.
					accum_dict = {key:np.asarray(value).mean() for key,value in accum_dict.items()}
					metrics = {	'iter': self.counter, 'time': self.alg_time }
					metrics.update(accum_dict)
					metrics.update(self.get_grad_counts())
					self.log_metrics(metrics)
					self.timer(self.counter, " upper loss: %.8f, lower loss: %.8f" % ( metrics['upper_loss'], metrics['lower_loss']))
					accum_dict = {key:[] for key in accum_dict.keys()}
				self.counter +=1
				out_dict = self.iteration(data)
				self.update_schedule()
				accum_dict = {key:value + [out_dict[key]] for key,value in  accum_dict.items()}
				if self.args.metrics.log_artifacts and self.counter % self.args.metrics.log_artefacts_freq==0:
					self.log(self.counter)

	def iteration(self,data):
		#print(self.upper_params)
		start_time_iter = time.time()
		if len(data)==1 and isinstance(data, list):
			data = data[0]
		data = utils.to_device(data,self.device,self.dtype)
		
		lower_loader = self.lower_loader
		amortized_grad = self.amortized_grad
		lower_params = self.lower_params


		lower_params_out, iterates = self.forward_solver.run(self.lower_loss,lower_loader,lower_params)
		
		utils.zero_grad(lower_params)
		utils.zero_grad(self.upper_params)
		loss = self.hyperloss.func(data)
		
		torch.autograd.backward(loss, retain_graph=True, create_graph=False, inputs=lower_params+self.upper_params)
		loss = loss.detach().cpu()
		# #print('loss'+ str( loss))
		self.hyperloss.counter_upper_grad +=1
		for p in lower_params:
			if p.grad is None:
				p.grad = torch.zeros_like(p)
		lower_grads = [p.grad for p in lower_params]


		out, amortized_grad =  self.backward_solver.run(self.lower_loss,
						lower_loader,
						self.upper_params,
						lower_params,
						iterates,
						amortized_grad,
						lower_grads)
		out = [o if o is not None else torch.zeros_like(p) for p,o in zip(self.upper_params,out)]
		for p,o in zip(self.upper_params, out):
			if p.grad is not None:
				p.grad  = p.grad + o
			else:
				p.grad = 1.*o
		
		if self.upper_grad is not None:
			self.upper_grad = tuple([p+1.*o.grad for p,o in zip(self.upper_grad,self.upper_params)])
		else:
			self.upper_grad = tuple([1.*p.grad for p in self.upper_params])
				
		
		for p, g in zip(self.upper_params,self.upper_grad):
			p.grad = 1.*g
		self.upper_optimizer.step()
		self.upper_optimizer.zero_grad()
		self.upper_grad = None

		self.amortized_grad = amortized_grad
		end_time_iter = time.time()
		self.alg_time += end_time_iter-start_time_iter

		data_loaders = self.loader.data['eval_lower_loader'], self.loader.data['eval_upper_loader']
		out_dict = self.eval_losses( data_loaders = data_loaders)
		return out_dict 

	def update_schedule(self):
		if self.use_lower_scheduler:
			self.lower_scheduler.step()
		if self.use_upper_scheduler:
			self.upper_scheduler.step()

	def eval_losses(self, data_loaders = None):
		out_dict = self.eval_upper(self.loader.data['eval_upper_loader'],self.hyperloss.func.forward, self.args.metrics.max_upper_iter)
		lower_loss = self.hyperloss.eval_lower_loss(self.loader.data['eval_lower_loader'],total=True, max_iter=self.args.metrics.max_lower_iter)
		if self.args.metrics.log_lower_cond and self.counter%self.args.metrics.freq_lower_cond==0:
			#with torch.enable_grad():
			kappa = utils.cond_loss(self.lower_loss,self.lower_loader,self.lower_params[0])
			out_dict.update({'kappa':kappa.item()})					
			print({'kappa':kappa.item()})
		utils.add_prefix_to_keys_dict(out_dict, 'upper_')
		out_dict.update({'lower_loss':lower_loss.item()})
		if self.args.metrics.eval_test:
			new_dict = self.eval_upper(self.loader.data['test_upper_loader'],self.hyperloss.func.forward, self.args.metrics.max_upper_iter)
			utils.add_prefix_to_keys_dict(new_dict, 'test_')
			out_dict.update(new_dict)
		return out_dict



	def eval_upper(self,loader,func,max_iter):
		upper_loss = 0
		upper_acc = 0
		counter = 0
		#func = self.hyperloss.func.forward
		args = inspect.getfullargspec(func)[0]

		for batch_idx, data in enumerate(loader):
			if batch_idx>max_iter and max_iter>0:
				break
			if len(data)==1 and isinstance(data, list):
				data = data[0]
			data = utils.to_device(data,self.device,self.dtype)
			counter += 1
			if 'with_acc' in args:
				loss,acc = func(data,with_acc=True)
			else:
				loss = func(data)
				acc = 0
			upper_loss = upper_loss + loss
			upper_acc = upper_acc + acc
		upper_acc = upper_acc/counter
		upper_loss = upper_loss/counter
		if 'with_acc' in args:
			return {'loss': upper_loss.item(), 'acc': 100*upper_acc}
		else:

			return {'loss': upper_loss.item()} 
 

	def log(self,step):
		if self.args.loss.objective.lower.name=='LogisticDistill':
			x,_,y =self.upper_params
			self.log_artifacts({'image': x.cpu().detach().numpy(), 'label':y.cpu().detach().numpy()},step, art_type='arrays', tag='distilled_image')
			N_h = 2
			fig = make_and_save_grid_images(x.cpu().detach(), N_h=N_h,N_w=int(y.shape[0]/N_h))
			self.log_artifacts(fig,step, art_type='figures', tag='distilled_image')



### Savers

  