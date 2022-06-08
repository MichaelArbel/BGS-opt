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

		# create data loaders for inner and outer problems 
		self.loader = hp.config_to_instance(config_module_name="loader",
											num_workers=self.args.system.num_workers,
											dtype=self.dtype,
											device=self.device,
											**self.args.data)
		self.inner_loader = self.loader.data['inner_loader'] 
		self.outer_loader = self.loader.data['outer_loader']
		self.data_info = self.loader.meta_data
		self.update_args()
		# create either a pytorch Module or a list of parameters
		inner_model_path = self.args.loss.model.inner.pop("path", None)
		self.inner_model = hp.config_to_instance(**self.args.loss.model.inner)
		self.inner_model = hp.init_model(self.inner_model,inner_model_path,
										self.dtype, 
										self.device, 
										is_inner=True
										)
		outer_model_path = self.args.loss.model.outer.pop("path", None)
		self.outer_model = hp.config_to_instance(**self.args.loss.model.outer)
		self.outer_model = hp.init_model(self.outer_model,outer_model_path,
										self.dtype, 
										self.device, 
										is_inner=False
										)
		

		# create a pytorch Modules whose output is a scalar
		
		self.inner_loss = hp.config_to_instance(**self.args.loss.objective.inner,
											outer_model=self.outer_model, 
											inner_model=self.inner_model, 
											device=self.device)
		self.outer_loss = hp.config_to_instance(**self.args.loss.objective.outer,
											outer_model=self.outer_model, 
											inner_model=self.inner_model, 
											device=self.device)
       
		# Construct the approximate solution to inner problem
		self.inner_params = self.inner_loss.inner_params
		self.outer_params = self.inner_loss.outer_params

		self.outer_optimizer = hp.config_to_instance(params=self.outer_params, **self.args.training.optimizer.outer)
		self.inner_optimizer = hp.config_to_instance(params=self.inner_params, **self.args.training.optimizer.inner)

		self.use_outer_scheduler = self.args.training.scheduler.outer.pop("use_scheduler", None)
		self.use_inner_scheduler = self.args.training.scheduler.inner.pop("use_scheduler", None)
		self.inner_scheduler = hp.config_to_instance(optimizer=self.inner_optimizer, **self.args.training.scheduler.inner)
		self.outer_scheduler = hp.config_to_instance(optimizer=self.outer_optimizer, **self.args.training.scheduler.outer)


		self.forward_solver = hp.config_to_instance(optimizer=self.inner_optimizer,**self.args.method.forward)
		self.backward_solver = hp.config_to_instance(**self.args.method.backward)

		# Construct the hyper-loss
		self.hyperloss = HyperLoss(self.outer_loss,
									self.inner_loss, 
									self.inner_loader,
									self.forward_solver,
									self.backward_solver)

		self.mode = 'train'
		self.counter = 0
		self.avg_outer_loss = 0. 
		self.avg_inner_loss = 0.
		self.amortized_grad = None
		self.outer_grad = None
		self.alg_time = 0.
		self.count_max = self.args.training.outer_iterations

		dev_count = torch.cuda.device_count()    


	def update_args(self):
		if self.args.loss.model.inner.name=='core.models.Linear':
			self.args.loss.model.inner.n_features =  self.data_info['n_features']
			self.args.loss.model.inner.n_classes = self.data_info['n_classes']
		if self.args.loss.model.outer.name=='core.models.Identity':
			self.args.loss.model.outer.dim = self.data_info['n_features']


	def main(self):
		print(f'==> Mode: {self.mode}')
		if self.mode == 'train':
			self.train()


 
	def get_grad_counts(self):
		return self.hyperloss.get_grad_counts()
	def train(self):
		self.outer_optimizer.zero_grad()
		init_accum_dict = self.eval_losses()
		accum_dict = { key:[value]  for key,value in init_accum_dict.items()}
		while self.counter<=self.count_max:
			for batch_idx, data in enumerate(self.outer_loader):
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
					self.timer(self.counter, " outer loss: %.8f, inner loss: %.8f" % ( metrics['outer_loss'], metrics['inner_loss']))
					accum_dict = {key:[] for key in accum_dict.keys()}
				self.counter +=1
				out_dict = self.iteration(data)
				self.update_schedule()
				accum_dict = {key:value + [out_dict[key]] for key,value in  accum_dict.items()}
				if self.args.metrics.log_artifacts and self.counter % self.args.metrics.log_artefacts_freq==0:
					self.log(self.counter)

	def iteration(self,data):
		#print(self.outer_params)
		start_time_iter = time.time()
		if len(data)==1 and isinstance(data, list):
			data = data[0]
		data = utils.to_device(data,self.device,self.dtype)
		
		inner_loader = self.inner_loader
		amortized_grad = self.amortized_grad
		inner_params = self.inner_params


		inner_params_out, iterates = self.forward_solver.run(self.inner_loss,inner_loader,inner_params)
		
		utils.zero_grad(inner_params)
		utils.zero_grad(self.outer_params)
		loss = self.hyperloss.func(data)
		
		torch.autograd.backward(loss, retain_graph=True, create_graph=False, inputs=inner_params+self.outer_params)
		loss = loss.detach().cpu()
		# #print('loss'+ str( loss))
		self.hyperloss.counter_outer_grad +=1
		for p in inner_params:
			if p.grad is None:
				p.grad = torch.zeros_like(p)
		inner_grads = [p.grad for p in inner_params]


		out, amortized_grad =  self.backward_solver.run(self.inner_loss,
						inner_loader,
						self.outer_params,
						inner_params,
						iterates,
						amortized_grad,
						inner_grads)
		out = [o if o is not None else torch.zeros_like(p) for p,o in zip(self.outer_params,out)]
		for p,o in zip(self.outer_params, out):
			if p.grad is not None:
				p.grad  = p.grad + o
			else:
				p.grad = 1.*o
		
		if self.outer_grad is not None:
			self.outer_grad = tuple([p+1.*o.grad for p,o in zip(self.outer_grad,self.outer_params)])
		else:
			self.outer_grad = tuple([1.*p.grad for p in self.outer_params])
				
		
		for p, g in zip(self.outer_params,self.outer_grad):
			p.grad = 1.*g
		self.outer_optimizer.step()
		self.outer_optimizer.zero_grad()
		self.outer_grad = None

		self.amortized_grad = amortized_grad
		end_time_iter = time.time()
		self.alg_time += end_time_iter-start_time_iter

		data_loaders = self.loader.data['eval_inner_loader'], self.loader.data['eval_outer_loader']
		out_dict = self.eval_losses( data_loaders = data_loaders)
		return out_dict 

	def update_schedule(self):
		if self.use_inner_scheduler:
			self.inner_scheduler.step()
		if self.use_outer_scheduler:
			self.outer_scheduler.step()

	def eval_losses(self, data_loaders = None):
		out_dict = self.eval_outer(self.loader.data['eval_outer_loader'],self.hyperloss.func.forward, self.args.metrics.max_outer_iter)
		inner_loss = self.hyperloss.eval_inner_loss(self.loader.data['eval_inner_loader'],total=True, max_iter=self.args.metrics.max_inner_iter)
		if self.args.metrics.log_inner_cond and self.counter%self.args.metrics.freq_inner_cond==0:
			#with torch.enable_grad():
			kappa = utils.cond_loss(self.inner_loss,self.inner_loader,self.inner_params[0])
			out_dict.update({'kappa':kappa.item()})					
			print({'kappa':kappa.item()})
		utils.add_prefix_to_keys_dict(out_dict, 'outer_')
		out_dict.update({'inner_loss':inner_loss.item()})
		if self.args.metrics.eval_test:
			new_dict = self.eval_outer(self.loader.data['test_outer_loader'],self.hyperloss.func.forward, self.args.metrics.max_outer_iter)
			utils.add_prefix_to_keys_dict(new_dict, 'test_')
			out_dict.update(new_dict)
		return out_dict



	def eval_outer(self,loader,func,max_iter):
		outer_loss = 0
		outer_acc = 0
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
			outer_loss = outer_loss + loss
			outer_acc = outer_acc + acc
		outer_acc = outer_acc/counter
		outer_loss = outer_loss/counter
		if 'with_acc' in args:
			return {'loss': outer_loss.item(), 'acc': 100*outer_acc}
		else:

			return {'loss': outer_loss.item()} 
 

	def log(self,step):
		if self.args.loss.objective.inner.name=='LogisticDistill':
			x,_,y =self.outer_params
			self.log_artifacts({'image': x.cpu().detach().numpy(), 'label':y.cpu().detach().numpy()},step, art_type='arrays', tag='distilled_image')
			N_h = 2
			fig = make_and_save_grid_images(x.cpu().detach(), N_h=N_h,N_w=int(y.shape[0]/N_h))
			self.log_artifacts(fig,step, art_type='figures', tag='distilled_image')



### Savers

  