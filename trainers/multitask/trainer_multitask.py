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

from torch import autograd
import copy 
from trainers.utils.viz import make_and_save_grid_images
from trainers.utils.metrics import Metrics


class Trainer(Experimentalist):
	def __init__(self, args):
		super(Trainer, self).__init__(args)
		self.args = utils.config_to_dict(args)
		#self.args = args
		#self.exp = Experiment(self.args)
		self.device = hp.assign_device(args.system.device)
		self.dtype = hp.get_dtype(args.system.dtype)
		
		self.mode = 'train'
		self.counter = 0
		self.epoch = 0
		self.avg_upper_loss = 0. 
		self.avg_lower_loss = 0.
		self.amortized_grad = None
		self.upper_grad = None
		self.alg_time = 0.

		self.build_model()



	def build_model(self):

		# create data loaders for lower and upper problems
		
		self.timer(self.counter, " loading data " )
		self.loader = hp.config_to_instance(config_module_name="loader",
											num_workers=self.args.system.num_workers,
											dtype=self.dtype,
											device=self.device,
											**self.args.loader)
		self.lower_loader = self.loader.data['lower_loader'] 
		self.upper_loader = self.loader.data['upper_loader']
		self.meta_data = self.loader.meta_data
		self.update_args()
		# create either a pytorch Module or a list of parameters
		self.timer(self.counter, " creating lower model " )
		lower_model_path = self.args.loss.model.lower.pop("path", None)
		self.lower_model = hp.config_to_instance(**self.args.loss.model.lower)
		self.lower_model = hp.init_model(self.lower_model,lower_model_path,
										self.dtype, 
										self.device, 
										is_lower=True
										)
		self.log_artifacts(self.lower_model,0,'torch_models', tag='lower_model')

		self.timer(self.counter, " creating upper model " )
		upper_model_path = self.args.loss.model.upper.pop("path", None)
		self.upper_model = hp.config_to_instance(**self.args.loss.model.upper)
		self.upper_model = hp.init_model(self.upper_model,upper_model_path,
										self.dtype, 
										self.device, 
										is_lower=False
										)
		

		# create a pytorch Modules whose output is a scalar
		self.timer(self.counter, " creating losses " )

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

		self.timer(self.counter, " creating optimizers " )

		self.upper_optimizer = hp.config_to_instance(params=self.upper_params, **self.args.training.optimizer.upper)
		self.lower_optimizer = hp.config_to_instance(params=self.lower_params, **self.args.training.optimizer.lower)

		self.use_upper_scheduler = self.args.training.scheduler.upper.pop("use_scheduler", None)
		self.use_lower_scheduler = self.args.training.scheduler.lower.pop("use_scheduler", None)
		if self.use_upper_scheduler:
			self.upper_scheduler = hp.config_to_instance(optimizer=self.upper_optimizer, **self.args.training.scheduler.upper)
		if self.use_lower_scheduler:
			self.lower_scheduler = hp.config_to_instance(optimizer=self.lower_optimizer, **self.args.training.scheduler.lower)
		
		self.timer(self.counter, " creating solvers " )		

		self.forward_solver = hp.config_to_instance(optimizer=self.lower_optimizer,**self.args.method.forward)
		self.backward_solver = hp.config_to_instance(module=self.lower_loss, **self.args.method.backward)

		# Construct the hyper-loss
		self.hyperloss = HyperLoss(self.upper_loss,
									self.lower_loss, 
									self.lower_loader,
									self.forward_solver,
									self.backward_solver)


		self.count_max, self.total_batches = self.set_count_max()

		self.timer(self.counter, " creating metrics " )

		self.build_metrics()
		dev_count = torch.cuda.device_count()    


	def set_count_max(self):
		total_batches = self.meta_data['total_samples']/self.meta_data['b_size']
		if self.args.training.by_epoch:
			return self.args.training.total_epoch*total_batches, total_batches
		else:
			return self.args.training.upper_iterations, total_batches


	def update_args(self):
		if self.args.loss.model.lower.name=='core.models.Linear':
			self.args.loss.model.lower.n_features =  self.meta_data['n_features']
			self.args.loss.model.lower.n_classes = self.meta_data['n_classes']
#		if self.args.loss.model.upper.name=='core.models.Identity':
#			self.args.loss.model.upper.dim = self.meta_data['n_features']

	def build_metrics(self):
		self.metrics = Metrics(self.args.metrics,self.device,self.dtype)
		name = self.args.metrics.name
		condition = lambda counter : counter%self.total_batches==0
		self.metrics.register_metric(self.upper_loss,
									self.loader.data['test_upper_loader'],
									0,
									'test_upper',
									metric=name,
									condition=condition)
		# self.metrics.register_metric(self.upper_loss,
		# 							self.loader.data['eval_upper_loader'],
		# 							self.args.metrics.max_upper_iter,
		# 							'train_upper',
		# 							metric=name)
		# self.metrics.register_metric(self.lower_loss,
		# 							self.loader.data['eval_lower_loader'],
		# 							self.args.metrics.max_lower_iter,
		# 							'train_lower',
		# 							metric=name)

		if self.args.metrics.log_lower_cond:
			condition = lambda counter : counter%self.args.freq_lower_cond==0
			self.metrics.register_metric(self.lower_loss,
										self.loader.data['eval_lower_loader'],
										1,
										'lower_kappa',
										func_args={'params':self.lower_params[0]},
										condition=condition,
										metric='cond')


	def main(self):
		print(f'==> Mode: {self.mode}')
		if self.mode == 'train':
			self.train()


 
	def get_grad_counts(self):
		return self.hyperloss.get_grad_counts()
	def train(self):
		self.upper_optimizer.zero_grad()
		if self.counter==0:
			self.alg_time = 0.
		while self.counter<=self.count_max:
			for batch_idx, data in enumerate(self.upper_loader):
				if self.counter>self.count_max:
					break					
				self.counter +=1
				metrics = self.iteration(data)
				self.metrics.eval_metrics(self.counter,metrics)
				if self.args.metrics.log_artifacts and self.counter % self.args.metrics.log_artefacts_freq==0:
					self.log_image(self.counter)
			self.update_schedule()
			self.disp_metrics()
			self.epoch += 1 

	def disp_metrics(self):

		metrics = self.metrics.avg_metrics()
		metrics.update({'iter': self.counter, 'time': self.alg_time, 'epoch':self.epoch})
		metrics.update(self.get_grad_counts())
		self.log_metrics(metrics)
		self.timer(self.counter, "Epoch %d | Alg Time %.2f | upper loss: %.2f, upper acc: %.2f | test loss: %.2f, test acc: %.2f" % 
							(self.epoch,self.alg_time,
							metrics['train_upper_loss'], metrics['train_upper_acc'],
							metrics['test_upper_loss_all'], metrics['test_upper_acc_all']))
		#self.timer(self.counter, " upper loss: , lower loss: " )

		#print('Epoch {}| Time {}'.format(self.epoch,self.alg_time))
		#print(metrics)
	
	def iteration(self,data):
		start_time_iter = time.time()
		if len(data)==1 and isinstance(data, list):
			data = data[0]
		data = utils.to_device(data,self.device,self.dtype)
		
		lower_loader = self.lower_loader
		amortized_grad = self.amortized_grad
		lower_params = self.lower_params
		lower_params_out, iterates,lower_loss = self.forward_solver.run(self.lower_loss,lower_loader,lower_params)
		
		utils.zero_grad(lower_params)
		utils.zero_grad(self.upper_params)
		loss, acc = self.hyperloss.func(data,with_acc=True)
		
		torch.autograd.backward(loss, retain_graph=True, create_graph=False, inputs=lower_params+self.upper_params)
		loss = loss.detach().cpu()
		#lower_loss, lower_acc = lower_out
		lower_loss = lower_loss.detach().cpu()
		#lower_acc = lower_acc.detach().cpu()
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
		
		# if self.upper_grad is not None:
		# 	self.upper_grad = tuple([p+1.*o.grad for p,o in zip(self.upper_grad,self.upper_params)])
		# else:
		# 	self.upper_grad = tuple([1.*p.grad for p in self.upper_params])
				
		# for p, g in zip(self.upper_params,self.upper_grad):
		# 	p.grad = 1.*g
		self.upper_optimizer.step()
		self.upper_optimizer.zero_grad()
		#self.upper_grad = None

		self.amortized_grad = amortized_grad

		end_time_iter = time.time()
		self.alg_time += end_time_iter-start_time_iter

		metrics = { 'train_upper_loss': loss.item(),
					'train_lower_loss': lower_loss.item(),
					'train_upper_acc': 100.*acc.item()}
		#print(self.upper_params)		
		return metrics

	def iteration(self,data):
		start_time_iter = time.time()
		if len(data)==1 and isinstance(data, list):
			data = data[0]
		data = utils.to_device(data,self.device,self.dtype)
		

		lower_loader = self.lower_loader
		amortized_grad = self.amortized_grad
		lower_params = self.lower_params
		
		time_1 = time.time()
		lower_params_out, iterates,lower_loss = self.forward_solver.run(self.lower_loss,lower_loader,lower_params)
		time_2 = time.time()
		print('Time Forward: {}'.format(time_2-time_1) )

		utils.zero_grad(lower_params)
		utils.zero_grad(self.upper_params)
		loss = self.hyperloss.func(data)
		
		torch.autograd.backward(loss, retain_graph=True, create_graph=False, inputs=lower_params+self.upper_params)
		loss = loss.detach().cpu()
		#lower_loss = lower_loss.detach().cpu()
		self.hyperloss.counter_upper_grad +=1
		for p in lower_params:
			if p.grad is None:
				p.grad = torch.zeros_like(p)
		lower_grads = [p.grad for p in lower_params]

		time_1 = time.time()
		out, amortized_grad =  self.backward_solver.run(self.lower_loss,
						lower_loader,
						self.upper_params,
						lower_params,
						iterates,
						amortized_grad,
						lower_grads)
		time_2 = time.time()
		print('Time Backward: {}'.format(time_2-time_1) )
		# out = [o if o is not None else torch.zeros_like(p) for p,o in zip(self.upper_params,out)]
		# for p,o in zip(self.upper_params, out):
		# 	if p.grad is not None:
		# 		p.grad  = p.grad + o
		# 	else:
		# 		p.grad = 1.*o
		
		# if self.upper_grad is not None:
		# 	self.upper_grad = tuple([p+1.*o.grad for p,o in zip(self.upper_grad,self.upper_params)])
		# else:
		# 	self.upper_grad = tuple([1.*p.grad for p in self.upper_params])
				
		# for p, g in zip(self.upper_params,self.upper_grad):
		# 	p.grad = 1.*g
		# self.upper_optimizer.step()
		# self.upper_optimizer.zero_grad()
		# self.upper_grad = None

		# self.amortized_grad = amortized_grad
		
		end_time_iter = time.time()
		self.alg_time += end_time_iter-start_time_iter
		metrics = { 'train_upper_loss': loss.item(),
					'train_lower_loss': lower_loss.item(),
					'train_upper_acc': 0.}

		return   metrics

	def update_schedule(self):
		if self.use_lower_scheduler:
			self.lower_scheduler.step()
		if self.use_upper_scheduler:
			self.upper_scheduler.step()

	def log_image(self,step):
		if self.args.loss.objective.lower.name=='LogisticDistill':
			x,_,y =self.upper_params
			self.log_artifacts({'image': x.cpu().detach().numpy(), 'label':y.cpu().detach().numpy()},step, art_type='arrays', tag='distilled_image')
			N_h = 2
			fig = make_and_save_grid_images(x.cpu().detach(), N_h=N_h,N_w=int(y.shape[0]/N_h))
			self.log_artifacts(fig,step, art_type='figures', tag='distilled_image')



### Savers  