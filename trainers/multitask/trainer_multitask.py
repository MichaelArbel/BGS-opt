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
		self.avg_outer_loss = 0. 
		self.avg_inner_loss = 0.
		self.amortized_grad = None
		self.outer_grad = None
		self.alg_time = 0.

		self.build_model()



	def build_model(self):

		# create data loaders for inner and outer problems
		
		self.timer(self.counter, " loading data " )
		self.loader = hp.config_to_instance(config_module_name="loader",
											num_workers=self.args.system.num_workers,
											dtype=self.dtype,
											device=self.device,
											**self.args.loader)
		self.inner_loader = self.loader.data['inner_loader'] 
		self.outer_loader = self.loader.data['outer_loader']
		self.meta_data = self.loader.meta_data
		self.update_args()
		# create either a pytorch Module or a list of parameters
		self.timer(self.counter, " creating inner model " )
		inner_model_path = self.args.loss.model.inner.pop("path", None)
		self.inner_model = hp.config_to_instance(**self.args.loss.model.inner)
		self.inner_model = hp.init_model(self.inner_model,inner_model_path,
										self.dtype, 
										self.device, 
										is_inner=True
										)
		self.log_artifacts(self.inner_model,0,'torch_models', tag='inner_model')

		self.timer(self.counter, " creating outer model " )
		outer_model_path = self.args.loss.model.outer.pop("path", None)
		self.outer_model = hp.config_to_instance(**self.args.loss.model.outer)
		self.outer_model = hp.init_model(self.outer_model,outer_model_path,
										self.dtype, 
										self.device, 
										is_inner=False
										)
		

		# create a pytorch Modules whose output is a scalar
		self.timer(self.counter, " creating losses " )

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

		self.timer(self.counter, " creating optimizers " )

		self.outer_optimizer = hp.config_to_instance(params=self.outer_params, **self.args.training.optimizer.outer)
		self.inner_optimizer = hp.config_to_instance(params=self.inner_params, **self.args.training.optimizer.inner)

		self.use_outer_scheduler = self.args.training.scheduler.outer.pop("use_scheduler", None)
		self.use_inner_scheduler = self.args.training.scheduler.inner.pop("use_scheduler", None)
		if self.use_outer_scheduler:
			self.outer_scheduler = hp.config_to_instance(optimizer=self.outer_optimizer, **self.args.training.scheduler.outer)
		if self.use_inner_scheduler:
			self.inner_scheduler = hp.config_to_instance(optimizer=self.inner_optimizer, **self.args.training.scheduler.inner)
		
		self.timer(self.counter, " creating solvers " )		

		self.forward_solver = hp.config_to_instance(optimizer=self.inner_optimizer,**self.args.method.forward)
		self.backward_solver = hp.config_to_instance(module=self.inner_loss, **self.args.method.backward)

		# Construct the hyper-loss
		self.hyperloss = HyperLoss(self.outer_loss,
									self.inner_loss, 
									self.inner_loader,
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
			return self.args.training.outer_iterations, total_batches


	def update_args(self):
		if self.args.loss.model.inner.name=='core.models.Linear':
			self.args.loss.model.inner.n_features =  self.meta_data['n_features']
			self.args.loss.model.inner.n_classes = self.meta_data['n_classes']
#		if self.args.loss.model.outer.name=='core.models.Identity':
#			self.args.loss.model.outer.dim = self.meta_data['n_features']

	def build_metrics(self):
		self.metrics = Metrics(self.args.metrics,self.device,self.dtype)
		name = self.args.metrics.name
		condition = lambda counter : counter%self.total_batches==0
		self.metrics.register_metric(self.outer_loss,
									self.loader.data['test_outer_loader'],
									0,
									'test_outer',
									metric=name,
									condition=condition)
		# self.metrics.register_metric(self.outer_loss,
		# 							self.loader.data['eval_outer_loader'],
		# 							self.args.metrics.max_outer_iter,
		# 							'train_outer',
		# 							metric=name)
		# self.metrics.register_metric(self.inner_loss,
		# 							self.loader.data['eval_inner_loader'],
		# 							self.args.metrics.max_inner_iter,
		# 							'train_inner',
		# 							metric=name)

		if self.args.metrics.log_inner_cond:
			condition = lambda counter : counter%self.args.freq_inner_cond==0
			self.metrics.register_metric(self.inner_loss,
										self.loader.data['eval_inner_loader'],
										1,
										'inner_kappa',
										func_args={'params':self.inner_params[0]},
										condition=condition,
										metric='cond')


	def main(self):
		print(f'==> Mode: {self.mode}')
		if self.mode == 'train':
			self.train()


 
	def get_grad_counts(self):
		return self.hyperloss.get_grad_counts()
	def train(self):
		self.outer_optimizer.zero_grad()
		if self.counter==0:
			self.alg_time = 0.
		while self.counter<=self.count_max:
			for batch_idx, data in enumerate(self.outer_loader):
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
		self.timer(self.counter, "Epoch %d | Alg Time %.2f | outer loss: %.2f, outer acc: %.2f | test loss: %.2f, test acc: %.2f" % 
							(self.epoch,self.alg_time,
							metrics['train_outer_loss'], metrics['train_outer_acc'],
							metrics['test_outer_loss_all'], metrics['test_outer_acc_all']))
		#self.timer(self.counter, " outer loss: , inner loss: " )

		#print('Epoch {}| Time {}'.format(self.epoch,self.alg_time))
		#print(metrics)
	
	def iteration(self,data):
		start_time_iter = time.time()
		if len(data)==1 and isinstance(data, list):
			data = data[0]
		data = utils.to_device(data,self.device,self.dtype)
		
		inner_loader = self.inner_loader
		amortized_grad = self.amortized_grad
		inner_params = self.inner_params
		inner_params_out, iterates,inner_loss = self.forward_solver.run(self.inner_loss,inner_loader,inner_params)
		
		utils.zero_grad(inner_params)
		utils.zero_grad(self.outer_params)
		loss, acc = self.hyperloss.func(data,with_acc=True)
		
		torch.autograd.backward(loss, retain_graph=True, create_graph=False, inputs=inner_params+self.outer_params)
		loss = loss.detach().cpu()
		#inner_loss, inner_acc = inner_out
		inner_loss = inner_loss.detach().cpu()
		#inner_acc = inner_acc.detach().cpu()
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
		
		# if self.outer_grad is not None:
		# 	self.outer_grad = tuple([p+1.*o.grad for p,o in zip(self.outer_grad,self.outer_params)])
		# else:
		# 	self.outer_grad = tuple([1.*p.grad for p in self.outer_params])
				
		# for p, g in zip(self.outer_params,self.outer_grad):
		# 	p.grad = 1.*g
		self.outer_optimizer.step()
		self.outer_optimizer.zero_grad()
		#self.outer_grad = None

		self.amortized_grad = amortized_grad

		end_time_iter = time.time()
		self.alg_time += end_time_iter-start_time_iter

		metrics = { 'train_outer_loss': loss.item(),
					'train_inner_loss': inner_loss.item(),
					'train_outer_acc': 100.*acc.item()}
		#print(self.outer_params)		
		return metrics

	def iteration(self,data):
		start_time_iter = time.time()
		if len(data)==1 and isinstance(data, list):
			data = data[0]
		data = utils.to_device(data,self.device,self.dtype)
		

		inner_loader = self.inner_loader
		amortized_grad = self.amortized_grad
		inner_params = self.inner_params
		
		time_1 = time.time()
		inner_params_out, iterates,inner_loss = self.forward_solver.run(self.inner_loss,inner_loader,inner_params)
		time_2 = time.time()
		print('Time Forward: {}'.format(time_2-time_1) )

		utils.zero_grad(inner_params)
		utils.zero_grad(self.outer_params)
		loss = self.hyperloss.func(data)
		
		torch.autograd.backward(loss, retain_graph=True, create_graph=False, inputs=inner_params+self.outer_params)
		loss = loss.detach().cpu()
		#inner_loss = inner_loss.detach().cpu()
		self.hyperloss.counter_outer_grad +=1
		for p in inner_params:
			if p.grad is None:
				p.grad = torch.zeros_like(p)
		inner_grads = [p.grad for p in inner_params]

		time_1 = time.time()
		out, amortized_grad =  self.backward_solver.run(self.inner_loss,
						inner_loader,
						self.outer_params,
						inner_params,
						iterates,
						amortized_grad,
						inner_grads)
		time_2 = time.time()
		print('Time Backward: {}'.format(time_2-time_1) )
		# out = [o if o is not None else torch.zeros_like(p) for p,o in zip(self.outer_params,out)]
		# for p,o in zip(self.outer_params, out):
		# 	if p.grad is not None:
		# 		p.grad  = p.grad + o
		# 	else:
		# 		p.grad = 1.*o
		
		# if self.outer_grad is not None:
		# 	self.outer_grad = tuple([p+1.*o.grad for p,o in zip(self.outer_grad,self.outer_params)])
		# else:
		# 	self.outer_grad = tuple([1.*p.grad for p in self.outer_params])
				
		# for p, g in zip(self.outer_params,self.outer_grad):
		# 	p.grad = 1.*g
		# self.outer_optimizer.step()
		# self.outer_optimizer.zero_grad()
		# self.outer_grad = None

		# self.amortized_grad = amortized_grad
		
		end_time_iter = time.time()
		self.alg_time += end_time_iter-start_time_iter
		metrics = { 'train_outer_loss': loss.item(),
					'train_inner_loss': inner_loss.item(),
					'train_outer_acc': 0.}

		return   metrics

	def update_schedule(self):
		if self.use_inner_scheduler:
			self.inner_scheduler.step()
		if self.use_outer_scheduler:
			self.outer_scheduler.step()

	def log_image(self,step):
		if self.args.loss.objective.inner.name=='LogisticDistill':
			x,_,y =self.outer_params
			self.log_artifacts({'image': x.cpu().detach().numpy(), 'label':y.cpu().detach().numpy()},step, art_type='arrays', tag='distilled_image')
			N_h = 2
			fig = make_and_save_grid_images(x.cpu().detach(), N_h=N_h,N_w=int(y.shape[0]/N_h))
			self.log_artifacts(fig,step, art_type='figures', tag='distilled_image')



### Savers  