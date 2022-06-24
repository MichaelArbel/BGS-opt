import math

import torch
import torch.nn as nn

import numpy as np

import csv
import sys
import os
import time
from datetime import datetime
from core.selection import Selection, Functional

import trainers.utils.helpers as hp   


from Experimentalist.experimentalist import Experimentalist
from core import utils
import core

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
		self.loaders = hp.config_to_instance(config_module_name="loader",
											num_workers=self.args.system.num_workers,
											dtype=self.dtype,
											device=self.device,
											**self.args.loader)
		self.lower_loader = self.loaders.data['lower_loader'] 
		self.upper_loader = self.loaders.data['upper_loader']
		self.meta_data = self.loaders.meta_data
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

		self.lower_var = tuple(self.lower_model.parameters())
		self.upper_var = tuple(self.upper_model.parameters())

		# create a pytorch Modules whose output is a scalar
		self.timer(self.counter, " creating losses " )

		self.lower_loss_module = hp.config_to_instance(**self.args.loss.objective.lower,
								upper_model=self.upper_model, 
								lower_model=self.lower_model, 
								device=self.device)


		self.upper_loss_module = hp.config_to_instance(**self.args.loss.objective.upper,
								upper_model=self.upper_model, 
								lower_model=self.lower_model, 
								device=self.device)
		
		## Make the loss modules functional
		self.lower_loss = Functional(self.lower_loss_module)
		self.upper_loss = Functional(self.upper_loss_module)


		self.timer(self.counter, " creating optimizers " )

		self.upper_optimizer = hp.config_to_instance(params=self.upper_var, **self.args.training.optimizer.upper)
		self.use_upper_scheduler = self.args.training.scheduler.upper.pop("use_scheduler", None)
		if self.use_upper_scheduler:
			self.upper_scheduler = hp.config_to_instance(optimizer=self.upper_optimizer, **self.args.training.scheduler.upper)
		
		# def l_loss(data, upper_var, lower_var):
		# 	#dummy_data = torch.sum(torch.stack([torch.sum(data**2) for var in upper_var],axis=0))
		# 	upper_reg = torch.sum(torch.stack([torch.sum(var**2) for var in upper_var],axis=0))
		# 	lower_reg = torch.sum(torch.stack([torch.sum(var**2) for var in lower_var],axis=0))
		# 	return upper_reg #+ lower_reg
		#Construct the selection
		self.selection = Selection(self.lower_loss,
									self.lower_var,
									self.lower_loader,
									self.args.selection,
									self.device,
									self.dtype)


		self.count_max, self.total_batches = self.set_count_max()

		self.timer(self.counter, " creating metrics " )

		self.metrics = Metrics(self.args.metrics,self.device,self.dtype)
		name = self.args.metrics.name
		condition = lambda counter : counter%self.total_batches==0
		self.metrics.register_metric(self.upper_loss,
									self.loaders.data['test_upper_loader'],
									0,
									'test_upper',
									func_args={'upper_var':self.upper_var,
												'lower_var':self.lower_var,
												'train_mode': False},
									metric=name,
									condition=condition)



	def main(self):
		print(f'==> Mode: {self.mode}')
		if self.mode == 'train':
			self.train()

	def train(self):
		self.upper_optimizer.zero_grad()
		if self.counter==0:
			self.alg_time = 0.
		while self.counter<=self.count_max:
			for batch_idx, data in enumerate(self.upper_loader):
				if self.counter>self.count_max:
					break					
				self.counter +=1
				loss,acc,lower_loss= self.iteration(data)
				metrics = { 'train_upper_loss': loss.item(),
							'train_lower_loss': lower_loss.item(),
							'train_upper_acc': 100.*acc.item()}
				self.metrics.eval_metrics(self.counter,metrics)
				if self.args.metrics.log_artifacts and self.counter % self.args.metrics.log_artefacts_freq==0:
					self.log_image(self.counter)
			self.update_schedule()
			self.disp_metrics()
			self.epoch += 1
	def zero_grad(self):
		self.upper_optimizer.zero_grad()
		for p in self.lower_var:
			p.grad = None
	def update_lower_var(self,opt_lower_var):
		for p,new_p in zip(self.lower_var,opt_lower_var):
			p.data.copy_(new_p.data)
	def iteration(self,data):
		start_time_iter = time.time()
		data = utils.to_device(data,self.device,self.dtype)
		
		self.zero_grad()
		params = self.lower_var + self.upper_var
		
		
		#zeros_lower_var = tuple([torch.zeros_like(p, requires_grad=True) for p in self.lower_var])

		#params = zeros_lower_var + self.upper_var

		opt_lower_var = self.selection(*params)
		
		loss,acc = self.upper_loss(data,self.upper_var,opt_lower_var, with_acc=True)		
		loss.backward()
		self.upper_optimizer.step()
		lower_loss = loss
		self.update_lower_var(opt_lower_var)
		#print(self.lower_var[0])
		
		end_time_iter = time.time()
		self.alg_time += end_time_iter-start_time_iter
		return   loss.detach(),acc.detach(),lower_loss.detach()

	def update_schedule(self):
		if self.use_upper_scheduler:
			self.upper_scheduler.step()

	def disp_metrics(self):

		metrics = self.metrics.avg_metrics()
		metrics.update({'iter': self.counter, 'time': self.alg_time, 'epoch':self.epoch})
		#metrics.update(self.get_grad_counts())
		self.log_metrics(metrics)
		self.timer(self.counter, "Epoch %d | Alg Time %.2f | upper loss: %.2f, upper acc: %.2f | test loss: %.2f, test acc: %.2f" % 
							(self.epoch,self.alg_time,
							metrics['train_upper_loss'], metrics['train_upper_acc'],
							metrics['test_upper_loss_all'], metrics['test_upper_acc_all']))

	def log_image(self,step):
		if self.args.loss.objective.lower.name=='LogisticDistill':
			x,_,y =self.upper_var
			self.log_artifacts({'image': x.cpu().detach().numpy(), 'label':y.cpu().detach().numpy()},step, art_type='arrays', tag='distilled_image')
			N_h = 2
			fig = make_and_save_grid_images(x.cpu().detach(), N_h=N_h,N_w=int(y.shape[0]/N_h))
			self.log_artifacts(fig,step, art_type='figures', tag='distilled_image')


	def set_count_max(self):
		total_batches = self.meta_data['total_samples']/self.meta_data['b_size']
		if self.args.training.by_epoch:
			return self.args.training.total_epoch*total_batches, total_batches
		else:
			return self.args.training.upper_iterations, total_batches





	# def get_grad_counts(self):
	# 	return self.hyperloss.get_grad_counts()

