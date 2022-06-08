from copy import deepcopy
from scipy.optimize import minimize

import torch
import torch.nn.functional as F
import numpy as np

import inspect

from core import utils


class Metrics(object):
	def __init__(self,args, device, dtype):
		self.args = args
		self.device = device
		self.dtype = dtype	
		self.metrics = []
		self.values = {}
		self.count_values = {}
	def register_metric(self,func, loader, max_iters,prefix,func_args = {}, condition=None, metric='value'):
		if condition is None:
			condition = lambda counter: True
		metric = {	'func': func,
					'loader': loader,
					'max_iters':max_iters,
					'func_args':  func_args,
					'prefix': prefix,
					'condition': condition,
					'metric':metric}
		self.metrics.append(metric)
	def eval_metrics(self, counter, values):
		out_dicts = [values]
		for metric_dict in self.metrics:
			if metric_dict['condition'](counter):
				metric = globals()[metric_dict['metric']]
				out_dict =  metric(metric_dict['func'], 
									  metric_dict['loader'],
									  metric_dict['func_args'],
									  metric_dict['max_iters'], 
									  metric_dict['prefix'], self.device, self.dtype) 
				out_dicts.append(out_dict)	
		out_dicts = {k:v for d in out_dicts for k, v in d.items()}

		for k,val in out_dicts.items():
			if k in self.values:
				self.values[k]+=val
				self.count_values[k] +=1
			else:
				self.values[k] = val
				self.count_values[k] =1
		#self.values.append(out_dicts)
		
	def avg_metrics(self):
		avg_dict = {key:value/self.count_values[key] for key,value in self.values.items()}
		self.values = {}
		return avg_dict


def cond(func,loader,func_args, max_iter,prefix, device, dtype):
		kappa = utils.cond_loss(func,loader,**func_args)
		out_dict = {prefix:kappa.item()}
		return out_dict


def value(func,loader,func_args, max_iter, prefix, device, dtype):
	value = 0
	accuracy = 0
	counter = 0
	args = inspect.getfullargspec(func)[0]
	func.eval()
	def eval_func(data,value, accuracy,counter):
		if len(data)==1 and isinstance(data, list):
			data = data[0]
		data = utils.to_device(data,device,dtype)
		counter += 1
		if 'with_acc' in args:
			loss,acc = func(data,with_acc=True,**func_args)
		else:
			loss = func(data,**func_args)
			acc = 0
		value = value + loss
		accuracy = accuracy + acc
		return value, accuracy, counter

	with torch.no_grad():
		if max_iter>0:
			while counter < max_iter:
				data = next(loader)
				value, accuracy,counter = eval_func(data, value, accuracy,counter)
		else:
			for data in loader:
				value, accuracy,counter = eval_func(data, value, accuracy,counter)
		
	accuracy = accuracy/counter
	value = value/counter
	if 'with_acc' in args:
		out_dict = {prefix+'_loss': value.item(), prefix+'_acc': 100*accuracy.item()}
	else:

		out_dict = {prefix+'_loss': value.item()}
	func.train()
	return out_dict





def multivalue(func,loader,func_args, max_iter, prefix, device, dtype):
	value = 0
	accuracy = 0
	counter = 0
	all_values = 0
	all_accuracy = 0
	args = inspect.getfullargspec(func)[0]
	func.eval()


	def eval_func(data,value, all_values, accuracy, all_accuracy, counter):
		if len(data)==1 and isinstance(data, list):
			data = data[0]
		data = utils.to_device(data,device,dtype)
		counter += 1
		loss,losses,acc,all_acc = func(data,with_acc=True,all_losses=True,**func_args)
		# else:
		# 	loss,losses = func(data,all_losses=True,**func_args)
		# 	acc = 0
		# 	all_acc = 0
		value = value + loss
		all_values = all_values+ losses
		accuracy = accuracy + acc
		all_accuracy = all_accuracy + all_acc
		return value, all_values, accuracy, all_accuracy, counter

	with torch.no_grad():
		if max_iter>0:
			while counter < max_iter:
				data = next(loader)
				value, all_values, accuracy, all_accuracy, counter = eval_func(data, value, all_values, accuracy, all_accuracy, counter)
		else:
			for data in loader:
				value, all_values, accuracy, all_accuracy, counter = eval_func(data, value, all_values, accuracy, all_accuracy, counter)

	accuracy = accuracy/counter
	value = value/counter
	all_accuracy = all_accuracy/counter
	all_values = all_values/counter
	
	all_values = torch.chunk(all_values,all_values.shape[0],dim=0)
	out_dict = {prefix+'_loss_all': value.item()}
	out_dict.update({prefix+'_loss_task_'+str(i): val.item()
						for i,val in enumerate(all_values)})
	out_dict.update({prefix+'_acc_all': 100*accuracy.item()})
	all_accuracy = torch.chunk(all_accuracy,all_accuracy.shape[0],dim=0)
	out_dict.update({prefix+'_acc_task_'+str(i): 100*acc.item() for i,acc in enumerate(all_accuracy)})
	func.train()
	return out_dict

















