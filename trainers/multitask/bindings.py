from trainers.utils.loaders import Loader, RingIteratorList, RingIterator

from core.losses import Loss
from core import utils
import torch
import dependencies.autolambda.create_network as models 
import dependencies.autolambda.create_dataset as datasets 

import importlib
import os
from torchvision import transforms
import torch.nn.functional as F
import pickle as pkl

class MutiTaskLoader(Loader):
	def __init__(self,num_workers,dtype,device,**args):
		super(MutiTaskLoader,self).__init__(num_workers,dtype,device,**args)
		self.make_loaders()
	def load_data(self,b_size):
		return data_multitask(self.args,b_size, self.dtype, self.device, self.num_workers)


class MutiTaskLoss(Loss):
	def __init__(self,outer_model,inner_model,num_tasks=1,weighted=True,reg=0., apply_reg=False, device=None):
		super(MutiTaskLoss,self).__init__(outer_model,inner_model,device=device)
		self.weighted = weighted
		self.reg= reg
		self.apply_reg= apply_reg
		self.num_tasks = num_tasks
	def format_data(self,data):
		tasks = torch.cat([d[1].repeat(d[0][0].shape[0]) for d in data],dim=0)
		tasks_onehot = torch.nn.functional.one_hot(tasks, num_classes=self.num_tasks)
		tasks_onehot  = utils.to_type(tasks_onehot,data[0][0][0].dtype)
		all_x = torch.cat([d[0][0] for d in data],dim=0)
		all_y = torch.cat([d[0][1].long() for d in data],dim=0)
		return all_x, all_y,tasks_onehot,tasks

	def forward(self,data,with_acc=False, all_losses=False):
		all_x, all_y,tasks,tasks_onehot = self.format_data(data)

		preds = self.inner_model(all_x,tasks,tasks_onehot)
		losses =  F.cross_entropy(preds, all_y, ignore_index=-1,reduction='none')

		sum_tasks = torch.sum(tasks,dim=0)
		inv_sum_tasks = 1./sum_tasks
		inv_sum_tasks[sum_tasks==0] = 0
		losses = torch.einsum('i,ik->k',losses,tasks)*inv_sum_tasks

		if self.weighted:
			loss = torch.einsum('i,i->',losses,self.outer_params[0])
		else:
			loss = torch.sum(losses)
		if with_acc:
			all_preds = preds.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
			all_acc = 1.*all_preds.eq(all_y.view_as(all_preds))
			acc = torch.mean(all_acc)
			all_acc = torch.einsum('id,ik->k', all_acc, tasks)*inv_sum_tasks
			

		if self.apply_reg:
			loss = loss + self.reg_inner()

		if all_losses:
			if with_acc:
				return loss,losses,acc,all_acc
			else:
				return loss,losses
		else:
			if with_acc:
				return loss,acc
			else:
				return loss

	# def forward(self,data,with_acc=False, all_losses=False):
	# 	#all_x, all_y,tasks,tasks_onehot = self.format_data(data)

	# 	preds = [self.inner_model(d[0][0],d[1]) for d in data]

	# 	losses =  torch.stack([F.cross_entropy(p, d[0][1].long(), ignore_index=-1) for p,d in zip(preds,data)],dim=0)

	# 	if self.weighted:
	# 		loss = torch.einsum('i,i->',losses,self.outer_params[0])
	# 	else:
	# 		loss = torch.sum(losses)
	# 	if with_acc:
	# 		all_preds = [p.argmax(dim=1, keepdim=True) for p in preds]  # get the index of the max log-probability
	# 		all_acc = torch.stack([torch.mean(1.*pp.eq(dd[0][1].view_as(pp))) for pp, dd in zip(all_preds, data)],dim=0)
	# 		#all_acc = torch.einsum('id,ik->k', all_acc, tasks)*inv_sum_tasks
	# 		acc = torch.mean(all_acc)
	# 	if self.apply_reg:
	# 		loss = loss + self.reg_inner()

	# 	if all_losses:
	# 		if with_acc:
	# 			return loss,losses,acc,all_acc
	# 		else:
	# 			return loss,losses
	# 	else:
	# 		if with_acc:
	# 			return loss,acc
	# 		else:
	# 			return loss


	def reg_outer(self):
		return 0.5*self.reg*torch.sum(self.outer_params[0]**2)

	def reg_inner(self):
		l2_penalty = 0.
		for p in self.inner_params:
			l2_penalty += torch.sum(p**2)
		return 0.5*self.reg*l2_penalty

def augmentations(name, dataset_type='train'):
	if name == 'CIFAR100MTL':
		if dataset_type=='train':
			return transforms.Compose([
				    transforms.RandomCrop(32, padding=4),
				    transforms.RandomHorizontalFlip(),
				    transforms.ToTensor(),
				    transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
					])
		else:
			return transforms.Compose([
			    	transforms.ToTensor(),
			    	transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
					])


def data_multitask(args,b_size,dtype, device, num_workers):
	download = False
	num_tasks = args['num_tasks']
	subset_id = args['subset_id']
	data_path = args['data_path']
	name = args['name']

	work_dir = os.getcwd()
	root = os.path.join(work_dir,data_path,name)
	path = os.path.join(work_dir,data_path,name+'.pkl')
	try:
		with open(path,'rb') as f:
			all_data = pkl.load(f)
			train_data, test_data = all_data
	except:


		module, attr = os.path.splitext(name)
		name = attr[1:]
		try: 
			module = importlib.import_module(module)
		except:
			module = globals()[module]
		klass = getattr(module, name)
		train_data = 	[(klass(root, train = True,transform=augmentations(name, dataset_type='train'), subset_id=i, download = download),i) for i in range(num_tasks)]
		if subset_id>=0:
			test_data = [(klass(root, train = False,transform=augmentations(name, dataset_type='test'), subset_id=subset_id, download = download),subset_id)]
		else:
			test_data = [(klass(root, train = False,transform=augmentations(name, dataset_type='test'), subset_id=i, download = download),i) for i in range(num_tasks)]
		all_data = train_data, test_data
		with open(path,'wb') as f:
			pkl.dump(all_data,f)


	loader_kwargs = {'shuffle':True, 'num_workers':2, 'pin_memory': True}

	train_loaders 	 = [RingIterator(torch.utils.data.DataLoader(dataset=data, batch_size=b_size, **loader_kwargs),
										task=i,device=device,dtype=dtype) for (data,i) in train_data ]
	test_loaders 	 = [RingIterator(torch.utils.data.DataLoader(dataset=data, batch_size=b_size, **loader_kwargs),
										task=i,device=device,dtype=dtype) for (data,i) in test_data ]

	if subset_id>=0:
		val_loaders  = [RingIterator(torch.utils.data.DataLoader(dataset=train_data[subset_id][0], batch_size=b_size, **loader_kwargs),
										task=subset_id,device=device,dtype=dtype)]
	else:
		train_loaders = [RingIterator(torch.utils.data.DataLoader(dataset=data, batch_size=b_size, **loader_kwargs),
										task=i,device=device,dtype=dtype) for (data,i) in train_data ]

	train_loaders 	 = RingIteratorList(train_loaders,device=device,dtype=dtype)
	val_loaders 	 = RingIteratorList(val_loaders,device=device,dtype=dtype)
	test_loaders 	 = RingIteratorList(test_loaders,device=device,dtype=dtype)

	data = {'inner_loader':train_loaders,
		 'outer_loader':val_loaders,
		 'test_outer_loader': test_loaders,
		 'test_inner_loader': None,
		}
	meta_data = {'num_tasks': num_tasks, 
				 'subset_id': subset_id, 
				 'total_samples': len(train_data[0][0]),
				 'b_size': b_size }
	one_data_path = os.path.join(work_dir,data_path,name+'_one_data.pkl')
	try:
		with open(one_data_path,'rb') as f:
			all_data = pkl.load(f)
			one_train_data,one_val_data,one_test_data = all_data
	except:
		one_train_data = next(train_loaders)
		one_val_data = (one_train_data[subset_id],)
		one_test_data = next(test_loaders)
		all_data = one_train_data,one_val_data,one_test_data
		with open(one_data_path,'wb') as f:
			pkl.dump(all_data,f)


	one_data_train_loader = [RingIterator([data[0]],
										task=data[1],device=device,dtype=dtype) for data in one_train_data ]
	one_data_train_loader = RingIteratorList(one_data_train_loader,device=device,dtype=dtype)


	one_data_val_loader = [RingIterator([data[0]],
										task=data[1],device=device,dtype=dtype) for data in one_val_data ]
	one_data_val_loader = RingIteratorList(one_data_val_loader,device=device,dtype=dtype)


	one_data_test_loader = [RingIterator([data[0]],
										task=data[1],device=device,dtype=dtype) for data in one_test_data ]
	one_data_test_loader = RingIteratorList(one_data_test_loader,device=device,dtype=dtype)

	data = {'inner_loader':one_data_train_loader,
		 'outer_loader':one_data_val_loader,
		 'test_outer_loader': one_data_test_loader,
		 'test_inner_loader': None,
		}
	meta_data = {'num_tasks': num_tasks, 
				 'subset_id': subset_id, 
				 'total_samples': b_size,
				 'b_size': b_size }

	return data, meta_data



# def data_multitask(args,b_size,dtype, device, num_workers):
# 	download = False
# 	num_tasks = args['num_tasks']
# 	subset_id = args['subset_id']
# 	path = args['data_path']
# 	name = args['name']

# 	work_dir = os.getcwd()
# 	root = os.path.join(work_dir,path,name)
# 	path = os.path.join(work_dir,path,name+'.pkl')
# 	try:
# 		with open(path,'rb') as f:
# 			all_data = pkl.load(f)
# 			train_data, test_data = all_data
# 	except:


# 		module, attr = os.path.splitext(name)
# 		name = attr[1:]
# 		try: 
# 			module = importlib.import_module(module)
# 		except:
# 			module = globals()[module]
# 		klass = getattr(module, name)
# 		train_data = 	[(klass(root, train = True,transform=augmentations(name, dataset_type='train'), subset_id=i, download = download),i) for i in range(num_tasks)]
# 		if subset_id>=0:
# 			test_data = [(klass(root, train = False,transform=augmentations(name, dataset_type='test'), subset_id=subset_id, download = download),subset_id)]
# 		else:
# 			test_data = [(klass(root, train = False,transform=augmentations(name, dataset_type='test'), subset_id=i, download = download),i) for i in range(num_tasks)]
# 		all_data = train_data, test_data
# 		#with open(path,'wb') as f:
# 		#	pkl.dump(all_data,f)


# 	loader_kwargs = {'shuffle':True, 'num_workers':2}

# 	train_loaders 	 = [RingIterator(torch.utils.data.DataLoader(dataset=train_data[0], batch_size=b_size, **loader_kwargs),
# 										task=0,device=device,dtype=dtype)]
# 	test_loaders 	 = [RingIterator(torch.utils.data.DataLoader(dataset=train_data[0], batch_size=b_size, **loader_kwargs),
# 										task=0,device=device,dtype=dtype)]
# 	val_loaders 	 = [RingIterator(torch.utils.data.DataLoader(dataset=train_data[0], batch_size=b_size, **loader_kwargs),
# 										task=0,device=device,dtype=dtype)]


# 	data = {'inner_loader':train_loaders,
# 		 'outer_loader':val_loaders,
# 		 'test_outer_loader': test_loaders,
# 		 'test_inner_loader': None,
# 		}
# 	meta_data = {'num_tasks': num_tasks, 
# 				 'subset_id': subset_id, 
# 				 'total_samples': len(train_data[0][0]),
# 				 'b_size': b_size }

# 	return data, meta_data





