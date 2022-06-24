from trainers.utils.loaders import Loader, RingIteratorList, RingIterator

from core import utils
import torch
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
										index=i) for (data,i) in train_data ]
	test_loaders 	 = [RingIterator(torch.utils.data.DataLoader(dataset=data, batch_size=b_size, **loader_kwargs),
										index=i) for (data,i) in test_data ]

	if subset_id>=0:
		val_loaders  = [RingIterator(torch.utils.data.DataLoader(dataset=train_data[subset_id][0], batch_size=b_size, **loader_kwargs),
										index=subset_id)]
	else:
		train_loaders = [RingIterator(torch.utils.data.DataLoader(dataset=data, batch_size=b_size, **loader_kwargs),
										index=i) for (data,i) in train_data ]

	train_loaders 	 = RingIteratorList(train_loaders)
	val_loaders 	 = RingIteratorList(val_loaders)
	test_loaders 	 = RingIteratorList(test_loaders)

	data = {'lower_loader':train_loaders,
		 'upper_loader':val_loaders,
		 'test_upper_loader': test_loaders,
		 'test_lower_loader': None,
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
										index=data[1]) for data in one_train_data ]
	one_data_train_loader = RingIteratorList(one_data_train_loader)


	one_data_val_loader = [RingIterator([data[0]],
										index=data[1]) for data in one_val_data ]
	one_data_val_loader = RingIteratorList(one_data_val_loader)


	one_data_test_loader = [RingIterator([data[0]],
										index=data[1]) for data in one_test_data ]
	one_data_test_loader = RingIteratorList(one_data_test_loader)

	data = {'lower_loader':one_data_train_loader,
		 'upper_loader':one_data_val_loader,
		 'test_upper_loader': one_data_test_loader,
		 'test_lower_loader': None,
		}
	meta_data = {'num_tasks': num_tasks, 
				 'subset_id': subset_id, 
				 'total_samples': b_size,
				 'b_size': b_size }

	return data, meta_data
