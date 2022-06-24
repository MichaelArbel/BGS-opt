import  torch 
from torch.autograd import Variable
import torch.nn as nn
from torch import autograd
from core.utils import grad_with_none, grad_unused_zero
import core.utils as utils
from core.utils import Config
import higher
from torch.autograd import grad as torch_grad
import copy
import numpy as np

import time
from functools import partial
import functorch
import TorchOpt
from itertools import cycle



class LinearSolverAlg(object):
	def __call__(self,res_op,init):
		raise NotImplementedError
class SGD(LinearSolverAlg):
	def __init__(self,lr=0.1,n_iter=1):
		super(SGD,self).__init__()
		self.n_iter= n_iter
		self.lr= lr
	def __call__(self,res_op,init):
		sol = init
		for i in range(self.n_iter):
			vhp = res_op(sol)
			sol = [ ag - self.lr*g if g is not None else 1.*ag for ag,g in zip(sol,vhp)]
		return sol

