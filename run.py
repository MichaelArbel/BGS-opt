
from __future__ import print_function

import scipy.stats._qmc
import argparse
import yaml
import torch
import os
import importlib
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.multiprocessing.set_sharing_strategy('file_system')
import hydra

# check whether we want to load a pretrained model depending on the given parameters


work_dir = os.getcwd()

@hydra.main(config_name='config.yaml',config_path='./configs')
def run(cfg):
	os.chdir(work_dir)
	module, attr = os.path.splitext(cfg.training.trainer_name)
	module = importlib.import_module(module)
	Trainer = getattr(module, attr[1:])
	trainer = Trainer(cfg)
	trainer.main()
	print('Finished!')

if __name__ == "__main__":

    run()

