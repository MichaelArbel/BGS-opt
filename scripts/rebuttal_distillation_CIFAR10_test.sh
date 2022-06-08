#!/bin/bash

log_name=rebuttal_distillation_FashionMNIST
epochs=50
outer_n_iter=4000
inner_forward_lr=100.
inner_backward_lr=1.
outer_lr=.0000000001
inner_forward_n_iter=10
inner_backward_n_iter=1
hours=1
cpus=1
gpumem=2
cluster_name=''
device=-1
dtype=64
loss='data_distillation'
data='distill_FashionMNIST'
max_outer_iter=1
max_inner_iter=1
b_size=60000
name=''
# ## Constant iterations


# python launch_jobs.py --multirun solver=BSA\
# 		solver.outer.epochs=$epochs\
# 		logs.log_name=$log_name\
# 		loss=$loss\
# 		data=$data\
# 		data.b_size=$b_size\
# 		solver.inner_forward.lr=$inner_forward_lr\
# 		solver.inner_backward.lr=$inner_backward_lr\
# 		solver.outer.lr=$outer_lr\
# 		solver.outer.n_iter=$outer_n_iter\
# 		solver.inner_forward.n_iter=$inner_forward_n_iter\
# 		solver.inner_backward.n_iter=$inner_backward_n_iter\
#  		system.device=$device\
#  		metrics.max_outer_iter=$max_outer_iter\
#  		metrics.max_inner_iter=$max_inner_iter\
# 		cluster.name=$name



# python launch_jobs.py --multirun solver=ASIG,ASIG-CG,BSA,AID-CG,AID-FP,AID-GD,stocBiO\
# 		solver.outer.epochs=$epochs\
# 		logs.log_name=$log_name\
# 		loss=$loss\
# 		data=$data\
# 		data.b_size=$b_size\
# 		solver.inner_forward.lr=$inner_forward_lr\
# 		solver.inner_backward.lr=$inner_backward_lr\
# 		solver.outer.lr=$outer_lr\
# 		solver.outer.n_iter=$outer_n_iter\
# 		solver.inner_forward.n_iter=$inner_forward_n_iter\
# 		solver.inner_backward.n_iter=$inner_backward_n_iter\
#  		system.device=$device\
#  		metrics.max_outer_iter=$max_outer_iter\
#  		metrics.max_inner_iter=$max_inner_iter\
# 		cluster.name=$name




# python launch_jobs.py --multirun solver=ITD,reverse\
# 		solver.outer.epochs=$epochs\
# 		logs.log_name=$log_name\
# 		loss=$loss\
# 		data=$data\
# 		data.b_size=$b_size\
# 		solver.inner_forward.lr=$inner_forward_lr\
# 		solver.inner_backward.lr=$inner_backward_lr\
# 		solver.outer.lr=$outer_lr\
# 		solver.outer.n_iter=$outer_n_iter\
# 		solver.inner_forward.n_iter=$inner_forward_n_iter\
#  		system.device=$device\
#  		metrics.max_outer_iter=$max_outer_iter\
#  		metrics.max_inner_iter=$max_inner_iter\
# 		cluster.name=$name

# python launch_jobs.py --multirun solver=TTSA\
# 		solver.outer.epochs=$epochs\
# 		logs.log_name=$log_name\
# 		loss=$loss\
# 		data=$data\
# 		data.b_size=$b_size\
# 		solver.inner_forward.lr=inner_forward_lr\
# 		solver.inner_backward.lr=$inner_backward_lr\
# 		solver.outer.lr=$outer_lr\
# 		solver.outer.n_iter=$outer_n_iter\
# 		solver.inner_forward.n_iter=1\
# 		solver.inner_backward.n_iter=$inner_backward_n_iter\
#  		system.device=$device\
#  		metrics.max_outer_iter=$max_outer_iter\
#  		metrics.max_inner_iter=$max_inner_iter\
# 		cluster.name=$name


# python launch_jobs.py --multirun solver=ASIG,ASIG-CG,BSA,AID-CG,AID-FP,AID-GD,stocBiO\
# 		solver.outer.epochs=$epochs\
# 		logs.log_name=$log_name\
# 		loss=$loss\
# 		data=$data\
# 		data.b_size=60000\
# 		solver.inner_forward.lr=$inner_forward_lr\
# 		solver.inner_backward.lr=$inner_backward_lr\
# 		solver.outer.lr=0.01,0.001,0.0001\
# 		solver.outer.n_iter=$outer_n_iter\
# 		solver.inner_forward.n_iter=1,5,10,20\
# 		solver.inner_backward.n_iter=1,5,10,20\
#  		system.device=$device\
#  		metrics.max_outer_iter=$max_outer_iter\
#  		metrics.max_inner_iter=$max_inner_iter\
# 		cluster.name=$name




# python launch_jobs.py --multirun solver=ITD,reverse\
# 		solver.outer.epochs=$epochs\
# 		logs.log_name=$log_name\
# 		loss=$loss\
# 		data=$data\
# 		data.b_size=60000\
# 		solver.inner_forward.lr=$inner_forward_lr\
# 		solver.inner_backward.lr=$inner_backward_lr\
# 		solver.outer.lr=0.01,0.001,0.0001\
# 		solver.outer.n_iter=$outer_n_iter\
# 		solver.inner_forward.n_iter=1,5,10,20\
#  		system.device=$device\
#  		metrics.max_outer_iter=$max_outer_iter\
#  		metrics.max_inner_iter=$max_inner_iter\
# 		cluster.name=$name

# python launch_jobs.py --multirun solver=TTSA\
# 		solver.outer.epochs=$epochs\
# 		logs.log_name=$log_name\
# 		loss=$loss\
# 		data=$data\
# 		data.b_size=60000\
# 		solver.inner_forward.lr=0.1,100.,10.,1.\
# 		solver.inner_backward.lr=$inner_backward_lr\
# 		solver.outer.lr=0.01,0.001,0.0001\
# 		solver.outer.n_iter=$outer_n_iter\
# 		solver.inner_forward.n_iter=1\
# 		solver.inner_backward.n_iter=1,5,10,20\
#  		system.device=$device\
#  		metrics.max_outer_iter=$max_outer_iter\
#  		metrics.max_inner_iter=$max_inner_iter\
# 		cluster.name=$name





log_name=debug


HYDRA_FULL_ERROR=1   OC_CAUSE=1 python -m ipdb run.py --multirun solver=ASIG-CG\
		solver.outer.epochs=$epochs\
		logs.log_name=$log_name\
		loss=$loss\
		data=$data\
		data.b_size=$b_size\
		solver.inner_forward.lr=$inner_forward_lr\
		solver.inner_backward.lr=$inner_backward_lr\
		solver.outer.lr=$outer_lr\
		solver.outer.n_iter=$outer_n_iter\
		solver.inner_forward.n_iter=$inner_forward_n_iter\
		solver.inner_backward.n_iter=$inner_backward_n_iter\
 		system.device=$device\
 		metrics.max_outer_iter=$max_outer_iter\
 		metrics.max_inner_iter=$max_inner_iter\






