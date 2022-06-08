#!/bin/bash

log_name=20newsgroup_gpu_ttsa
outer_n_iter=10000
dtype=32
loss=kernel_ridge
inner_forward_lr=100.
inner_backward_lr=0.5
outer_lr=100.
disp_freq=10
inner_backward_fac_increase=0
inner_forward_n_iter=10
inner_backward_n_iter=10
data=20newsgroup
device=-1

hours=1
cpus=1
gpumem=2
cluster_name=''


# python  launch_jobs.py --multirun solver=BSA,AID-CG,AID-FP,AID-GD,stocBiO \
# 		loss=$loss \
# 		data=$data \
# 		logs.log_name=$log_name \
# 		solver.inner_forward.lr=$inner_forward_lr \
# 		solver.inner_backward.lr=0.5,10 \
# 		solver.outer.lr=$outer_lr \
# 		solver.outer.n_iter=$outer_n_iter \
# 		system.dtype=$dtype \
# 		system.device=$device\
# 		cluster.name=$cluster_name\
# 		metrics.disp_freq=$disp_freq \
# 		solver.inner_forward.n_iter=$inner_forward_n_iter \
# 		solver.inner_backward.n_iter=$inner_backward_n_iter \
# 		data.b_size=100,1000,2000,4000

# python  launch_jobs.py --multirun solver=ITD,reverse \
# 		loss=$loss \
# 		data=$data \
# 		logs.log_name=$log_name \
# 		solver.inner_forward.lr=$inner_forward_lr \
# 		solver.inner_backward.lr=$inner_backward_lr \
# 		solver.outer.lr=$outer_lr \
# 		solver.outer.n_iter=$outer_n_iter \
# 		system.dtype=$dtype \
# 		system.device=$device\
# 		cluster.name=$cluster_name\
# 		metrics.disp_freq=$disp_freq \
# 		solver.inner_forward.n_iter=$inner_forward_n_iter \
# 		data.b_size=100,1000,2000,4000


# python  launch_jobs.py --multirun solver=ASIG,ASIG-CG \
# 		loss=$loss \
# 		data=$data \
# 		logs.log_name=$log_name \
# 		solver.inner_forward.lr=$inner_forward_lr \
# 		solver.inner_backward.lr=0.5,10. \
# 		solver.outer.lr=$outer_lr \
# 		solver.outer.n_iter=$outer_n_iter \
# 		system.dtype=$dtype \
# 		system.device=$device\
# 		cluster.name=$cluster_name\
# 		metrics.disp_freq=$disp_freq \
# 		solver.inner_forward.n_iter=10,20 \
# 		solver.inner_backward.n_iter=5,10 \
# 		data.b_size=100,1000,2000,4000


# python  launch_jobs.py --multirun solver=ITD,reverse \
# 		loss=$loss \
# 		data=$data \
# 		logs.log_name=$log_name \
# 		solver.inner_forward.lr=$inner_forward_lr \
# 		solver.inner_backward.lr=$inner_backward_lr \
# 		solver.outer.lr=$outer_lr \
# 		solver.outer.n_iter=$outer_n_iter \
# 		system.dtype=$dtype \
# 		system.device=$device\
# 		cluster.name=$cluster_name\
# 		metrics.disp_freq=$disp_freq \
# 		solver.inner_forward.n_iter=20 \
# 		data.b_size=100,1000,2000,4000


# python  launch_jobs.py --multirun solver=BSA,AID-CG,AID-FP,AID-GD,stocBiO \
# 		loss=$loss \
# 		data=$data \
# 		logs.log_name=$log_name \
# 		solver.inner_forward.lr=$inner_forward_lr \
# 		solver.inner_backward.lr=0.5,10 \
# 		solver.outer.lr=$outer_lr \
# 		solver.outer.n_iter=$outer_n_iter \
# 		system.dtype=$dtype \
# 		system.device=$device\
# 		cluster.name=$cluster_name\
# 		metrics.disp_freq=$disp_freq \
# 		solver.inner_forward.n_iter=10,20 \
# 		solver.inner_backward.n_iter=5,10 \
# 		data.b_size=100,1000,2000,4000




# python  launch_jobs.py --multirun solver=TTSA \
# 		loss=$loss \
# 		data=$data \
# 		logs.log_name=$log_name \
# 		solver.inner_forward.lr=$inner_forward_lr \
# 		solver.inner_backward.lr=0.5,10 \
# 		solver.outer.lr=$outer_lr \
# 		solver.outer.n_iter=$outer_n_iter \
# 		system.dtype=$dtype \
# 		system.device=$device\
# 		cluster.name=$cluster_name\
# 		metrics.disp_freq=$disp_freq \
# 		solver.inner_forward.n_iter=1 \
# 		solver.inner_backward.n_iter=5,10 \
# 		data.b_size=100,1000,2000,4000


# HYDRA_FULL_ERROR=1   OC_CAUSE=1 python -m ipdb  run.py --multirun solver=TTSA \
# 		loss=$loss \
# 		data=$data \
# 		logs.log_name=$log_name \
# 		solver.inner_forward.lr=$inner_forward_lr \
# 		solver.inner_backward.lr=0.5,10 \
# 		solver.outer.lr=$outer_lr \
# 		solver.outer.n_iter=$outer_n_iter \
# 		system.dtype=$dtype \
# 		system.device=$device\
# 		cluster.name=$cluster_name\
# 		metrics.disp_freq=$disp_freq \
# 		solver.inner_forward.n_iter=1 \
# 		solver.inner_backward.n_iter=5,10 \
# 		data.b_size=100,1000,2000,4000




# log_name=20newsgroup_vrbo_cpu
# outer_n_iter=1000
# dtype=32
# loss=kernel_ridge
# inner_forward_lr=100.
# inner_backward_lr=0.5
# outer_lr=100.
# disp_freq=10
# inner_backward_fac_increase=0
# inner_forward_n_iter=10
# inner_backward_n_iter=10
# data=20newsgroup
# device=-2

# hours=1
# cpus=3
# gpumem=null
# cluster_name='thoth'

# python launch_jobs.py --multirun solver=VRBO \
# 		loss=$loss \
# 		data=$data \
# 		logs.log_name=$log_name \
# 		solver.inner_forward.lr=$inner_forward_lr \
# 		solver.inner_backward.lr=0.5,10 \
# 		solver.outer.lr=$outer_lr \
#  		solver.outer.n_iter=$outer_n_iter \
# 		system.dtype=$dtype \
# 		metrics.disp_freq=$disp_freq \
# 		solver.inner_forward.n_iter=1,5,10\
# 		solver.inner_backward.n_iter=1,5,10\
# 		data.b_size=100,1000,2000,4000\
# 		system.device=$device\
# 		solver.outer.spider_epoch=1,2,5\
#  		cluster.name=$cluster_name\

# 		loss=$loss \
# 		data=$data \
# 		logs.log_name=$log_name \
# 		solver.inner_forward.lr=$inner_forward_lr \
# 		solver.inner_backward.lr=0.5,10 \
# 		solver.outer.lr=$outer_lr \
# 		solver.outer.n_iter=$outer_n_iter \
# 		system.dtype=$dtype \
# 		system.device=$device\
# 		cluster.name=$cluster_name\
# 		metrics.disp_freq=$disp_freq \
# 		solver.inner_forward.n_iter=$inner_forward_n_iter \
# 		solver.inner_backward.n_iter=$inner_backward_n_iter \
# 		data.b_size=100,1000,2000,4000


# # ##############   5*(       2+1+8   )
log_name=debug
outer_n_iter=100
dtype=64
loss=kernel_ridge
training=SGD
inner_forward_lr=100.
inner_backward_lr=0.5
outer_lr=100.
disp_freq=1
inner_backward_fac_increase=0
inner_forward_n_iter=10
inner_backward_n_iter=10
data=20newsgroup
device=-1









HYDRA_FULL_ERROR=1   OC_CAUSE=1  python -m ipdb run.py --multirun method=AmIGO  \
		loss=$loss \
		data=$data \
		training=$training \
		training.optimizer.inner.lr=$inner_forward_lr \
		training.optimizer.outer.lr=$outer_lr \
 		training.outer_iterations=$outer_n_iter \
		method.backward.lr=.5 \
		method.forward.n_iter=1\
		method.backward.n_iter=1\
		data.b_size=1000\
		system.dtype=$dtype \
		system.device=$device\
		metrics.disp_freq=$disp_freq\
		logs.log_name=$log_name \





