#!/bin/bash

# log_name=miniimagenet_gpu_2
# epochs=4
# dtype=32
# loss=meta_loss_miniimagenet
# inner_forward_lr=0.05
# inner_backward_lr=0.001
# outer_lr=0.002
# meta_b_size=32
# disp_freq=$meta_b_size
# inner_forward_n_iter=10
# inner_backward_n_iter=10
# data=MiniImagenet

# # hours=15
# # device=-2
# # cpus=4
# # gpumem=null

# hours=10
# cpus=1
# gpumem=1
# device=-1
# epochs=10


# data=MiniImagenet

# #### 10 random seeds

# python  launch_jobs.py --multirun solver=ANIL\
# 		logs.log_name=$log_name \
# 		data=$data \
# 		loss=$loss \
# 		solver.outer.lr=$outer_lr \
# 		system.dtype=$dtype \
# 		system.device=$device \
# 		metrics.disp_freq=$disp_freq \
# 		data.meta_b_size=$meta_b_size \
# 		solver.outer.epochs=$epochs \
# 		solver.inner_forward.lr=0.05 \
# 		solver.inner_forward.n_iter=10 \
# 		launcher.hours=$hours \
# 		launcher.cpus=$cpus\
# 		system.seed.torch=1

		

# python  launch_jobs.py --multirun solver=MAML\
# 		data=$data \
# 		loss=$loss \
# 		logs.log_name=$log_name \
# 		system.dtype=$dtype \
# 		system.device=$device \
# 		metrics.disp_freq=$disp_freq \
# 		data.meta_b_size=$meta_b_size \
# 		solver.outer.epochs=$epochs \
# 		solver.inner_forward.lr=0.5 \
# 		solver.outer.lr=0.003 \
# 		solver.inner_forward.n_iter=3 \
# 		launcher.hours=$hours \
# 		launcher.cpus=$cpus\
# 		system.seed.torch=1,2,3,4,5,6,7,8,9,10

# python  launch_jobs.py --multirun solver=Meta_ITD \
# 		data=$data \
# 		loss=$loss \
# 		logs.log_name=$log_name \
# 		solver.inner_forward.lr=$inner_forward_lr \
# 		solver.outer.lr=$outer_lr \
# 		system.dtype=$dtype \
# 		system.device=$device \
# 		metrics.disp_freq=$disp_freq \
# 		data.meta_b_size=$meta_b_size \
# 		solver.inner_forward.n_iter=$inner_forward_n_iter \
# 		solver.outer.epochs=$epochs \
# 		launcher.hours=$hours \
# 		launcher.cpus=$cpus\
# 		system.seed.torch=1,2,3,4,5,6,7,8,9,10

# python  launch_jobs.py --multirun solver=MetaAID \
# 		data=$data \
# 		loss=$loss \
# 		logs.log_name=$log_name \
# 		solver.inner_forward.lr=$inner_forward_lr \
# 		solver.inner_backward.lr=$inner_backward_lr \
# 		solver.outer.lr=$outer_lr \
# 		system.dtype=$dtype \
# 		system.device=$device \
# 		metrics.disp_freq=$disp_freq \
# 		data.meta_b_size=$meta_b_size \
# 		solver.inner_forward.n_iter=$inner_forward_n_iter \
# 		solver.inner_backward.n_iter=$inner_backward_n_iter \
# 		solver.outer.epochs=$epochs \
# 		launcher.hours=$hours \
# 		launcher.cpus=$cpus\
# 		launcher.gpumem=$gpumem\
# 		system.seed.torch=2,3,4,5,6,7,8,9\
# 		loss.inner.reg=true
## 

# python  launch_jobs.py --multirun solver=MetaASIG \
# 		data=$data \
# 		loss=$loss \
# 		logs.log_name=$log_name \
# 		solver.inner_forward.lr=$inner_forward_lr \
# 		solver.inner_backward.lr=$inner_backward_lr \
# 		solver.outer.lr=$outer_lr \
# 		system.dtype=$dtype \
# 		system.device=$device \
# 		metrics.disp_freq=$disp_freq \
# 		data.meta_b_size=$meta_b_size \
# 		solver.inner_forward.n_iter=$inner_forward_n_iter \
# 		solver.inner_backward.n_iter=$inner_backward_n_iter \
# 		solver.outer.epochs=$epochs \
# 		launcher.hours=$hours \
# 		launcher.cpus=$cpus\
# 		launcher.gpumem=$gpumem\
# 		system.seed.torch=2,3,4,5,6,7,8,9\
# 		loss.inner.reg=true

# log_name=debug
# epochs=4
# dtype=32
# loss=meta_loss_miniimagenet
# inner_forward_lr=0.05
# inner_backward_lr=0.001
# outer_lr=0.002
# meta_b_size=32
# disp_freq=$meta_b_size
# inner_forward_n_iter=10
# inner_backward_n_iter=10
# data=MiniImagenet
# device=-1
# hours=15

# HYDRA_FULL_ERROR=1   OC_CAUSE=1 python -m ipdb run.py --multirun solver=MetaASIG \
# 		data=$data \
# 		loss=$loss \
# 		logs.log_name=$log_name \
# 		solver.inner_forward.lr=$inner_forward_lr \
# 		solver.inner_backward.lr=$inner_backward_lr \
# 		solver.outer.lr=$outer_lr \
# 		system.dtype=$dtype \
# 		system.device=$device \
# 		metrics.disp_freq=$disp_freq \
# 		data.meta_b_size=$meta_b_size \
# 		solver.inner_forward.n_iter=$inner_forward_n_iter \
# 		solver.inner_backward.n_iter=$inner_backward_n_iter \
# 		solver.outer.epochs=$epochs \
# 		loss.inner.reg=true\






log_name=debug
epochs=4
dtype=32
loss=meta_loss_regression
inner_forward_lr=0.001
inner_backward_lr=0.001
outer_lr=0.1
meta_b_size=1
disp_freq=32
inner_forward_n_iter=1
inner_backward_n_iter=1
data=MiniImagenet
device=-1
hours=15

# HYDRA_FULL_ERROR=1   OC_CAUSE=1 python -m ipdb run.py --multirun solver=MetaOptNet \
# 		data=$data \
# 		loss=$loss \
# 		logs.log_name=$log_name \
# 		solver.inner_forward.lr=$inner_forward_lr \
# 		solver.outer.lr=$outer_lr \
# 		system.dtype=$dtype \
# 		system.device=$device \
# 		metrics.disp_freq=$disp_freq \
# 		data.meta_b_size=$meta_b_size \
# 		solver.inner_forward.n_iter=$inner_forward_n_iter \
# 		solver.outer.epochs=$epochs \
# 		metrics.max_outer_iter=1\
# 		metrics.max_inner_iter=1\

HYDRA_FULL_ERROR=1   OC_CAUSE=1 python -m ipdb run.py --multirun solver=MetaFull \
		data=$data \
		loss=$loss \
		logs.log_name=$log_name \
		solver.inner_forward.lr=$inner_forward_lr \
		solver.inner_backward.lr=$inner_backward_lr \
		solver.outer.lr=$outer_lr \
		system.dtype=$dtype \
		system.device=$device \
		metrics.disp_freq=$disp_freq \
		data.meta_b_size=$meta_b_size \
		solver.inner_forward.n_iter=$inner_forward_n_iter \
		solver.inner_backward.n_iter=$inner_backward_n_iter \
		solver.outer.epochs=$epochs \
		metrics.max_outer_iter=1\
		metrics.max_inner_iter=1\
		solver.outer.nesterov=true\
		solver.outer.momentum=0.9\
		solver.outer.weight_decay=0.0005\
		loss.inner.reg=true\
		loss.inner.reg_lambda=50.

# HYDRA_FULL_ERROR=1   OC_CAUSE=1  python -m ipdb run.py --multirun solver=ANIL\
# 		data=$data \
# 		loss=$loss \
# 		logs.log_name=$log_name \
# 		system.dtype=$dtype \
# 		system.device=$device \
# 		metrics.disp_freq=$disp_freq \
# 		data.meta_b_size=$meta_b_size \
# 		solver.outer.epochs=$epochs \
# 		solver.inner_forward.lr=0.01 \
# 		solver.outer.lr=0.003 \
# 		solver.inner_forward.n_iter=10 \
# 		launcher.hours=$hours \
# 		launcher.cpus=$cpus\
# 		system.seed.torch=1



