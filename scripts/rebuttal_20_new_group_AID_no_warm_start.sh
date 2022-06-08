#!/bin/bash

log_name=20newsgroup_gpu_AID_no_warm_start
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
device=-2

hours=1
cpus=1
gpumem=2
cluster_name='thoth'


# python  launch_jobs.py --multirun solver=AID-CG-nowarm \
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





log_name=debug

HYDRA_FULL_ERROR=1   OC_CAUSE=1 python -m ipdb  run.py --multirun solver=AID-CG-nowarm \
		loss=$loss \
		data=$data \
		logs.log_name=$log_name \
		solver.inner_forward.lr=$inner_forward_lr \
		solver.inner_backward.lr=10 \
		solver.outer.lr=$outer_lr \
		solver.outer.n_iter=$outer_n_iter \
		system.dtype=$dtype \
		system.device=$device\
		cluster.name=$cluster_name\
		metrics.disp_freq=$disp_freq \
		solver.inner_forward.n_iter=10 \
		solver.inner_backward.n_iter=10 \
		data.b_size=1000



