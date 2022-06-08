#!/bin/bash

log_name=rebuttal_AID_CG_nowarm_2
epochs=50
outer_n_iter=3200
inner_forward_lr=0.9
inner_backward_lr=0.9
outer_lr=1.
inner_forward_n_iter=10
inner_backward_n_iter=10
outer_cond=10
inner_dim=1000
outer_dim=2000
hours=1
cpus=1
gpumem=2
cluster_name='thoth'
device=-2
dtype=64


## Constant iterations
python launch_jobs.py --multirun solver=AID-CG-nowarm \
		solver.outer.epochs=$epochs\
		logs.log_name=$log_name\
		loss.outer.cond=$outer_cond\
		loss.inner.dim=$inner_dim\
		loss.outer.dim=$outer_dim\
		solver.inner_forward.lr=$inner_forward_lr\
		solver.inner_backward.lr=$inner_backward_lr\
		solver.outer.lr=$outer_lr\
		solver.outer.n_iter=$outer_n_iter\
		solver.inner_forward.warm_start=false\
 		loss.inner.cond=1.001,10,100,1000,10000,100000,1000000,10000000\
		solver.inner_forward.n_iter=1,10,100,1000\
		solver.inner_backward.n_iter=1,10,100,1000\
 		cluster.name=$cluster_name\
 		launcher.hours=$hours \
 		launcher.cpus=$cpus\
 		system.device=$device


# ## Increasing iterations
python launch_jobs.py --multirun solver=AID-CG-nowarm \
		solver.outer.epochs=$epochs\
		logs.log_name=$log_name\
		loss.outer.cond=$outer_cond\
		loss.inner.dim=$inner_dim\
		loss.outer.dim=$outer_dim\
		solver.inner_forward.lr=$inner_forward_lr\
		solver.inner_backward.lr=$inner_backward_lr\
		solver.outer.lr=$outer_lr\
		solver.outer.n_iter=$outer_n_iter\
		solver.inner_forward.warm_start=false\
		solver.inner_forward.increase_n_iter=true\
		solver.inner_forward.fac_increase=1000\
 		loss.inner.cond=1.001,10,100,1000,10000,100000,1000000,10000000\
		solver.inner_forward.n_iter=$inner_forward_n_iter\
		solver.inner_backward.n_iter=1,10,100,1000\
 		cluster.name=$cluster_name\
 		launcher.hours=$hours \
 		launcher.cpus=$cpus\
 		system.device=$device



log_name=debug




# HYDRA_FULL_ERROR=1   OC_CAUSE=1 python -m ipdb run.py --multirun solver=AID-CG-nowarm \
# 		solver.outer.epochs=$epochs\
# 		logs.log_name=$log_name\
# 		loss.outer.cond=$outer_cond\
# 		loss.inner.dim=$inner_dim\
# 		loss.outer.dim=$outer_dim\
# 		solver.inner_forward.lr=$inner_forward_lr\
# 		solver.inner_backward.lr=$inner_backward_lr\
# 		solver.outer.lr=$outer_lr\
# 		solver.outer.n_iter=$outer_n_iter\
#  		loss.inner.cond=1000.\
# 		solver.inner_forward.n_iter=1\
# 		solver.inner_backward.n_iter=1000\
#  		system.device=$device









