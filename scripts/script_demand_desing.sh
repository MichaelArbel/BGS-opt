#!/bin/bash


log_name=demand_design
dtype=32
disp_freq=1
data=demand_design
loss=DFIV
device=-1
epochs=300

hours=1
cpus=1
gpumem=2
name=''


python  launch_jobs.py --multirun solver=DFIV_ASIG \
		loss=$loss \
		data=$data \
		logs.log_name=$log_name \
		metrics.disp_freq=$disp_freq \
		system.dtype=$dtype \
		system.device=$device\
		solver.outer.epochs=$epochs\
		cluster.name=$name\
		system.seed.torch=1,2,3,4,5,6,7,8,9,10\
		solver.inner_backward.n_iter=5,10,20\
		solver.inner_forward.n_iter=5,10,20,40

python  launch_jobs.py --multirun solver=DFIV_AID \
		loss=$loss \
		data=$data \
		logs.log_name=$log_name \
		metrics.disp_freq=$disp_freq \
		system.dtype=$dtype \
		system.device=$device\
		solver.outer.epochs=$epochs\
		cluster.name=''\
		system.seed.torch=1,2,3,4,5,6,7,8,9,10\
		solver.inner_backward.n_iter=5,10,20\
		solver.inner_forward.n_iter=5,10,20,40

python  launch_jobs.py --multirun solver=DFIV \
		loss=$loss \
		data=$data \
		logs.log_name=$log_name \
		metrics.disp_freq=$disp_freq \
		system.dtype=$dtype \
		system.device=$device\
		solver.outer.epochs=$epochs\
		cluster.name=''\
		system.seed.torch=1,2,3,4,5,6,7,8,9,10\
		solver.inner_forward.n_iter=5,10,20,40








# HYDRA_FULL_ERROR=1   OC_CAUSE=1  python -m ipdb  launch_jobs.py --multirun solver=DFIV_ASIG \
# 		loss=$loss \
# 		data=$data \
# 		logs.log_name=$log_name \
# 		metrics.disp_freq=$disp_freq \
# 		system.dtype=$dtype \
# 		system.device=$device\
# 		solver.outer.epochs=$epochs \
# 		system.seed.torch=1 \
# 		solver.inner_backward.n_iter=5\
# 		cluster.name=''










