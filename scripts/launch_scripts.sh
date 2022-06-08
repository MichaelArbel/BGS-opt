#!/bin/bash


# python launch_jobs.py --multirun solver=AID-FP,AID-CG,stocBiO solver.outer.epochs=50 logs.log_name=vary_cond_number_iter_high_dim  loss.outer.cond=10 loss.inner.cond=1.001,10,100,1000,10000,100000,1000000,10000000 loss.inner.dim=1000 loss.outer.dim=2000 ++solver.inner_backward.fac_increase=1,10,100,1000 solver.inner_forward.lr=0.9 solver.inner_backward.lr=0.9 solver.outer.lr=1. &&
# python launch_jobs.py --multirun solver=ITD,reverse solver.outer.epochs=50 logs.log_name=vary_cond_number_iter_high_dim  loss.outer.cond=10 loss.inner.cond=1.001,10,100,1000,10000,100000,1000000,10000000 loss.inner.dim=1000 loss.outer.dim=2000 ++solver.inner_forward.fac_increase=1,10,100,1000 solver.inner_forward.lr=0.9 solver.inner_backward.lr=0.9 solver.outer.lr=1. &&


# python launch_jobs.py --multirun solver=ITD,reverse solver.outer.epochs=50 logs.log_name=vary_cond_number_iter_high_dim loss.outer.cond=10 loss.inner.cond=1.001,10,100,1000,10000,100000,1000000,10000000 loss.inner.dim=1000 loss.outer.dim=2000 ++solver.inner_forward.fac_increase=0 solver.inner_forward.lr=0.9 solver.inner_backward.lr=0.9 solver.inner_forward.n_iter=1,10,20 solver.outer.lr=1. &&
# python launch_jobs.py --multirun solver=AID-FP,AID-CG,stocBiO solver.outer.epochs=50 logs.log_name=vary_cond_number_iter_high_dim loss.outer.cond=10  loss.inner.cond=1.001,10,100,1000,10000,100000,1000000,10000000 loss.inner.dim=1000 loss.outer.dim=2000 ++solver.inner_backward.fac_increase=0 solver.inner_forward.lr=0.9 solver.inner_backward.lr=0.9 solver.inner_forward.n_iter=1,10,20 solver.outer.lr=1. solver.inner_backward.n_iter=1,10,20



# python launch_jobs.py --multirun solver=AID-FP solver.outer.epochs=50 logs.log_name=vary_cond_number_iter_high_dim_2  loss.outer.cond=10 loss.inner.cond=1.001,10,100,1000,10000,100000,1000000,10000000 loss.inner.dim=1000 loss.outer.dim=2000 solver.inner_backward.fac_increase=1,10,100,1000 solver.inner_forward.lr=0.9 solver.inner_backward.lr=0.9 solver.outer.lr=1.   
# python launch_jobs.py --multirun solver=stocBiO solver.outer.epochs=50 logs.log_name=vary_cond_number_iter_high_dim_2  loss.outer.cond=10 loss.inner.cond=1.001,10,100,1000,10000,100000,1000000,10000000 loss.inner.dim=1000 loss.outer.dim=2000 solver.inner_backward.fac_increase=1,10,100,1000 solver.inner_forward.lr=0.9 solver.inner_backward.lr=0.9 solver.outer.lr=1. 
# python launch_jobs.py --multirun solver=AID-CG solver.outer.epochs=50 logs.log_name=vary_cond_number_iter_high_dim_2  loss.outer.cond=10 loss.inner.cond=1.001,10,100,1000,10000,100000,1000000,10000000 loss.inner.dim=1000 loss.outer.dim=2000 solver.inner_backward.fac_increase=1,10,100,1000 solver.inner_forward.lr=0.9 solver.inner_backward.lr=0.9 solver.outer.lr=1. 


# python launch_jobs.py --multirun solver=ASIG solver.outer.epochs=50 logs.log_name=vary_cond_number_iter_high_dim_2  loss.outer.cond=10 loss.inner.cond=1.001,10,100,1000,10000,100000,1000000,10000000  loss.inner.dim=1000 loss.outer.dim=2000 solver.inner_forward.lr=0.9 solver.inner_backward.lr=0.9 solver.inner_backward.n_iter=1,10,100,1000 solver.outer.lr=1. 
# python launch_jobs.py --multirun solver=ASIG-CG solver.outer.epochs=50 logs.log_name=vary_cond_number_iter_high_dim_2  loss.outer.cond=10 loss.inner.cond=1.001,10,100,1000,10000,100000,1000000,10000000  loss.inner.dim=1000 loss.outer.dim=2000 solver.inner_forward.lr=0.9 solver.inner_backward.lr=0.9 solver.inner_backward.n_iter=1,10,100,1000 solver.outer.lr=1. 


python launch_jobs.py --multirun solver=AID-FP solver.outer.epochs=50 logs.log_name=vary_cond_number_iter_high_dim_2 loss.outer.cond=10  loss.inner.cond=1.001,10,100,1000,10000,100000,1000000,10000000 loss.inner.dim=1000 loss.outer.dim=2000 solver.inner_backward.fac_increase=0 solver.inner_forward.lr=0.9 solver.inner_backward.lr=0.9 solver.inner_forward.n_iter=1,10,100,1000 solver.outer.lr=1. solver.inner_backward.n_iter=1,10,100,1000 
python launch_jobs.py --multirun solver=AID-CG solver.outer.epochs=50 logs.log_name=vary_cond_number_iter_high_dim_2 loss.outer.cond=10  loss.inner.cond=1.001,10,100,1000,10000,100000,1000000,10000000 loss.inner.dim=1000 loss.outer.dim=2000 solver.inner_backward.fac_increase=0 solver.inner_forward.lr=0.9 solver.inner_backward.lr=0.9 solver.inner_forward.n_iter=1,10,100,1000 solver.outer.lr=1. solver.inner_backward.n_iter=1,10,100,1000 
python launch_jobs.py --multirun solver=stocBiO solver.outer.epochs=50 logs.log_name=vary_cond_number_iter_high_dim_2 loss.outer.cond=10  loss.inner.cond=1.001,10,100,1000,10000,100000,1000000,10000000 loss.inner.dim=1000 loss.outer.dim=2000 solver.inner_backward.fac_increase=0 solver.inner_forward.lr=0.9 solver.inner_backward.lr=0.9 solver.inner_forward.n_iter=1,10,100,1000 solver.outer.lr=1. solver.inner_backward.n_iter=1,10,100,1000 





## without increasing inner iters
python launch_jobs.py --multirun solver=reverse solver.outer.epochs=50 logs.log_name=vary_cond_number_iter_high_dim_2  loss.outer.cond=10 loss.inner.cond=1.001,10,100,1000,10000,100000,1000000,10000000 loss.inner.dim=1000 loss.outer.dim=2000 solver.inner_forward.fac_increase=1,10,100,1000 solver.inner_forward.lr=0.9 solver.inner_backward.lr=0.9 solver.outer.lr=1. 
python launch_jobs.py --multirun solver=ITD solver.outer.epochs=50 logs.log_name=vary_cond_number_iter_high_dim_2  loss.outer.cond=10 loss.inner.cond=1.001,10,100,1000,10000,100000,1000000,10000000 loss.inner.dim=1000 loss.outer.dim=2000 solver.inner_forward.fac_increase=1,10,100,1000 solver.inner_forward.lr=0.9 solver.inner_backward.lr=0.9 solver.outer.lr=1. 
python launch_jobs.py --multirun solver=reverse solver.outer.epochs=50 logs.log_name=vary_cond_number_iter_high_dim_2 loss.outer.cond=10 loss.inner.cond=1.001,10,100,1000,10000,100000,1000000,10000000 loss.inner.dim=1000 loss.outer.dim=2000 solver.inner_forward.fac_increase=0 solver.inner_forward.lr=0.9 solver.inner_backward.lr=0.9 solver.inner_forward.n_iter=1,10,100, 1000 solver.outer.lr=1. 
python launch_jobs.py --multirun solver=ITD solver.outer.epochs=50 logs.log_name=vary_cond_number_iter_high_dim_2 loss.outer.cond=10 loss.inner.cond=1.001,10,100,1000,10000,100000,1000000,10000000 loss.inner.dim=1000 loss.outer.dim=2000 solver.inner_forward.fac_increase=0 solver.inner_forward.lr=0.9 solver.inner_backward.lr=0.9 solver.inner_forward.n_iter=1,10,100, 1000 solver.outer.lr=1. 
