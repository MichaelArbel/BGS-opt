python -m ipdb run.py --multirun solver=asig,stocBiO,AID-CG


python  launch_jobs.py --multirun solver=asig,stocBiO,AID-CG



HYDRA_FULL_ERROR=1   OC_CAUSE=1 python -m ipdb launch_jobs.py --multirun solver=AID-FP,BSA,ITD,reverse,TTSA,asig,stocBiO,AID-CG solver.outer.epochs=10 logs.log_name=test_log solver.outer.epochs=50


HYDRA_FULL_ERROR=1   OC_CAUSE=1 python -m ipdb launch_jobs.py --multirun solver=AID-FP,BSA,ITD,reverse,TTSA,ASIG,ASIG-CG,stocBiO,AID-CG solver.outer.epochs=10 logs.log_name=linear_high_dim solver.outer.epochs=50 model.inner.dim=300 model.outer.dim=100 solver.outer.lr=0.001 solver.inner_forward.lr=0.001 solver.inner_backward.lr=0.001 launcher=single_gpu





python launch_jobs.py --multirun solver=AID-FP,BSA,TTSA,stocBiO solver.outer.epochs=50 logs.log_name=ablation_vary_steps solver.outer.epochs=100









#### Conditioning number

python launch_jobs.py --multirun solver=AID-FP,AID-CG,stocBiO,ITD,reverse,ASIG,ASIG-CG solver.outer.epochs=100 logs.log_name=vary_cond_number solver.outer.epochs=100 loss.inner.cond=1.001,2,5,10,20,50,100,200,500,1000 solver. solver.inner_forward.lr=0.9 solver.inner_backward.lr=0.9





#### Conditioning number and num_iter: log_name= vary_cond_number_iter  (small dim)

python launch_jobs.py --multirun solver=AID-FP,AID-CG,stocBiO solver.outer.epochs=100 logs.log_name=vary_cond_number_iter  loss.inner.cond=1.001,2,5,10,20,50,100,200,500,1000 ++solver.inner_backward.fac_increase=0,1,10,100,1000 solver.inner_forward.lr=0.9 solver.inner_backward.lr=0.9
python launch_jobs.py --multirun solver=ITD,reverse solver.outer.epochs=100 logs.log_name=vary_cond_number_iter  loss.inner.cond=1.001,2,5,10,20,50,100,200,500,1000 ++solver.inner_forward.fac_increase=0,1,10,100,1000 solver.inner_forward.lr=0.9 solver.inner_backward.lr=0.9
python launch_jobs.py --multirun solver=ASIG,ASIG-CG solver.outer.epochs=100 logs.log_name=vary_cond_number_iter  loss.inner.cond=1.001,2,5,10,20,50,100,200,500,1000  solver.inner_forward.lr=0.9 solver.inner_backward.lr=0.9

## without increasing inner iters
python launch_jobs.py --multirun solver=AID-FP,AID-CG,stocBiO solver.outer.epochs=100 logs.log_name=vary_cond_number_iter  loss.inner.cond=1.001,2,5,10,20,50,100,200,500,1000 ++solver.inner_backward.fac_increase=0 solver.inner_forward.lr=0.9 solver.inner_backward.lr=0.9 solver.inner_forward.n_iter=1,5,20,50 solver.inner_backward.n_iter=1,5,20,50
python launch_jobs.py --multirun solver=ITD,reverse solver.outer.epochs=100 logs.log_name=vary_cond_number_iter  loss.inner.cond=1.001,2,5,10,20,50,100,200,500,1000 ++solver.inner_forward.fac_increase=0 solver.inner_forward.lr=0.9 solver.inner_backward.lr=0.9 solver.inner_forward.n_iter=1,5,20,50




### Tests 
HYDRA_FULL_ERROR=1   OC_CAUSE=1 python -m ipdb run.py --multirun solver=AID-FP solver.outer.epochs=100 logs.log_name=debug  loss.inner.cond=1.001 ++solver.inner_backward.fac_increase=3. solver.inner_forward.lr=0.9 solver.inner_backward.lr=0.9 solver.inner_forward.n_iter=1 solver.inner_backward.n_iter=1



############ Higher dim


python launch_jobs.py --multirun solver=AID-FP solver.outer.epochs=50 logs.log_name=vary_cond_number_iter_high_dim_2  loss.outer.cond=10 loss.inner.cond=1.001,10,100,1000,10000,100000,1000000,10000000 loss.inner.dim=1000 loss.outer.dim=2000 solver.inner_backward.fac_increase=1,10,100,1000 solver.inner_forward.lr=0.9 solver.inner_backward.lr=0.9 solver.outer.lr=1. 
python launch_jobs.py --multirun solver=stocBiO solver.outer.epochs=50 logs.log_name=vary_cond_number_iter_high_dim_2  loss.outer.cond=10 loss.inner.cond=1.001,10,100,1000,10000,100000,1000000,10000000 loss.inner.dim=1000 loss.outer.dim=2000 solver.inner_backward.fac_increase=1,10,100,1000 solver.inner_forward.lr=0.9 solver.inner_backward.lr=0.9 solver.outer.lr=1. 
python launch_jobs.py --multirun solver=AID-CG solver.outer.epochs=50 logs.log_name=vary_cond_number_iter_high_dim_2  loss.outer.cond=10 loss.inner.cond=1.001,10,100,1000,10000,100000,1000000,10000000 loss.inner.dim=1000 loss.outer.dim=2000 solver.inner_backward.fac_increase=1,10,100,1000 solver.inner_forward.lr=0.9 solver.inner_backward.lr=0.9 solver.outer.lr=1. 




python launch_jobs.py --multirun solver=AID-FP solver.outer.epochs=50 logs.log_name=vary_cond_number_iter_high_dim_2 loss.outer.cond=10  loss.inner.cond=1.001,10,100,1000,10000,100000,1000000,10000000 loss.inner.dim=1000 loss.outer.dim=2000 solver.inner_backward.fac_increase=0 solver.inner_forward.lr=0.9 solver.inner_backward.lr=0.9 solver.inner_forward.n_iter=100,1000 solver.outer.lr=1. solver.inner_backward.n_iter=1,10,100,1000 
python launch_jobs.py --multirun solver=AID-CG solver.outer.epochs=50 logs.log_name=vary_cond_number_iter_high_dim_2 loss.outer.cond=10  loss.inner.cond=1.001,10,100,1000,10000,100000,1000000,10000000 loss.inner.dim=1000 loss.outer.dim=2000 solver.inner_backward.fac_increase=0 solver.inner_forward.lr=0.9 solver.inner_backward.lr=0.9 solver.inner_forward.n_iter=100,1000 solver.outer.lr=1. solver.inner_backward.n_iter=1,10,100,1000 
python launch_jobs.py --multirun solver=stocBiO solver.outer.epochs=50 logs.log_name=vary_cond_number_iter_high_dim_2 loss.outer.cond=10  loss.inner.cond=1.001,10,100,1000,10000,100000,1000000,10000000 loss.inner.dim=1000 loss.outer.dim=2000 solver.inner_backward.fac_increase=0 solver.inner_forward.lr=0.9 solver.inner_backward.lr=0.9 solver.inner_forward.n_iter=100,1000 solver.outer.lr=1. solver.inner_backward.n_iter=1,10,100,1000 



python launch_jobs.py --multirun solver=ASIG solver.outer.epochs=50 logs.log_name=vary_cond_number_iter_high_dim_2  loss.outer.cond=10 loss.inner.cond=1.001,10,100,1000,10000,100000,1000000,10000000  loss.inner.dim=1000 loss.outer.dim=2000 solver.inner_forward.lr=0.9 solver.inner_backward.lr=0.9 solver.inner_backward.n_iter=1,10,100,1000 solver.inner_forward.n_iter=10,100,1000 solver.outer.lr=1. 
python launch_jobs.py --multirun solver=ASIG-CG solver.outer.epochs=50 logs.log_name=vary_cond_number_iter_high_dim_2  loss.outer.cond=10 loss.inner.cond=1.001,10,100,1000,10000,100000,1000000,10000000  loss.inner.dim=1000 loss.outer.dim=2000 solver.inner_forward.lr=0.9 solver.inner_backward.lr=0.9 solver.inner_backward.n_iter=1,10,100,1000 solver.inner_forward.n_iter=10,100,1000 solver.outer.lr=1. 



python launch_jobs.py --multirun solver=ASIG solver.outer.epochs=50 logs.log_name=vary_cond_number_iter_high_dim_2  loss.outer.cond=10 loss.inner.cond=1.001,10,100,1000,10000,100000,1000000,10000000  loss.inner.dim=1000 loss.outer.dim=2000 solver.inner_forward.lr=0.9 solver.inner_backward.lr=0.9 solver.inner_backward.n_iter=1,10,100,1000 solver.outer.lr=1. 
python launch_jobs.py --multirun solver=ASIG-CG solver.outer.epochs=50 logs.log_name=vary_cond_number_iter_high_dim_2  loss.outer.cond=10 loss.inner.cond=1.001,10,100,1000,10000,100000,1000000,10000000  loss.inner.dim=1000 loss.outer.dim=2000 solver.inner_forward.lr=0.9 solver.inner_backward.lr=0.9 solver.inner_backward.n_iter=1,10,100,1000 solver.outer.lr=1. 


## without increasing inner iters
python launch_jobs.py --multirun solver=reverse solver.outer.epochs=50 logs.log_name=vary_cond_number_iter_high_dim_2  loss.outer.cond=10 loss.inner.cond=1.001,10,100,1000,10000,100000,1000000,10000000 loss.inner.dim=1000 loss.outer.dim=2000 solver.inner_forward.fac_increase=1,10,100,1000 solver.inner_forward.lr=0.9 solver.inner_backward.lr=0.9 solver.outer.lr=1. 
python launch_jobs.py --multirun solver=ITD solver.outer.epochs=50 logs.log_name=vary_cond_number_iter_high_dim_2  loss.outer.cond=10 loss.inner.cond=1.001,10,100,1000,10000,100000,1000000,10000000 loss.inner.dim=1000 loss.outer.dim=2000 solver.inner_forward.fac_increase=1,10,100,1000 solver.inner_forward.lr=0.9 solver.inner_backward.lr=0.9 solver.outer.lr=1. 
python launch_jobs.py --multirun solver=reverse solver.outer.epochs=50 logs.log_name=vary_cond_number_iter_high_dim_2 loss.outer.cond=10 loss.inner.cond=1.001,10,100,1000,10000,100000,1000000,10000000 loss.inner.dim=1000 loss.outer.dim=2000 solver.inner_forward.fac_increase=0 solver.inner_forward.lr=0.9 solver.inner_backward.lr=0.9 solver.inner_forward.n_iter=1,10,100,1000 solver.outer.lr=1. 
python launch_jobs.py --multirun solver=ITD solver.outer.epochs=50 logs.log_name=vary_cond_number_iter_high_dim_2 loss.outer.cond=10 loss.inner.cond=1.001,10,100,1000,10000,100000,1000000,10000000 loss.inner.dim=1000 loss.outer.dim=2000 solver.inner_forward.fac_increase=0 solver.inner_forward.lr=0.9 solver.inner_backward.lr=0.9 solver.inner_forward.n_iter=1,10,100,1000 solver.outer.lr=1. 



python launch_jobs.py --multirun solver=AID-GD solver.outer.epochs=50 logs.log_name=vary_cond_number_iter_high_dim_2  loss.outer.cond=10 loss.inner.cond=1.001,10,100,1000,10000,100000,1000000,10000000 loss.inner.dim=1000 loss.outer.dim=2000 solver.inner_backward.fac_increase=1,10,100,1000 solver.inner_forward.lr=0.9 solver.inner_backward.lr=0.9 solver.outer.lr=1. 
python launch_jobs.py --multirun solver=AID-GD solver.outer.epochs=50 logs.log_name=vary_cond_number_iter_high_dim_2 loss.outer.cond=10  loss.inner.cond=1.001,10,100,1000,10000,100000,1000000,10000000 loss.inner.dim=1000 loss.outer.dim=2000 solver.inner_backward.fac_increase=0 solver.inner_forward.lr=0.9 solver.inner_backward.lr=0.9 solver.inner_forward.n_iter=1,10,20 solver.outer.lr=1. solver.inner_backward.n_iter=1,10,100,1000 



#### Total jobs:
#####     8*(3*4+2*4+2*4+2*3+3*3*3)


##### 8*(4 + 9)
##### 8*(4+3)









###### Tests

HYDRA_FULL_ERROR=1   OC_CAUSE=1 python -m ipdb run.py --multirun solver=ASIG-CG solver.outer.epochs=100 logs.log_name=debug  loss.inner.cond=10000000 loss.outer.cond=10 solver.inner_forward.lr=0.9 solver.inner_backward.lr=0.9 solver.outer.lr=1. loss.inner.dim=1000 loss.outer.dim=2000 solver.inner_backward.n_iter=10 solver.inner_forward.n_iter=10 system.device=-1



HYDRA_FULL_ERROR=1   OC_CAUSE=1 python -m ipdb run.py --multirun solver=AID-CG solver.outer.epochs=100 logs.log_name=debug  loss.inner.cond=10000000 loss.outer.cond=10 solver.inner_forward.lr=0.9 solver.inner_backward.lr=0.9 solver.outer.lr=1. loss.inner.dim=1000 loss.outer.dim=2000 solver.inner_backward.n_iter=1000 solver.inner_backward.increase_n_iter=true solver.inner_backward.fac_increase=0 system.device=-1



HYDRA_FULL_ERROR=1   OC_CAUSE=1 python -m ipdb run.py --multirun solver=stocBiO solver.outer.epochs=100 logs.log_name=debug  loss.inner.cond=1.001 loss.outer.cond=10 solver.inner_forward.lr=0.9 solver.inner_backward.lr=0.9 solver.outer.lr=1. loss.inner.dim=1000 loss.outer.dim=2000 solver.inner_forward.n_iter=10 solver.inner_backward.n_iter=10 solver.inner_backward.increase_n_iter=false solver.inner_backward.fac_increase=0


############################### Kernel regression







### Test kernel ridge

HYDRA_FULL_ERROR=1   OC_CAUSE=1 python -m ipdb run.py --multirun solver=ASIG loss=kernel_ridge data=20newsgroup logs.log_name=debug solver.inner_forward.lr=100. solver.inner_backward.lr=10. solver.outer.lr=100. system.dtype=32 data.b_size=100  solver.inner_forward.n_iter=10 solver.inner_backward.n_iter=10 solver.outer.epochs=100 solver.outer.decrease_lr=false 


HYDRA_FULL_ERROR=1   OC_CAUSE=1 python -m ipdb run.py --multirun solver=AID-FP,AID-CG,stocBiO,BSA loss=kernel_ridge data=20newsgroup logs.log_name=debug solver.inner_forward.lr=100. solver.inner_backward.lr=10. solver.outer.lr=100. system.dtype=32 data.b_size=100  solver.inner_forward.n_iter=1 solver.inner_backward.n_iter=1 ++solver.inner_backward.fac_increase=1 solver.outer.epochs=100 




HYDRA_FULL_ERROR=1   OC_CAUSE=1 python -m ipdb run.py --multirun solver=ITD,reverse loss=kernel_ridge data=20newsgroup logs.log_name=debug solver.inner_forward.lr=100. solver.inner_backward.lr=10. solver.outer.lr=100. system.dtype=32 data.b_size=10  solver.inner_forward.n_iter=1 solver.outer.epochs=100 ++solver.inner_forward.fac_increase=1

HYDRA_FULL_ERROR=1   OC_CAUSE=1 python -m ipdb run.py --multirun solver=AID-FP,AID-CG,stocBiO,BSA loss=kernel_ridge data=20newsgroup logs.log_name=debug solver.inner_forward.lr=100. solver.inner_backward.lr=10. solver.outer.lr=100. system.dtype=32 data.b_size=100  solver.inner_forward.n_iter=1 solver.inner_backward.n_iter=1 solver.outer.epochs=100 ++solver.inner_backward.fac_increase=0

HYDRA_FULL_ERROR=1   OC_CAUSE=1 python -m ipdb run.py --multirun solver=ITD,reverse loss=kernel_ridge data=20newsgroup logs.log_name=debug solver.inner_forward.lr=100. solver.inner_backward.lr=10. solver.outer.lr=100. system.dtype=32 data.b_size=100  solver.inner_forward.n_iter=1  solver.outer.epochs=100 ++solver.inner_forward.fac_increase=0




#launched
python  launch_jobs.py --multirun solver=AID-FP,AID-CG,stocBiO,BSA loss=kernel_ridge data=20newsgroup logs.log_name=20newsgroup_new solver.inner_forward.lr=100. solver.inner_backward.lr=10. solver.outer.lr=100. system.dtype=32 data.b_size=10,100,1000 metrics.disp_freq=10 solver.inner_forward.n_iter=1,5,10 solver.inner_backward.n_iter=1 solver.outer.epochs=100 ++solver.inner_backward.fac_increase=1,10,100 &
python  launch_jobs.py --multirun solver=ITD,reverse loss=kernel_ridge data=20newsgroup logs.log_name=20newsgroup_new solver.inner_forward.lr=100. solver.inner_backward.lr=10. solver.outer.lr=100. system.dtype=32 data.b_size=10,100,1000 metrics.disp_freq=10 solver.inner_forward.n_iter=1 solver.outer.epochs=100 ++solver.inner_forward.fac_increase=1,10,100 &



python  launch_jobs.py --multirun solver=AID-FP,AID-CG,stocBiO,BSA loss=kernel_ridge data=20newsgroup logs.log_name=20newsgroup_new solver.inner_forward.lr=100. solver.inner_backward.lr=10. solver.outer.lr=100. system.dtype=32 data.b_size=10,100,1000 metrics.disp_freq=10 solver.inner_forward.n_iter=1,5,10 solver.inner_backward.n_iter=1,5,10 solver.outer.epochs=100 ++solver.inner_backward.fac_increase=0 &
python  launch_jobs.py --multirun solver=ITD,reverse loss=kernel_ridge data=20newsgroup logs.log_name=20newsgroup_new solver.inner_forward.lr=100. solver.inner_backward.lr=10. solver.outer.lr=100. system.dtype=32 data.b_size=10,100,1000 metrics.disp_freq=10 solver.inner_forward.n_iter=1,5,10  solver.outer.epochs=100 ++solver.inner_forward.fac_increase=0 &

python  launch_jobs.py --multirun solver=ASIG loss=kernel_ridge data=20newsgroup logs.log_name=20newsgroup_new solver.inner_forward.lr=100. solver.inner_backward.lr=10. solver.outer.lr=100. system.dtype=32 data.b_size=10,100,1000 metrics.disp_freq=10 solver.inner_forward.n_iter=1,5,10 solver.inner_backward.n_iter=1,5,10 solver.outer.epochs=100 solver.outer.decrease_lr=false & 
python  launch_jobs.py --multirun solver=ASIG loss=kernel_ridge data=20newsgroup logs.log_name=20newsgroup_new solver.inner_forward.lr=100. solver.inner_backward.lr=10. solver.outer.lr=100. system.dtype=32 data.b_size=10,100,1000 metrics.disp_freq=10 solver.inner_forward.n_iter=1,5,10 solver.inner_backward.n_iter=1,5,10 solver.outer.epochs=100 solver.outer.decrease_lr=true & 



#### Test kernel





##########################   
#### Meta Learning
HYDRA_FULL_ERROR=1   OC_CAUSE=1 python -m ipdb run.py --multirun solver=MetaAsig loss=meta_loss_miniimagenet data=MiniImagenet logs.log_name=debug  solver.inner_forward.lr=.05 solver.inner_backward.lr=.05 solver.outer.lr=.02 system.dtype=32 metrics.disp_freq=32 solver.inner_forward.n_iter=20  solver.outer.epochs=1000 system.device=-1 data.num_tasks=20000 solver.inner_backward.n_iter=10

HYDRA_FULL_ERROR=1   OC_CAUSE=1 python -m ipdb run.py --multirun solver=MetaITD loss=meta_loss_miniimagenet data=MiniImagenet logs.log_name=debug  solver.inner_forward.lr=.05 solver.inner_backward.lr=.05 solver.outer.lr=.02 system.dtype=32 metrics.disp_freq=32 solver.inner_forward.n_iter=20  solver.outer.epochs=1000 system.device=-1 data.num_tasks=20000 solver.inner_forward.warm_start=false





HYDRA_FULL_ERROR=1   OC_CAUSE=1 python -m ipdb run.py --multirun solver=MetaITD loss=meta_loss_miniimagenet data=MiniImagenet logs.log_name=debug  solver.inner_forward.lr=.1 solver.inner_backward.lr=.1 solver.outer.lr=.002 system.dtype=32 metrics.disp_freq=32 solver.inner_forward.n_iter=5  solver.outer.epochs=1000 system.device=-1 data.num_tasks=20000 solver.inner_forward.warm_start=false




#########################################
##### Tests



HYDRA_FULL_ERROR=1   OC_CAUSE=1 python -m ipdb run.py --multirun solver=ASIG solver.outer.epochs=100 logs.log_name=debug  loss.inner.cond=1.001  solver.inner_forward.lr=0.9 solver.inner_backward.lr=0.9
















### deleting jobs 

seq 1 n | xargs -n 5 echo
