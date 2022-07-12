#!/bin/bash


log_name=multitask_cifar100_new
dtype=32
training=multitask_cifar100
trainer_name='trainers.multitask.trainer_multitask_new_interface.Trainer'
resume=True

gpumem=14
cluster_name=''
device=-1
hours=17
launcher='besteffort'
app='/scratch/clear/marbel/anaconda3/bin/python'

# correction=True
# python launch_jobs.py --multirun   \
# 		system.dtype=$dtype\
# 		system.device=$device\
# 		launcher=$launcher\
# 		launcher.app=$app\
# 		launcher.gpumem=$gpumem\
# 		launcher.hours=$hours\
# 		cluster.name=$cluster_name\
# 		logs.log_name=$log_name\
# 		training=$training \
# 		training.resume=$resume\
# 		training.trainer_name=$trainer_name\
# 		training.lower.selection.correction=$correction\
# 		training.lower.selection.linear_solver.residual_op.name='core.selection.FiniteDiffResidual','core.selection.Residual','core.selection.NormalResidual'\


correction=True
python launch_jobs.py --multirun   \
		system.dtype=$dtype\
		system.device=$device\
		launcher=$launcher\
		launcher.app=$app\
		launcher.gpumem=$gpumem\
		launcher.hours=$hours\
		cluster.name=$cluster_name\
		logs.log_name=$log_name\
		training=$training \
		training.resume=$resume\
		training.trainer_name=$trainer_name\
		training.lower.selection.correction=$correction\
		training.lower.selection.linear_solver.residual_op.name='core.selection.NormalResidual'\
		training.lower.selection.optimizer.lr=0.01,0.001


correction=True
clip=True
max_norm=1. 
python launch_jobs.py --multirun   \
		system.dtype=$dtype\
		system.device=$device\
		launcher=$launcher\
		launcher.app=$app\
		launcher.gpumem=$gpumem\
		launcher.hours=$hours\
		cluster.name=$cluster_name\
		logs.log_name=$log_name\
		training=$training \
		training.resume=$resume\
		training.trainer_name=$trainer_name\
		training.lower.selection.correction=$correction\
		training.lower.selection.linear_solver.residual_op.name='core.selection.NormalResidual'\
		training.lower.selection.optimizer.lr=0.01,0.001\
		training.upper.clip=$clip\



correction=False
warm_start_iter=0
unrolled_iter=1

# python launch_jobs.py --multirun   \
# 		system.dtype=$dtype\
# 		system.device=$device\
# 		launcher=$launcher\
# 		launcher.app=$app\
# 		launcher.gpumem=$gpumem\
# 		launcher.hours=$hours\
# 		cluster.name=$cluster_name\
# 		logs.log_name=$log_name\
# 		training=$training \
# 		training.resume=$resume\
# 		training.trainer_name=$trainer_name\
# 		training.lower.selection.correction=$correction\
# 		training.lower.selection.warm_start_iter=$warm_start_iter\
# 		training.lower.selection.unrolled_iter=$unrolled_iter\


# correction=True
# python launch_jobs.py --multirun   \
# 		system.dtype=$dtype\
# 		system.device=$device\
# 		launcher=$launcher\
# 		launcher.app=$app\
# 		launcher.gpumem=$gpumem\
# 		launcher.hours=$hours\
# 		cluster.name=$cluster_name\
# 		logs.log_name=$log_name\
# 		training=$training \
# 		training.resume=$resume\
# 		training.trainer_name=$trainer_name\
# 		training.lower.selection.correction=$correction\
# 		training.lower.selection.linear_solver.residual_op.name='core.selection.NormalResidual'\
# 		training.lower.selection.warm_start_iter=$warm_start_iter\
# 		training.lower.selection.unrolled_iter=$unrolled_iter\
	

