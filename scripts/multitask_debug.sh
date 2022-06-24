#!/bin/bash


log_name=debug
dtype=32
loss=multitask_cifar100
training=multitask_cifar100
loader=multitask_cifar100
metrics=multitask_cifar100
selection=BGS
device=1
trainer_name='trainers.multitask.trainer_multitask_new_interface.Trainer'




HYDRA_FULL_ERROR=1   OC_CAUSE=1  python -m ipdb run.py --multirun selection=$selection  \
		loss=$loss \
		loader=$loader \
		training=$training \
		training.trainer_name=$trainer_name\
		metrics=$metrics\
		system.dtype=$dtype \
		system.device=$device\
		logs.log_name=$log_name \
		loss.model.lower.name='trainers.multitask.models.models.VectMTLVGG16'\





