#!/bin/bash


log_name=debug
dtype=32
loss=multitask_cifar100
training=multitask_cifar100
loader=multitask_cifar100
metrics=multitask_cifar100
method=AmIGO
device=-1










HYDRA_FULL_ERROR=1   OC_CAUSE=1  python -m ipdb run.py --multirun method=$method  \
		loss=$loss \
		loader=$loader \
		training=$training \
		metrics=$metrics\
		system.dtype=$dtype \
		system.device=$device\
		logs.log_name=$log_name \





