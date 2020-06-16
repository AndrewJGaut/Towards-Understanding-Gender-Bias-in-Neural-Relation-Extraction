#!/bin/bash

models[0]='pcnn'
models[1]='cnn'
models[2]='rnn'
models[3]='birnn'

selectors[0]='att'
selectors[1]='ave'
selectors[2]='cross_max'

for j in {0..2}; #skip models already done
do
        for i in {0..3}; #NOTE: we skip att because it was already run for all modles
        do
                #echo ${models[($i)]} ${selectors[($j)]}
                CUDA_VISIBLE_DEVICES=3 nohup python3 -u test_debiasingoptions.py --encoder=${models[($i)]} --selector=${selectors[($j)]} --male_test_files &
                wait
        done
done
