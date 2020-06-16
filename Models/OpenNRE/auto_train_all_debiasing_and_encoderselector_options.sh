#!/bin/bash

models[0]='pcnn' 
models[1]='cnn'
models[2]='rnn'
models[3]='birnn'

selectors[0]='att'
selectors[1]='ave'
selectors[2]='cross_max'

deb_options[0] = '-gs'
deb_options[1] = '-egm'
deb_options[2] = '-na'
deb_options[3] = '-de'
deb_options[4] = '-sn'

test_options[0] = '-female'
test_options[1] = '-male'

for j in {0..2}; #skip models already done
do
	for i in {0..3}; #NOTE: we skip att because it was already run for all modles
	do
		#echo ${models[($i)]} ${selectors[($j)]}
		nohup python2.7 -u train_demo.py nyt ${models[($i)]} ${selectors[($j)]} > TRAINING_LOGS/log_${models[($i)]}_${selectors[($j)]}.txt &
		wait
	done
done
