#!/bin/bash

# enumerate all the debiasing options os we can run them all
deboptions[0]='-na -egm -gs'
deboptions[1]='-na -egm'
deboptions[2]='-egm -gs'
deboptions[3]='-egm'
deboptions[4]='-na -egm -gs -de'
deboptions[5]='-na -egm -de'
deboptions[6]='-egm -gs -de'
deboptions[7]='-egm -de'
deboptions[8]='-na -gs'
deboptions[9]='-na'
deboptions[10]='-gs'
deboptions[11]='-na -gs -de'
deboptions[12]='-na -de'
deboptions[13]='-gs -de'
deboptions[14]=''
deboptions[15]='-de'

test_options[0]='--female_test_files'
test_options[1]='--male_test_files'

for i in {8..15}; #get all debiasing opitions
do
	for j in {0..1}; #get all testing options
	do
		CUDA_VISIBLE_DEVICES=3 nohup python3 -u test_debiasingoptions.py ${deboptions[($i)]} ${test_options[($j)]} & 
		wait
	done
done
