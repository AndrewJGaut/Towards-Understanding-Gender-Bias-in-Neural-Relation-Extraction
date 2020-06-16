#!/bin/bash

# enumerate all the debiasing optoins
# this is so we can call each of these values individually to get all the debiasgin optoins run and trained
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

for i in {8..15}; #get all debiasing opitions
do
	nohup python3 -u train_debiasingoptions.py ${deboptions[($i)]} & 
	wait
done
