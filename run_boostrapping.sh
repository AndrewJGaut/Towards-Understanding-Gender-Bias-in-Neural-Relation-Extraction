#!/bin/bash

# enumerate all the debiasing optoins
# this is so we can call each of these values individually to get all the debiasgin optoins run and trained
deboptions[1]='-na'
deboptions[2]='-egm'
deboptions[3]='-na -egm'
deboptions[5]='-na -egm -de'
deboptions[6]='-egm -de'
deboptions[7]='-na -de'
deboptions[0]=''
deboption4[4]='-de'
deboptions[8]='-na -gs'
deboptions[9]='-gs'
deboptions[10]='-na -egm -gs -de'
deboptions[11]='-na -egm -gs'
deboptions[12]='-egm -gs'
deboptions[13]='-egm -gs -de'
deboptions[14]='-na -gs -de'
deboptions[15]='-gs -de'
deboptions[16]='-gs -nt'
deboptions[17]='-na -gs -nt'

gpu=3




for i in {0..15}; #get all debiasing opitions
do
	# first, create the dataset
	CUDA_VISIBLE_DEVICES=$gpu python3 -u WikigenderJsonParsing/DebiasJsonDataset.py ${deboptions[($i)]} -bs & 
	wait
	echo "done creating dataset"
	#for j in {0..10}; 
	for j in {0..10};
	do
		echo $j
		echo ${deboptions[($i)]} 	
		rm Models/OpenNRE/_processed_data/* &
		rm Models/OpenNRE/checkpoint/*
		echo "done deleting problematic preprocessed data"
		
		CUDA_VISIBLE_DEVICES=$gpu python3 -u WikigenderJsonParsing/CreateBootstrappedDatasets.py ${deboptions[($i)]} -bs & 
		wait
		echo "done creating bootstrapped dataset"
		CUDA_VISIBLE_DEVICES=$gpu python3 -u WikigenderJsonParsing/ConvertWikigenderToOpenNREFormat.py ${deboptions[($i)]} -bs & 
		wait
		echo "done convertin to opennre format"

		CUDA_VISIBLE_DEVICES=$gpu python3 -u WordEmbeddings/Word2VecTraining.py ${deboptions[($i)]}  -bs &
		wait
		echo "done creating and debiasing embeddings"
		CUDA_VISIBLE_DEVICES=$gpu python3 -u Models/OpenNRE/train_debiasingoptions.py ${deboptions[($i)]} -bs &
		wait
		echo "done training model" 
		CUDA_VISIBLE_DEVICES=$gpu python3 -u Models/OpenNRE/test_debiasingoptions.py ${deboptions[($i)]} -bs -bs_num=$j &
		wait
		echo "done testing model on female data" 
		CUDA_VISIBLE_DEVICES=$gpu python3 -u Models/OpenNRE/test_debiasingoptions.py ${deboptions[($i)]} --male_test_files -bs -bs_num=$j &
		wait
		echo "done testing model on male data" 

		rm Models/OpenNRE/_processed_data/* &
		rm Models/OpenNRE/checkpoint/*
		echo "done deleting problematic preprocessed data" #if we kept soeme prepr data, then oprennre would never prepcorcess the new training data and word embeddigns (Since the proprocessed file for these would have already been created since each bootstrapped smaple has the same file names) 
 		wait
	done
	#nohup python3 -u train_debiasingoptions.py ${deboptions[($i)]} & 
	#wait
done
