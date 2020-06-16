#!/bin/bash

# enumerate all the debiasing optoins
# this is so we can call each of these values individually to get all the debiasgin optoins run and trained
encoders[0]='pcnn'
encoders[1]='cnn'
encoders[2]='bert'

selectors[0]='att'
selectors[1]='ave'

equalized[0]=''
equalized[1]='-egm'

num_bootstrap_samples=10
gpu=0,1

for i in {0..1}; #get all debiasing opitions
do
	for k in {0..1};
	do
		for m in {0..1};
		do
			for j in {0..4};
			do

				echo $j

				rm Models/OpenNRE/_processed_data/* &
				wait
				rm Models/OpenNRE/_processed_data/checkpoint/* &
				echo "done deleting problematic preprocessed data" #if we kept soeme prepr data, then oprennre would never prepcorcess the new training data and word embeddigns (Since the proprocessed file for these would have already been created since each bootstrapped smaple has the same file names) 
				CUDA_VISIBLE_DEVICES=$gpu python3 -u WikigenderJsonParsing/CreateBootstrappedDatasets.py ${equalized[($m)]} -bs & 
				wait
				echo "done creating dataset"
				CUDA_VISIBLE_DEVICES=$gpu python3 -u WikigenderJsonParsing/ConvertWikigenderToOpenNREFormat.py ${equalized[($m)]} -bs & 
				wait
				echo "done convertin to opennre format"

				CUDA_VISIBLE_DEVICES=$gpu  python3 -u WordEmbeddings/Word2VecTraining.py ${equalized[($m)]} -bs &
				wait
				echo "done creating and debiasing embeddings"
				CUDA_VISIBLE_DEVICES=$gpu python3 -u Models/OpenNRE/train_debiasingoptions.py --encoder=${encoders[($i)]} --selector=${selectors[($k)]} ${equalized[($m)]} -bs --data_parallel &
				wait
				echo "done training model" 
				CUDA_VISIBLE_DEVICES=$gpu python3 -u Models/OpenNRE/test_debiasingoptions.py --encoder=${encoders[($i)]} --selector=${selectors[($k)]} ${equalized[($m)]} -bs -bs_num=$j --data_parallel &
				wait
				echo "done testing model on female data" 
			        CUDA_VISIBLE_DEVICES=$gpu python3 -u Models/OpenNRE/test_debiasingoptions.py  --encoder=${encoders[($i)]} --selector=${selectors[($k)]} --male_test_files ${equalized[($m)]} -bs -bs_num=$j --data_parallel &
				wait
				echo "done testing model on male data" 

				rm Models/OpenNRE/_processed_data/*train*.* &
				wait
				rm Models/OpenNRE/_processed_data/*word_vec*.* &
				echo "done deleting problematic preprocessed data" #if we kept soeme prepr data, then oprennre would never prepcorcess the new training data and word embeddigns (Since the proprocessed file for these would have already been created since each bootstrapped smaple has the same file names) 
			done
		done
		#nohup python3 -u train_debiasingoptions.py ${deboptions[($i)]} & 
		#wait

	done
done