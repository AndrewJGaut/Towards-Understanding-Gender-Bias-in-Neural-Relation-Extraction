# The source code for the paper titled "Towards Understanding Gender Bias in Neural Relation Extraction" by Tony Sun and Andrew Gaut et. al

This code contains several different modules used for the experimentation given in the paper

## General Usage

* Running full experiments
  * To run all experiments for varying encoder/selector, use ./run_bootstrapping_modelcombos.sh
  * To run all experiments varying debiasing method, use ./run_bootstrapping.sh

* All sub-modules use the same command line arguments. These are
  * -gs : indicates that you want to use counterfactual data augmentation
  * -egm : indicates you want to use balanced gender mentions
  * -de : indicates you want to use debiased word embeddings
  * -na : if you want to use name anonymization (see Zhao et al 2017[https://www.aclweb.org/anthology/N18-2003.pdf]); note that this implementation was buggy, so we did not report those results in the paper.  
  
  
* Submodules
  * WikigenderJsonParsing: contains code for debiasing the WikiGenderBias dataset and for generating data for OpenNRE experimentation.
  * WordEmbeddings: contains code for training word embeddings on our dataset and for debiasing embeddings.
  * Models: contains files for models we used for experimentation (note that we only made very minor edits to OpenNRE's model's evaluation method).
  
  
If you would like to conduct your own experiments, and thus can't use our pre-made bash scripts, we've included additional instructions below.
  
## WikigenderJsonParsing

To run the main code here, use `python3.6 DebiasJsonDataset.py [args]` (for example, `python3.6 DebiasJsonDataset.py -gs -egm` will generate a dataset with counterfactual data augmentation and equaliziation debiasing options applied to the dataset). This will create a dataset with the specified debiasing options in the `JsonData`.

To generate the data in the format for OpenNRE, simply run `python3.6 ConvertWikigenderToOpenNREFormat.py [args]` using the args specified when you created the dataset above. Noe that this will create the OpenNRE data in the folders `WikigenderJsonParsing/OpenNREData` and `Models/OpenNRE/data/Wikigender`. 

## Word Embeddings
Use `python3.6 Word2VecTraining.py [args]` to generate the word embeddings for your given dataset. You *must* use the same options as you used to create the original dataset (this is so that the word embeddings are trained on the correct dataset, since we train a different model for each debiasing option).


## Models/OpenNRE

Use `python3.6 train_debiasingoptions.py [args]` and `python3.6 test_debiasingoptions.py [args]` to train and test, respectively. Remember to use the same debiasing options you used above.

`test_debiasingoptions.py` can be run directly after training the model (OpenNRE will do the saving and loading). After running `test_debiasingoptions.py`, the results will be saved in the folder `test_result`.


## ModelResultParsing

Use this folder to parse results from the models.

First, place the results from the `test_result` folder into the `bootstrapped_test_results` folder and then run `python3.6 GetBootstrappedResults.py` and it will produce results for every model combination in the folder `bootstrapped_parsed_results`.




