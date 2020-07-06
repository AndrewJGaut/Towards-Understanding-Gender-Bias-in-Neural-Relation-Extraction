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


