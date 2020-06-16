from __future__ import unicode_literals
import os
import pdb
import logging
import sklearn
from web.datasets.analogy import fetch_google_analogy
from web.embeddings import *
from web.embedding import * 
from web.vocabulary import *
import scipy

#%matplotlib inline
import json



logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')


import debiaswe as dwe
import debiaswe.we as we
from debiaswe.we import WordEmbedding
from debiaswe.data import load_professions
from debiaswe.debias import debias
from Utility import *

#E = WordEmbedding('./embeddings/w2v_gnews_small.txt')

def debiasEmbedding(filename, bootstrapped=False):
    outfile_path = './embeddings/' + 'debiased_' + filename
    if(bootstrapped or not os.path.exists(outfile_path)):
        E = WordEmbedding('./embeddings/' + filename)


        with open('./data/definitional_pairs.json', "r") as f:
            defs = json.load(f)
        #print("definitional", defs)

        with open('./data/equalize_pairs.json', "r") as f:
            equalize_pairs = json.load(f)

        with open('./data/gender_specific_seed.json', "r") as f:
            gender_specific_words = json.load(f)
        #print("gender specific", len(gender_specific_words), gender_specific_words[:10])

        debias(E, gender_specific_words, defs, equalize_pairs)

        E.save(outfile_path)

        #return E.model

def debiasEmbeddings():
    true_false_combos = getTrueFalseCombos(3)
    for combo in true_false_combos:
        suffix =  getNameSuffixSimulateArgs(equalized_gender_mentions=combo[0],
                                                                name_anonymized=combo[1],
                                                                gender_swapped=combo[2], swap_names=False)
        wordvec_file_name = getWordEmbeddingFileName(suffix, '.txt')

        try:
            debiasEmbedding(wordvec_file_name)
        except Exception as e:
            print(e)
            continue


def debiasEmbeddingsWithArgs():
    '''
    The same as debiasEmbeddings but it takes arguments and only debiases the embedding that fits those arguments.
    :return:
    '''
    args = getCommandLineArgs()

    args.debiased_embeddings = False

    suffix = getNameSuffix(args)

    wordvec_file_name = getWordEmbeddingFileName(suffix, '.txt')

    try:
        debiasEmbedding(wordvec_file_name, args.bootstrapped)
    except Exception as e:
        print(e)

if __name__ == '__main__':
    #debiasEmbeddings()
    #os.chdir('./WordEmbeddings')	
    debiasEmbeddingsWithArgs()
    #os.chdir('../')
