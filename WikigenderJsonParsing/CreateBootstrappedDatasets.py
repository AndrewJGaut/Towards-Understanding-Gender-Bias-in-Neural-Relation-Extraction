from Utility import *
import argparse
import os
import random
#from sklearn.utils import resample

WORD_EMBEDDING_DIRECTORY = '../WordEmbeddings/'

BOOTSTRAP_FRACTION = 0.9


def createBootstrappedDataset(dataset_name, args):
    '''
    :param dataset_name: the name of the original dataset (the dataset without debiasing)
    :param equalized: boolean flag; if true, the data read in will have equalized mentions
    :param name_anonymized: boolean flag; if true, the data read in will be name anonymized
    :param gender_swapped: boolean flag; if true, the data read in will be gender-swapped
    :param swap_names:
    :return:
    '''
    # get the full name of the dataset!
    infile_names = dataset_name.split('.')
    old_bs = args.bootstrapped
    args.bootstrapped = False
    infile_names[0] += getNameSuffix(args)
    args.bootstrapped = old_bs

    infile_name = infile_names[0] + "." + infile_names[1]
    # read the data
    data = readFromJsonFile(infile_name)

    print('BOOSTRAPPED? {}'.format(args.bootstrapped))
    if args.bootstrapped:
        infile_names[0] += "_bootstrapped"

    #data = random.sample(data, bootstrap_percentage * len(data))
    #data['train'] = resample(data['train'], replace=True, n_samples=None)
    data['train'] = random.sample(data['train'], int(BOOTSTRAP_FRACTION * len(data['train'])))

    # write the bootstrapped dataset to a file
    outfile_name = infile_names[0] + '.' + infile_names[1]
    print('creating {}'.format(outfile_name))
    writeToJsonFile(data, outfile_name)
    writeToJsonFile(data, os.path.join(WORD_EMBEDDING_DIRECTORY, outfile_name)) # also write it to the word embeddings directory

    return data


if __name__ == '__main__':
    os.chdir('./WikigenderJsonParsing/') #this is for running a script in the directory above this
    args = getCommandLineArgs()
    createBootstrappedDataset('JsonData/Wikigender.json', args)
    os.chdir('../') # return to original directory
