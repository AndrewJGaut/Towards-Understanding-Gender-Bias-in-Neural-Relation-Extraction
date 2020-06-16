'''In this script, we want to find all the attention scores using the regex below'''

import re
import matplotlib.pyplot as plt
import numpy as np
import os
from Utility import readFromJsonFile, getModelNameSimulateArgs

NREDATA_PATH = '../WikigenderJsonParsing/OpenNREData/'
REG_DATA_PATH = os.path.join(NREDATA_PATH, 'train.json')
NA_DATA_PATH = os.path.join(NREDATA_PATH, 'train_NA_NoEq_NoGS_bootstrapped.json')
EGM_DATA_PATH = os.path.join(NREDATA_PATH, 'train_NoNA_Eq_NoGS_bootstrapped.json')
GS_DATA_PATH = os.path.join(NREDATA_PATH, 'train_NoNA_NoEq_GS_NoNS_bootstrapped.json')

ATT_RSLTS_PATH = './Attention_Rslts2/'
ATT_SCORES_EXTENSION = '_attscores.npy'
ENTITY_PAIR_REL_EXTENSION = '_entity_pair_rel_names.npy'
SENTENCES_EXTENSION = '_sentences.npy'

DICT_KEY_ENTITIES = 'entities'
DICT_KEY_SENTENCES = 'sentences'
DICT_KEY_ATTSCORES = 'attention_scores'
DICT_KEY_IDS = 'ids'



def loadDataWithPickle(data_file_name):
    '''
    This function is necessary to make numpy load work with certain objects/arrays
    :param data_file_name:
    :return: the loaded nujmpy array
    '''
    np_load_old = np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

    ret_arr = np.load(data_file_name)

    # restore old fuctionality
    np.load = np_load_old

    return ret_arr




def getID2Entity(dataset_name):
    '''

    :param dataset_name: the name of the training set to read from; must be OpenNRE data!
    :return: an entity id --> word (name) mapping from the training set
    '''
    id2entity = dict()

    data = readFromJsonFile(dataset_name)
    for entry in data:
        id2entity[entry['head']['id']] = entry['head']['word']
        id2entity[entry['tail']['id']] = entry['tail']['word']

    return id2entity

def getRealEntityPairNames(id2entity, entity_pair_rel_names):
    ids = entity_pair_rel_names.split('#')[:-1]

    names = []
    names[0] = id2entity[ids[0]]
    names[1] = id2entity[ids[1]]

    return (names[0], names[1]), (ids[0], ids[1])


def getBagsMapping(model_name, id2entity):
    att_scores = np.load(os.path.join(ATT_RSLTS_PATH, model_name + ATT_SCORES_EXTENSION))
    entity_pair_rel_names = loadDataWithPickle(os.path.join(ATT_RSLTS_PATH, model_name + ENTITY_PAIR_REL_EXTENSION))
    sentences = loadDataWithPickle(os.path.join(ATT_RSLTS_PATH, model_name + SENTENCES_EXTENSION))

    bags = dict()
    for i in range(len(att_scores)):
        '''get values for CURRENT bag'''

        curr_bag_att_scores = att_scores[i]
        curr_bag_entity_names, curr_bag_ids = getRealEntityPairNames(id2entity, entity_pair_rel_names[i])
        curr_bag_sentences = sentences[i]

        bags[curr_bag_ids] = dict()
        bags[curr_bag_ids][DICT_KEY_ENTITIES] = curr_bag_entity_names
        bags[curr_bag_ids][DICT_KEY_ATTSCORES] = curr_bag_att_scores
        bags[curr_bag_ids][DICT_KEY_SENTENCES] = curr_bag_sentences
        bags[curr_bag_ids][DICT_KEY_IDS] = curr_bag_ids

    return bags

def aggregateDiff(vector1, vector2):
    if len(vector1) != len(vector2):
        raise Exception
    
    diff = 0
    for i in range(vector1):
        diff += abs(vector1[i] - vector2[i])
        
    return diff

def compareBags(bags1, bags2):
    '''
    compare two bags
    :param bags1: a dict of the form bags1[('entity1name', 'entity2name')] = {
                                                    entity_names: (('entity1name', 'entity2name')),
                                                    att_scores: [0.23 0.53 0.12]
                                                    sentences: ['i like this' , 'sentences 2', 'another arandom sentence'] }

    :param bags2: same as bags 1
    :return: return the aggregate difference in att_scores for the same sentences and bags with biggest difference
    '''
    bigger_bags = bags2
    smaller_bags = bags1
    if len(bags1) > len(bags2):
        bigger_bags = bags1
        smaller_bags = bags2

    biggest_diff = 0
    biggest_diff_entry = None
    aggregate_att_diff = 0
    for entity_ids in bigger_bags:
        if entity_ids in smaller_bags:
            # compare the att scores
            att_scores_1 = bigger_bags[entity_ids][DICT_KEY_ATTSCORES]
            att_scores_2 = smaller_bags[entity_ids][DICT_KEY_ATTSCORES]

            diff = aggregateDiff(att_scores_1, att_scores_2)
            aggregate_att_diff += diff

            if(diff > biggest_diff):
                biggest_diff_entry = (bigger_bags[entity_ids], smaller_bags[entity_ids])

    return aggregate_att_diff, biggest_diff_entry


if __name__ == '__main__':
    id2e_gs = getID2Entity(GS_DATA_PATH)
    id2e_reg = getID2Entity(REG_DATA_PATH)
    id2e_na = getID2Entity(NA_DATA_PATH)

    print(loadDataWithPickle(ATT_RSLTS_PATH + 'Wikigender_pcnn_att__NoNA_NoEq_GS_NoNS_bootstrapped_NoDE_attscores.npy'))

    gs_bag = getBagsMapping(getModelNameSimulateArgs(gender_swapped=True, boostrapped=True), id2e_gs)
    #reg_bag = getBagsMapping(getModelNameSimulateArgs(boostrapped=True), id2e_reg)
    #na_bag = getBagsMapping(getModelNameSimulateArgs(name_anonymized=True,boostrapped=True), id2e_na)

    print(gs_bag)




