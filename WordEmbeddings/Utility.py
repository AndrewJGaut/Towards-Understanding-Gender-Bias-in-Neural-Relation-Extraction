'''
This file contains functions which are used in more than one file. Thus, this prevents repetition.
All files in the WikigenderJSsonParsing file import Utility functions.
'''
import json
import argparse
import itertools

DataTypes = ['train', 'dev', 'male_test', 'female_test'] # names for each of the portions of the datasets

# start the NER server up
import os




def writeToJsonFile(data, outfile_name, prettify=False):
    with open(outfile_name, 'w') as outfile:
        if(prettify):
            json.dump(data, outfile, indent=4, sort_keys=True)
        else:
            json.dump(data, outfile)

def readFromJsonFile(infile_name):
    with open(infile_name, 'r') as infile:
        return json.load(infile)
    return ""


def getNamesFromFileToDict(filename):
    file = open(filename, 'r')
    namesDict = set()

    for line in file.readlines():
        namesDict.add(line.strip())

    return namesDict

def getMaleAndFemaleNames():
    male_names = getNamesFromFileToDict('NamesAndSwapLists/male_names.txt')
    female_names = getNamesFromFileToDict('NamesAndSwapLists/female_names.txt')

    return male_names, female_names


def getAllEntries(data):
    '''
    :param data: has ALL the data in Wikigender format (including train, dev, amle_test, etc)
    :return: all entries from every data set
    '''
    entries = list()
    for data_type in DataTypes:
        curr_data = data[data_type]
        for entry in curr_data:
            entries.append(entry)
    return entries

def getEntriesForDataType(data, data_type):
    '''
    :param data: has ALL the data in Wikigender format (including train, dev, amle_test, etc)
    :param data_type: train, test, dev, etc.
    :return: all entries from teh data_type portion of the data
    '''
    entries = list()
    curr_data = data[data_type]
    for entry in curr_data:
        entries.append(entry)
    return entries


def addPunctuationToTags(pos_tags, word_tokenization):
    '''

    :param pos_tags: pos_tags are the pos_tags for the sentence
    these should be a list with each element a tuple like so: [('Jack', 'PERSON'), ('was', 'O'), etc.]
    crucially, this does NOT contain punctuation information!
    :param word_tokenize: the same sentence as for the pos tags, but just word tokenized
    this DOES include punctuation information
    :return: the pos_tags list but with punctuation information added
    '''
    ret_pos_tags = list()
    pos_tag_index = 0
    for word_index in range(len(word_tokenization)):
        word = word_tokenization[word_index]
        if pos_tag_index < len(pos_tags) and word == pos_tags[pos_tag_index][0]:
            ret_pos_tags.append(pos_tags[pos_tag_index])
            pos_tag_index += 1
        else:
            ret_pos_tags.append((word, 'O'))

    return ret_pos_tags, word_tokenization




def getSpecificEntries(entries, names):
    '''
    Parameters:
    - entries: an object representing all the entires in dict form
    Returns:
    - an object like entires but only containing entries that have names in the names hashset passed in
    '''
    specific_entries = list()

    for entry in entries:
        name = entry['entity1']
        if name in names:
            specific_entries.append(entry)

    return specific_entries




def getNameSuffix(args):
    name_suffix = ''

    if not args.name_anonymize and not args.equalized_gender_mentions and not args.gender_swap:
        # then we don't want to use any suffix or anything; we're using the regular dataset
        return ''

    if args.name_anonymize:
        name_suffix += "_NA"
    else:
        name_suffix += "_NoNA"
    if args.equalized_gender_mentions:
        name_suffix += "_Eq"
    else:
        name_suffix += "_NoEq"
    if args.gender_swap:
        name_suffix += "_GS"
        if args.swap_names:
            name_suffix += "_NS"
        else:
            name_suffix += "_NoNS"
        if args.neutralize:
            name_suffix += "NT"
    else:
        name_suffix += "_NoGS"

    if args.bootstrapped:
        name_suffix += "_bootstrapped"

    return name_suffix

def getNameSuffixSimulateArgs(equalized_gender_mentions=False, gender_swapped=False, name_anonymized=False, swap_names=False, debiased_embeddings=False, boostrapped=False, neutralize=False):
    args = getCommandLineArgs()

    args.name_anonymize = name_anonymized
    args.gender_swap = gender_swapped
    args.equalized_gender_mentions = equalized_gender_mentions
    args.debiased_embeddings = debiased_embeddings
    args.swap_names = swap_names
    args.bootstrapped = boostrapped
    args.neutralize = neutralize

    return getNameSuffix(args)

def getWordEmbeddingFileName(suffix, extension):
    return 'word_vec_' + suffix + extension

def getTrueFalseCombos(num):
    items = [True, False]
    return [list(i) for i in itertools.product(items, repeat=num)]




def getCommandLineArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', nargs='?', default='Wikigender')
    parser.add_argument('--encoder', nargs='?', default='pcnn')
    parser.add_argument('--selector', nargs='?', default='att')
    parser.add_argument("--gender_swap", "-gs", action="store_true")
    parser.add_argument("--equalized_gender_mentions", "-egm", action="store_true")
    parser.add_argument("--swap_names", "-sn", action="store_true")
    parser.add_argument("--name_anonymize", "-na", action="store_true")
    parser.add_argument("--debiased_embeddings", "-de", action="store_true")
    parser.add_argument("--neutralize", "-nt", action="store_true")
    parser.add_argument("--bootstrapped", "-bs", action="store_true")
    args = parser.parse_args()

    return args
