'''
This file contains functions which are used in more than one file. Thus, this prevents repetition.
All files in the WikigenderJSsonParsing file import Utility functions.
'''
import json
import itertools
import argparse
import os
import subprocess
from sner import Ner
import nltk
import string
import copy

DataTypes = ['train', 'dev', 'male_test', 'female_test'] # names for each of the portions of the datasets
RELATIONS = ['spouse', 'birthDate', 'birthPlace', 'hypernym'] # the relations


def computePositioning(entry, relation, sentence):
    '''

    :return: If the positions are found, return [lower, upper] bounds on entity 1 and 2 pos in sentence. If not found, return [0,0]
    '''
    head_entity_pos = [0, 0]
    tail_entity_pos = [0, 0]
    curr_gender = entry['gender_of_entity1']
    try:
        head_entity_pos = findCharacterPosInSent(entry['entity1'], sentence, soft_matching=True, gender=curr_gender)
    except Exception as e:
        print("Exception for entity1 {}: Exception {}".format(entry['entity1'], e))

    try:
        tail_entity_pos = findCharacterPosInSent(relation['entity2'], sentence)
    except Exception as e:
        print("Exception for entity2 {}: Exception {}".format(relation['entity2'], e))

    return head_entity_pos, tail_entity_pos



def findCharacterPosInSent(entity, sent, soft_matching=False, gender='Male'):
    '''
    NOTE: this was used to create the positions we use in Wikigender.json dataset
    :param entity: the entity we're trying to find. This entity may have a full name which is unlikely to appear
    in the sentence (e.g. Barack Hussein Obama)
    :param sent: The sentence in which we want to find teh entity
    :param soft_matching: using our distant supervision assumption, we're not sure entities will always be in text;
    so we soft match their positioning using pronouns (i.e., if their gender is male, then we look for he or he's; if its female, we look
    for she/she's to find the position of entity in text
    :return: The CHARACTER range of the entity in sent (e.g., (index_of_starting_char, index_of_last_char)

    To find the entity, we find all combinations of the entity's name
    (e.g. barack hussein obama, barack hussein, barack obama, barack, hussein, obama) in order from longest to shortest
    and return the first occurrence of that longest portion of the enitty's name. This way, we can hopefully always
    obtain the position of the correct entity (e.g. instead of searching just the name obama, when that may be ambiguous)
    '''
    MALE_PRONOUN_LIST = ['he', 'He', 'he\'s', 'He\'s']
    FEMALE_PRONOUN_LIST = ['she', 'She', 'she\'s', 'She\'s']

    def cleanWord(string, chars_to_clean):
        ret_str = ''
        for c in string:
            if c not in chars_to_clean:
                ret_str += c
        return ret_str

    def getPosRangeForNames(names, blank_chars):
        '''

        :param names: a list of the names you want to look through
         here, if we have ['barack', 'hussein', 'obama'], then we:
          look for barack hussein obama
          then for barack hussein
          then for barack obama
          then for barack
          then for hussein
          then for obama
        :param blank_chars
        :return: the character range of the occurrence of the biggest combination of names we're looking for
        (e.g. ,barack hussein obama has priority over barack obama)
        range is returned as a list e.g. the range from a to b (a is inclusive, b is exclusive) is returned as the list [a,b] ( [a,b) --> [a,b] )
        '''

        def genCharacterNGrams(str, n):
            return [str[i:i + n] for i in range(len(str) - 1)]

        pos = -1
        pos_range = [-1, -1]  # initial value
        # look at all combinations of entity name
        for L in range(len(names), 0, -1):
            for subset in itertools.combinations(names, L):
                name = ' '.join(subset)

                pos = sent.find(name, 0, len(name))  # if it's at the beginning of sentence
                pos_range = [pos, pos + len(name) + 1]
                if pos == -1:
                    char_ngrams = genCharacterNGrams(sent, len(name) + 2)

                    for i in range(len(char_ngrams)):
                        char_ngram = char_ngrams[i]
                        if char_ngram[1:len(char_ngram) - 1] == name and char_ngram[0] and char_ngram[
                            0] in blank_chars and (char_ngram[-1] in blank_chars or char_ngram[-1] == 's'):
                            pos = i + 1
                            pos_range = [pos, pos + len(name)]
                            break

                if pos == -1:
                    pos = sent.find(name, len(sent) - len(name), len(sent))  # see if it's the last word in teh setnence
                    pos_range = [pos, pos + len(name) + 1]

                if pos != -1 and pos_range != [-1, -1]:
                    return pos_range

        # if we make it out here, then entity is not in sentence
        return [-1, -1]

        # raise Exception('Entity is not in sentence')

    # for doing character processing
    whitespace = set(string.whitespace)
    punc = set(string.punctuation)
    blank_chars = whitespace.union(punc)

    # look for the enitty's names
    # get all entity names
    entity = cleanWord(entity, punc.difference('-'))  # leave the hiphen because that's often legitimate
    names = entity.split()
    pos_range = getPosRangeForNames(names, blank_chars)

    if pos_range == [-1, -1]:
        # then, entity was not found in teh sentence
        # if we can do soft matching, we can try that
        if soft_matching:
            # we can look for pronouns
            pronoun_list = None
            if gender == 'Male':
                pronoun_list = MALE_PRONOUN_LIST
            else:
                pronoun_list = FEMALE_PRONOUN_LIST

            for i in range(len(pronoun_list)):
                pos_range = getPosRangeForNames([pronoun_list[i]], blank_chars)
                if pos_range != [-1, -1]:
                    return pos_range

            # if we get out here, then the entity is definitely not in the sentence
            # we will set a default value to 0
            #raise Exception("Entity not found in sentence!")
            pos_range = [0, 0]

    if pos_range == [-1,-1]:
        return [0,0]
    else:
        return pos_range


def findWordPosInSent(entity, sent):
    '''
        :param entity: the entity we're trying to find. This entity may have a full name which is unlikely to appear
        in the sentence (e.g. Barack Hussein Obama)
        :param sent: The sentence in which we want to find teh entity
        :return: The WORD range of the entity in sent (e.g., (index_of_starting_word, index_of_last_word)

        To find the entity, we find all combinations of the entity's name
        (e.g. barack hussein obama, barack hussein, barack obama, barack, hussein, obama) in order from longest to shortest
        and return the first occurrence of that longest portion of the enitty's name. This way, we can hopefully always
        obtain the position of the correct entity (e.g. instead of searching just the name obama, when that may be ambiguous)
        '''
    # get all entity names
    names = entity.split()

    sent_words = nltk.word_tokenize(sent)

    # look at all combinations of entity name
    for L in range(len(names), -1, -1):
        for subset in itertools.combinations(names, L):
            pos = findListSubset(subset, sent_words)

            if pos != -1:
                return pos

    # if we make it out here, then entity is not in sentence
    raise Exception('Entity is not in sentence')


def findListSubset(subset, full_list):
    '''
    Finds the first occurrence of subset in full list and returns its positioning
    '''
    i = 0
    while i < len(full_list):
        old_i = i
        for j in range(len(subset)):
            # print("{}, {}, {}, {}".format(i, full_list[i], j, subset[j]))
            if full_list[i] == subset[j]:
                if j == len(subset) - 1:
                    return old_i, i + 1
                i += 1
            else:
                break
        i = old_i + 1

    return -1


# start the NER server up
'''
import os
import subprocess
os.chdir('./StanfordNER/')
subprocess.Popen(['./run_stanfordner_server.sh']) # start the server up
os.chdir('../')
# get the NER object
from sner import Ner
ner_tagger = Ner(host='localhost',port=9199)
'''
def startNERServer():
    #os.chdir('./StanfordNER/')
    #subprocess.Popen(['./run_stanfordner_server.sh'])  # start the server up
    #os.chdir('../')
    # get the NER object
    ner_tagger = Ner(host='localhost', port=9199)

    return ner_tagger

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




def getSpecificEntries(entries, names, relation=None):
    '''
    Parameters:
    - entries: an object representing all the entires in dict form
    - relation: IF THIS IS NOT NONE, then we ONLY want to return instances from a particular relaiton!!
    Returns:
    - an object like entires but only containing entries that have names in the names hashset passed in
    '''
    specific_entries = list()

    for entry in entries:

        name = entry['entity1']
        if name in names:
            if relation == None:
                specific_entries.append(copy.deepcopy(entry))
            else:
                for entry_relation in entry['relations']:
                    if entry_relation['relation_name'] == relation:
                        new_entry = dict()
                        new_entry['entity1'] = entry['entity1']
                        new_entry['relations'] = list()
                        new_entry['relations'].append(entry_relation)
                        specific_entries.append(new_entry)

    return specific_entries

def getTrueFalseCombos(num):
    items = [True, False]
    return [list(i) for i in itertools.product(items, repeat=num)]

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', nargs='?', default='Wikigender')
    parser.add_argument('--encoder', nargs='?', default='pcnn')
    parser.add_argument('--selector', nargs='?', default='att')
    parser.add_argument("--gender_swap", "-gs", action="store_true")
    parser.add_argument("--equalized_gender_mentions", "-egm", action="store_true")
    parser.add_argument("--swap_names", "-sn", action="store_true")
    parser.add_argument("--name_anonymize", "-na", action="store_true")
    parser.add_argument("--debiased_embeddings", "-de", action="store_true")
    parser.add_argument("--bootstrapped", "-bs", action="store_true")
    parser.add_argument("--neutralize", "-nt", action="store_true")
    args = parser.parse_args()

    args.name_anonymize = name_anonymized
    args.gender_swap = gender_swapped
    args.equalized_gender_mentions = equalized_gender_mentions
    args.debiased_embeddings = debiased_embeddings
    args.swap_names = swap_names
    args.bootstrapped = boostrapped
    args.neutralize = neutralize

    return getNameSuffix(args)



def getModelName(args, dataset_name, encoder, selector):
    name_suffix = getNameSuffix(args)
    if args.debiased_embeddings:
        name_suffix += "_DE"
    else:
        name_suffix += "_NoDE"

    if name_suffix == '':
        return dataset_name + "_" + encoder + "_" + selector
    elif name_suffix == '_DE' or name_suffix == '_NoDE':
        return dataset_name + "_" + encoder + "_" + selector + name_suffix
    else:
        return dataset_name + "_" + encoder + "_" + selector + "_" + name_suffix


def getModelNameSimulateArgs(equalized_gender_mentions=False, gender_swapped=False, name_anonymized=False, swap_names=False, debiased_embeddings=False, boostrapped=False, encoder="pcnn", selector="att", neutralize=False):
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

    args.name_anonymize = name_anonymized
    args.gender_swap = gender_swapped
    args.equalized_gender_mentions = equalized_gender_mentions
    args.debiased_embeddings = debiased_embeddings
    args.swap_names = swap_names
    args.bootstrapped = boostrapped
    args.neutralize = neutralize

    return getModelName(args, encoder=encoder, selector=selector)



def getWordEmbeddingFileName(args):
    if args.debiased_embeddings:
        return 'debiased_' + 'word_vec_' + getNameSuffix(args) + '.json'
    else:
        return 'word_vec_' + getNameSuffix(args) + '.json'


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

