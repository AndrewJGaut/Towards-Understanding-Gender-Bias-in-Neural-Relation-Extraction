'''This file contains different debiasing methods for the json dataset including:
- gender-swapping
- name-anonymzation
- equalization of gender mentions thorugh sampling
Wikigender format:
{
'train':
 [ {
    entity1: NAME,
    relations: [
        {
            relation_name: spouse,
            entity2: NAME,
            sentences: [
            ]
        }

    ]
} ],
'dev' : [ {
    entity1: NAME,
    relations: [
        {
            relation_name: spouse,
            entity2: NAME,
            sentences: [
                ...
            ]
            positions:[
                {
                    entity1: INT,
                    entity2: INT
                },
                ...
            ] (THESE ARE THE SAME LENGTH AS TEH SENTENCES where positions[i] gives entity1, entity2 positions in sent[i]
        }

    ]
}, ... ], ...
}
'''
from Utility import *
from genderSwap import *
import nltk
import random
import os
import subprocess
import copy





'''genderswapping'''
def genderSwapSubsetOfDataset(entries, swap_names, neutralize):
    '''

    :param entries: a list of JSON dictionary objects from the dataset
            have the following format: [ {
                                            entity1: NAME,
                                            relations: [
                                                {
                                                    name: spouse,
                                                    entity2: NAME,
                                                    sentences: [
                                                    ]
                                                }

                                            },
                                            ...
                                        ]
    :param swap_names: if true, then the names will be gender-swapped as well
    :return: entries with all SENTENCES gender-swapped only
    '''
    new_entries = list()
    for index in range(len(entries)):
        entry = entries[index] # get the current entry
        new_entry = copy.deepcopy(entry) # we need to deepcopy this so they don't both share the reference, point to same object

        # overwrite gender of entity
        if new_entry['gender_of_entity1'] == 'Male':
            new_entry['gender_of_entity1'] = 'Female'
        else:
            new_entry['gender_of_entity1'] = 'Male'

        relations = entry['relations']
        for relation_index in range(len(relations)):
            relation = relations[relation_index]
            sentences = relation['sentences']
            positions = relation['positions']
            for sentence_index in range(len(sentences)):
                sentence = sentences[sentence_index] # get current sentence
        
                if(index % 100 == 0 and sentence_index == 0):
                    print("INDEX: {}; {}".format(index, sentence))

                # gender swap sentence
                genderswapped_sentence = genderSwap(sentence, swap_names, neutralize)
                new_entry['relations'][relation_index]['sentences'][sentence_index] = genderswapped_sentence



                # recompute positioning
                entity1_pos, entity2_pos = computePositioning(new_entry, relation, sentence)
                new_entry['relations'][relation_index]['positions'][sentence_index] = {'entity1': entity1_pos, 'entity2': entity2_pos}

        # now, set overwrite the old entry in entries with the new entry with gender_swapped sentences
        new_entries.append(new_entry)

    return new_entries




def createGenderSwappedDatasetEntries(json_data, swap_names=False, neutralized=False):
    '''

    :param data: data in wikigender format
    :param swap_names: true if the names should be swapped as well
    :return: json_data with all sentences gender-swapped; also writes that data to the file with outfile_name
    '''
    #overwrite with gender-swapped stuff
    if neutralized:
        # neutralize everything
        json_data['train'] = genderSwapSubsetOfDataset(json_data['train'], swap_names, neutralized)
        json_data['dev'] = genderSwapSubsetOfDataset(json_data['dev'], swap_names, neutralized)
        json_data['male_test'] = genderSwapSubsetOfDataset(json_data['male_test'], swap_names, neutralized)
        json_data['female_test'] = genderSwapSubsetOfDataset(json_data['female_test'], swap_names, neutralized)

    else:
        json_data['train'].extend(genderSwapSubsetOfDataset(json_data['train'], swap_names, neutralized)) # train on union
        json_data['dev'].extend(genderSwapSubsetOfDataset(json_data['dev'], swap_names, neutralized)) # dev test on union
        # leave the test sets the same
        json_data['male_test'] = json_data['male_test']
        json_data['female_test'] = json_data['female_test']

    return json_data

'''end genderswapping'''




def getLenEntries(entries):
    len_counter = 0
    for entry in entries:
        for relation in entry['relations']:
            sentences = relation['sentences']
            len_counter += len(sentences)

    return len_counter

def getRandomArrayIndex(array):
    if(len(array) > 0):
        return random.randint(0, len(array)-1)
    return None

def getEqualizedEntriesThroughDownSampling(entries, male_names, female_names):
    '''
    :param entries: the list of JSON dictionary entries in the dataset; includes entity1, and all entity2,sentences pairs for each attribute
    :param male_names: hashset of male names from Wikipedia
    :param female_names: hashset of female names from wikipedia
    :return: entries, but with equalized gender numbers in train and dev set THROUGH REMOVING DATAPOINTS FROM LESSER CLASS
                here, for example, if there are less male datapoints than female, then we duplicate the male datapoints until there are as many male as female datapoints
                note that we randomly sample to do this duplication, so the order is randomized
    '''
    # get male and female entries
    male_entries = getSpecificEntries(entries, male_names)
    female_entries = getSpecificEntries(entries, female_names)

    # find lesser of the two
    smaller_entries = male_entries
    larger_entries = female_entries
    if(getLenEntries(female_entries) < getLenEntries(male_entries)):
        smaller_entries = female_entries
        larger_entries = male_entries

    # now, randomly sample until the smaller thing has as many datapoints as the larger one
    while(getLenEntries(smaller_entries) < getLenEntries(larger_entries)):
        entry_index = getRandomArrayIndex(larger_entries)
        if entry_index == None:
            continue
        entry = larger_entries[entry_index]

        # now, choose a random relation
        relations = entry['relations']
        rel_index = getRandomArrayIndex(relations)
        if rel_index == None:
            continue
        relation = relations[rel_index]

        # now, choose random sentence
        sentences = relation['sentences']
        sent_index = getRandomArrayIndex(sentences)
        #print('{}, {}, {}'.format(entry_index, rel_index, sent_index))
        #print('{}, {}'.format(getLenEntries(smaller_entries), getLenEntries(larger_entries)))
        if sent_index != None:
            # remove sentence
            entry['relations'][rel_index]['sentences'].pop(sent_index)

            # remove the position associated with this sentence
            entry['relations'][rel_index]['positions'].pop(sent_index)


        larger_entries[entry_index] = entry

    larger_entries.extend(smaller_entries)
    return larger_entries

def createEqualizedJsonDataset(old_dataset_name, new_dataset_name):
    '''
    :param old_dataset_name:
    :param new_dataset_name:
    :return:
    '''

    data = readFromJsonFile(old_dataset_name)

    male_names = getNamesFromFileToDict('NamesAndSwapLists/male_names.txt')
    female_names = getNamesFromFileToDict('NamesAndSwapLists/female_names.txt')

    equalized_train_data = getEqualizedEntriesThroughDownSampling(data['train'], male_names, female_names)
    equalized_dev_data = getEqualizedEntriesThroughDownSampling(data['dev'], male_names, female_names)
    male_test_data = data['male_test']  # this stays constant
    female_test_data = data['female_test']  # this stays constant

    # just overwrite the old stuff in data
    data['train'] = equalized_train_data
    data['dev'] = equalized_dev_data
    data['male_test'] = male_test_data
    data['female_test'] = female_test_data

    # write to file
    writeToJsonFile(data, new_dataset_name)

    return data

def createEqualizedJsonDatasetEntries(data):
    '''
    :param data: json data from Wikigender style file
    :return: that data with equalized entries
    '''
    male_names = getNamesFromFileToDict('NamesAndSwapLists/male_names.txt')
    female_names = getNamesFromFileToDict('NamesAndSwapLists/female_names.txt')

    equalized_train_data = getEqualizedEntriesThroughDownSampling(data['train'], male_names, female_names)
    equalized_dev_data = getEqualizedEntriesThroughDownSampling(data['dev'], male_names, female_names)
    male_test_data = data['male_test']  # this stays constant
    female_test_data = data['female_test']  # this stays constant

    # just overwrite the old stuff in data
    data['train'] = equalized_train_data
    data['dev'] = equalized_dev_data
    data['male_test'] = male_test_data
    data['female_test'] = female_test_data

    return data


'''end equalize mentions'''


'''name anonymization'''

def clean(str, sentence = False):
    '''
    What it does:
        Removes all non-alphanumerics from word AND makes the word singular (not plural)
        Used to check if words are in the set
    '''
    cleanStr = ""
    for char in str:
        if (char.isalpha()):
            cleanStr += char.lower()
        if(sentence):
            if(char.isspace()):
                cleanStr += char

    return cleanStr

def addNameToAnonymizationDict(input_str, dictionary, name_counter):
    '''

    :param input_str: striong representing name
    :param dictionary: names_2_anonymizaiton dicitonary mapping name to E + number (dictionary[Mary] --> E1)
    :param name_counter: tells us what number name it is (for the E + number anoymized name)
    :return: the dictionary with name added and name_counter
    '''

    # split the name up and add each thing separetly
    names = nltk.word_tokenize(input_str)
    for name in names:
        name = clean(name)
        if name not in dictionary:
            dictionary[name] = "E" + str(name_counter)
            name_counter += 1

    return dictionary, name_counter





def createNameAnonymizationDict(entries):
    '''

    :param entries: json data to be anonymized later in teh foloowing format:
                                        [ {
                                            entity1: NAME,
                                            relations: [
                                                {
                                                    name: spouse,
                                                    entity2: NAME,
                                                    sentences: [
                                                    ]
                                                }

                                            },
                                            ...
                                        ]
    :return: A dict mapping names in input_str to anonymized names (e.g. dict[Mary] = E1, dict[John] = E2, etc.)
    Note: we assume that any entity2 for the spouse relation will be a name and any entity1 is a name! (this should alwasy be true; both of these should be people)
    '''

    print('creating name anonymization dict')

    ner_tagger = startNERServer()
    names_2_anonymizations = dict()
    name_counter = 0

    for index in range(len(entries)):
        entry = entries[index] # get the current entry

        # add entity 1
        #print('anonymizing entity1: ' + entry['entity1'])
        names_2_anonymizations, name_counter = addNameToAnonymizationDict(entry['entity1'], names_2_anonymizations, name_counter)

        relations = entry['relations']
        for relation in relations:
            relation_name = relation['relation_name']
            if 'spouse' in clean(relation_name):
                # then, we want to get anonymization dict for entity2
                names_2_anonymizations, name_counter = addNameToAnonymizationDict(relation['entity2'], names_2_anonymizations, name_counter)

            # now get anonymization dict for all the sentences
            sentences = relation['sentences']
            for sentence_index in range(len(sentences)):
                sentence = sentences[sentence_index] # get current sentence
                pos_tags = ner_tagger.get_entities(sentence)
                pos_tags, _ = addPunctuationToTags(pos_tags, word_tokenize(sentence))
                for i in range(len(pos_tags)):
                    if pos_tags[i][1] == 'PERSON':
                        # then, this is a person
                        clean_name = clean(pos_tags[i][0])
                        names_2_anonymizations, name_counter = addNameToAnonymizationDict(clean_name, names_2_anonymizations,name_counter)

    return names_2_anonymizations


def nameAnonymizeJson(entries, names_2_anonymizations):
    '''
    :param entries:
    :param names_2_anonymizations:
    :return:
    '''
    for index in range(len(entries)):
        entry = entries[index]  # get the current entry

        # anonymize e1
        entry['entity1'] = nameAnonymizeStr(clean(entry['entity1'], True), names_2_anonymizations)

        relations = entry['relations']
        for relation_index in range(len(relations)):
            relation = relations[relation_index]
            relation_name = relation['relation_name']
            if 'spouse' in clean(relation_name):
                # then, anonymize entity2
                entry['relations'][relation_index]['entity2'] = nameAnonymizeStr(clean(entry['relations'][relation_index]['entity2']), names_2_anonymizations)

            # now anonymize all the sentences
            sentences = relation['sentences']
            for sentence_index in range(len(sentences)):
                sentence = sentences[sentence_index]
                entry['relations'][relation_index]['sentences'][sentence_index] = nameAnonymizeStr(sentence, names_2_anonymizations)

                # recompute positioning
                entity1_pos, entity2_pos = computePositioning(entry, relation, entry['relations'][relation_index]['sentences'][sentence_index])
                entry['relations'][relation_index]['positions'][sentence_index] = {'entity1': entity1_pos,'entity2': entity2_pos}

        entries[index] = entry
    return entries


def join_punctuation(seq, characters='.,;?!'):
    '''
    :param seq: string to have punctuation joined
    :param characters:
    :return: joins the punctuation in a word
    '''
    characters = set(characters)
    seq = iter(seq)
    current = next(seq)

    for nxt in seq:
        if nxt in characters:
            current += nxt
        else:
            yield current
            current = nxt

    yield current

def nameAnonymizeStr(input_str, names_2_anonymizations):
    '''
    This takes str and replaces all names in str with their corresponding anonymizations
    This function should ONLY be called after createAnonymizationDict was run on input_str, and names_2_anonymizations should be return value from CreateAnonymizationDict(input_str)
    '''
    out_str = ""

    for line in input_str.split('\n'):
        words = nltk.word_tokenize(line)
        for i in range(len(words)):
            word = clean(words[i])
            if word in names_2_anonymizations:
                words[i] = names_2_anonymizations[word]
        out_str += ' '.join(join_punctuation(words))

    return out_str

def nameAnonymizeSubsetOfDataset(entries):
    '''
    :param entries: colletion of json data; should be data['train'] or data[some_key], NOT data itself
    :return: those entries name anonymized
    '''
    names_2_anonymizations = createNameAnonymizationDict(entries)
    return nameAnonymizeJson(entries, names_2_anonymizations)


def createNameAnonymizedJsonDataset(infile_name, outfile_name):
    '''
    Replaces all names in dataset with E1, E2, ..., En (if there are n entities in the dataset) using a mapping
    Then returns a new dataset
    '''
    data = readFromJsonFile(infile_name)

    # get anonymization dict
    names_2_anonymizations = createNameAnonymizationDict(getAllEntries(data))

    #now, we need to anonymize the dataset
    for data_type in DataTypes:
        #print('anonyjmizing ' + data_type)
        data[data_type] = nameAnonymizeJson(data[data_type], names_2_anonymizations)
        #print('done anonymizing ' + data_type)

    # write to file
    writeToJsonFile(data, outfile_name, True)

    # now, return the data
    return data

def createNameAnonymizedJsonDatasetEntries(data):
    '''
    :param data: data in wIkigender format
    :return: input data with name anonymization applied
    '''
    # get anonymization dict
    names_2_anonymizations = createNameAnonymizationDict(getAllEntries(data))

    # now, we need to anonymize the dataset
    for data_type in DataTypes:
        #print('anonymizing ' + data_type)
        data[data_type] = nameAnonymizeJson(data[data_type], names_2_anonymizations)
        #print('done anonymizing ' + data_type)

    # now, return the data
    return data

'''end name anonymization'''


def createDebiasedDataset(infile_name, equalized=False, name_anonymized=False, gender_swapped=False, swap_names=False, neutralized=False, name_anonymized_data=None, gender_equalized_data=None):
    '''
    :param infile_name: file that holds the original, unaltered json dataset (should be in Wikigender format)
    :param equalized: boolean flag; if true, the returned data will have equalized mentions
    :param name_anonymized: boolean flag; if true, the returned data will be name anonymized
    :param gender_swapped: boolean flag; if true, the returned data will be gender-swapped
    :param swap_names: boolean flag; if true, and gender_swapped is true, then returend data will be gender-swapped with names also gender-swapped
    :param name_anonymized_entries: the name anonymized entries from the infile. The reason we have this flag is so that we can anonymize those entires just once then pass them into each function. Thisi s bedcause anme anonymization takes a very long time.
    :return: the debiased dataset, with the debiasing customized based on input flags
    '''

    print('creating dataset: {}, {}, {}, {}'.format(equalized, name_anonymized, gender_swapped, swap_names))

    data = readFromJsonFile(infile_name)
    infile_names = infile_name.split('.')

    infile_names[0] += getNameSuffixSimulateArgs(equalized_gender_mentions=equalized, name_anonymized=name_anonymized, gender_swapped=gender_swapped, swap_names=swap_names, neutralize=neutralized)

    # don't recreate the same dataset
    outfile_name = infile_names[0] + "." + infile_names[1]
    if(os.path.exists(outfile_name)):
        print("DATASET ALREADY EXISTS!")
        return


    if equalized:
        if gender_equalized_data == None:
            data = createEqualizedJsonDatasetEntries(data)
        else:
            data = gender_equalized_data
    if name_anonymized:
        if gender_equalized_data != None or name_anonymized_data == None:
            data = createNameAnonymizedJsonDatasetEntries(data)
        else:
            data = name_anonymized_data
    if gender_swapped:
        data = createGenderSwappedDatasetEntries(data, swap_names, neutralized)



    # write the new dataset
    outfile_name = infile_names[0] + "." + infile_names[1]
    writeToJsonFile(data, outfile_name)

    print('dataset created')

    return data


def createAllDebiasedDatasets(infile_name):
    '''

    :param infile_name:
    NOTE: right now we're not doing swap names
    :return: Create all debiased datastse for dataset at infile_name with all combos
    '''
    data = readFromJsonFile(infile_name)
    name_anonymized_data = createNameAnonymizedJsonDatasetEntries(data)
    #equalized_data = createEqualizedJsonDatasetEntries(data)


    true_false_combos = getTrueFalseCombos(4)
    for combo in true_false_combos:
        createDebiasedDataset(infile_name, name_anonymized=combo[0], gender_swapped=combo[1], equalized=combo[2], neutralized=combo[3], swap_names=False, name_anonymized_data= name_anonymized_data)
    #createDebiasedDataset(infile_name, equalized=True, gender_equalized_data=equalized_data)
    #createDebiasedDataset(infile_name, equalized=True, gender_swapped=True, gender_equalized_data=equalized_data)
    #createDebiasedDataset(infile_name, equalized=True, name_anonymized=True, gender_equalized_data=equalized_data)
    #createDebiasedDataset(infile_name, equalized=True, name_anonymized=True, gender_swapped=True, gender_equalized_data=equalized_data)
    #createDebiasedDataset(infile_name, equalized=True, name_anonymized=True, gender_swapped=True, swap_names=True)
    #createDebiasedDataset(infile_name, name_anonymized=True, name_anonymized_data=name_anonymized_data)
    #createDebiasedDataset(infile_name, name_anonymized=True, gender_swapped=True, name_anonymized_data=name_anonymized_data)
    #createDebiasedDataset(infile_name, name_anonymized=True, gender_swapped=True, swap_names = True)
    #createDebiasedDataset(infile_name, gender_swapped=True)
    #createDebiasedDataset(infile_name, swap_names=True)

def createDebiasedDatasetWithArgs(args):
    old_args = args
    args.bootstrapped=False
    name_suffix = getNameSuffix(args)
    args = old_args

    data = readFromJsonFile('JsonData/' + args.dataset + '.json')

    infile_names = args.dataset + name_suffix

    # don't recreate the same dataset
    outfile_name = 'JsonData/' + infile_names + ".json"
    if (os.path.exists(outfile_name)):
        print("DATASET {} ALREADY EXISTS!".format(outfile_name))
        return

    if args.equalized_gender_mentions:
        data = createEqualizedJsonDatasetEntries(data)
    if args.name_anonymize:
        data = createNameAnonymizedJsonDatasetEntries(data)
    if args.gender_swap:
        data = createGenderSwappedDatasetEntries(data, args.swap_names, args.neutralize)

    writeToJsonFile(data,outfile_name)

    print('dataset created at path ' + outfile_name)

    return data


'''main'''
if __name__ == '__main__':
    # for usage with the central controlling bash script for bootstrapping
    os.chdir('./WikigenderJsonParsing')
    args = getCommandLineArgs()
    createDebiasedDatasetWithArgs(args)
    os.chdir('../')

    # now, create the different datasets
    #createAllDebiasedDatasets('JsonData/Wikigender.json')

