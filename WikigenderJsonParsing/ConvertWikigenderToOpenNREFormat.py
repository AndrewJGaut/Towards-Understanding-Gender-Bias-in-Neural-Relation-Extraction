'''NOTE: We may want to change this file name to be ConvertToPerEntryFormat (so we can add position embeddings for any architecure)'''

from Utility import *
import argparse
import os
import random
import re



def genId():
    '''
    :return: an id in format that OpenNRE recognizes: /guid/9202a8c04000641f80000000000xxxxx, with the last 5 x's being charaters generated in the function
    '''
    chars = "abcdefghijklmnopqrstuvwxyz0123456789"

    id = '/guid/9202a8c04000641f80000000000'

    for i in range(5):
        id += chars[random.randint(0, 35)]

    return id

def genIdMapping(all_data):
    '''
    :param all_data: a list of entries for ALL datasets! (train, dev, AND test)
    :return: a mapping from an entity to an id AND a hashset containing all created IDs
    note that each ID for each pair must be unique!
    '''
    entity_to_id = dict()
    previously_created_ids = set()

    for entry in all_data:

        # add entity 1
        e1 = entry['entity1']
        if e1 not in entity_to_id:
            id = genId()
            while (id in previously_created_ids):
                # generate until we get a unique one (shouldn't take long)
                id = genId()
            entity_to_id[e1] = id
            previously_created_ids.add(id)

        # add entity 2's
        for relation in entry['relations']:
            e2 = relation['entity2']
            if e2 not in entity_to_id:
                id = genId()
                while (id in previously_created_ids):
                    # generate until we get a unique one (shouldn't take long)
                    id = genId()
                entity_to_id[e2] = id
                previously_created_ids.add(id)

    return entity_to_id




def convertEntryToOpenNREFormat(entry):
    pass

def convertEntriesToOpenNREFormat(entries, entity_to_id):
    '''

    :param entries:
    :param entity_2_id:
    :return:
    '''
    json_data = list()
    for entry in entries:
        e1 = entry['entity1']
        for relation in entry['relations']:
            e2 = relation['entity2']

            positions = relation['positions']
            for i in range(len(relation['sentences'])):
                sentence = relation['sentences'][i]
                # get the current entry
                # NOTE: for newer versions of OpenNRE, change 'sentence' to 'text', 'head' to 'h' and 'tail' to 't'
                curr_json_data = dict()
                curr_json_data['sentence'] = sentence
                curr_json_data['relation'] = relation['relation_name']

                curr_json_data['head'] = dict()
                curr_json_data['head']['word'] = e1
                curr_json_data['head']['id'] = entity_to_id[e1]
                curr_json_data['head']['pos'] = positions[i]['entity1']

                curr_json_data['tail'] = dict()
                curr_json_data['tail']['word'] = e2
                curr_json_data['tail']['id'] = entity_to_id[e2]
                curr_json_data['tail']['pos'] = positions[i]['entity2']

                json_data.append(curr_json_data)

    return json_data

def getSuffixNameFromFileName(infile_name):
    '''

    :param infile_name: the file name to obtain the suffix name from
    :return: the suffix name for the file name
    for instance, if the file name is: Wikigender_NoEq_NA_GS.json,
    thisf unction should return _NoEq_NA_GS
    '''
    infile_name = infile_name.strip()
    if infile_name[0] == '.':
        infile_name = infile_name[1:]
    name_suffix = infile_name.split('.')[0]
    if '/' in name_suffix:
        name_suffix = name_suffix.split('/')[-1]
    if '_' in name_suffix:
        name_suffix = ''.join(re.split('(_)', name_suffix)[1:])

    return name_suffix

def createOpenNREFile(infile_name, name_suffix):
    # get the file
    data = readFromJsonFile(infile_name)


    # prep some data
    all_entries = getAllEntries(data)
    entity_to_id = genIdMapping(all_entries)

    # obtain data and generate datasets
    for data_type in DataTypes:
        curr_data = data[data_type]
        curr_data = convertEntriesToOpenNREFormat(curr_data, entity_to_id)
        print('creating {}'.format('OpenNREData/' + data_type + name_suffix + '.json'))
        writeToJsonFile(curr_data, 'OpenNREData/' + data_type + name_suffix + '.json')
        writeToJsonFile(curr_data, '../Models/OpenNRE/data/Wikigender/' + data_type + name_suffix + '.json')
        os.system('chmod 777 ' + '../Models/OpenNRE/data/Wikigender/' + data_type + name_suffix + '.json') # fix weird bug with permissions

def createOpenNREFilesWithArgs(folder_name):
    '''

    :param folder_name: the folder the dataset can be found in
    :return:
    '''
    args = getCommandLineArgs()

    infile_name = args.dataset

    suffix = getNameSuffix(args)

    infile_name += suffix + '.json'

    createOpenNREFile(os.path.join(folder_name, infile_name), suffix)


def createOpenNREFilesFromFolder(folder_name):
    ''''
    :param folder_name: folder with all the files that we want to conver to opennre format
    :return: returns nothing, but creates a folder wiht all the new opennre data in it wiht correct names
    '''
    print('creating opennere files for ' + folder_name)
    for filename in os.listdir(folder_name):
        print('creating opennre file ' + filename)
        createOpenNREFile(os.path.join(folder_name, filename))


if __name__ == '__main__':
    #createOpenNREFilesFromFolder('./JsonData/')
    os.chdir('./WikigenderJsonParsing/')
    createOpenNREFilesWithArgs('./JsonData/')
    os.chdir('../')
    #createOpenNREFile('JsonData/Wikigender.json')

