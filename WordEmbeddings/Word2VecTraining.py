import string
import gensim
from gensim.models import Word2Vec
import nltk
from Utility import *
import os
from DebiasEmbeddings import debiasEmbedding
import argparse
import itertools

DATASET_NAME = 'JsonData/Wikigender'




def getAllSentences(data):
    '''
    :param data: data in the Wikigender dataset json format
    :return: all the sentences in this Wikigender data
    '''

    entries = getEntriesForDataType(data, 'train')

    sentences = list()
    for entry in entries:
        #print(entry['entity1'])
        for relation in entry['relations']:
            for sentence in relation['sentences']:
                sentence = sentence.translate(str.maketrans('', '', string.punctuation))

                sentences.append(nltk.word_tokenize(sentence))
    return sentences


'''
Precondition:
    vec_file is a word vector file of the form word num1 num2 ... numn \n word2 num1 ...etc
Postcondition:
    formats vector file so it works for OpenNRE
'''
'''
def formatWordVectorFile(model):
    formatted_word_vecs_string = "[\n"

    for word in model.wv.vocab:
        print(str(word))
        formatted_word_vecs_string += "\t{\"word\": \"" + str(word) + "\",\"vec\": ["
        for item in model[word]:
            formatted_word_vecs_string += str(item) + ", "
        formatted_word_vecs_string = formatted_word_vecs_string[0:-2]
        formatted_word_vecs_string += "]},\n"

    formatted_word_vecs_string = formatted_word_vecs_string[0:-2]
    formatted_word_vecs_string += "}\n]"

    out_file = open('word_vec.json', 'w')
    out_file.write(formatted_word_vecs_string)
    out_file.close()
'''

def formatWordVectors(model, outfile_name):
    '''
    :param vec_file: word2vec format
    :return: convert word2vec formatted file into opennre's word embedding format
    '''
    word_vec_json_data = list()
    for word in model.wv.vocab:
        word_vec_json_data.append(dict())
        word_vec_json_data[-1]['word'] = word
        word_vec_json_data[-1]['vec'] = list()

        for item in model.wv[word]:
            word_vec_json_data[-1]['vec'].append(item.item())


    writeToJsonFile(word_vec_json_data, outfile_name)

def formatWordVectorFile(wordvec_file, outfile_name):
    '''

    :param wordvec_file:
    :param outfile_name:
    :return:  same as above function but read the vectors from a file rather than the actual model
    '''

    file = open(wordvec_file, 'r')
    word_vec_json_data = list()
    for line in file.readlines():
        word_vec_json_data.append(dict())

        items=line.split()
        word_vec_json_data[-1]['word'] = items[0]
        word_vec_json_data[-1]['vec'] = list()

        for item in items[1:]:
            word_vec_json_data[-1]['vec'].append(item)

    writeToJsonFile(word_vec_json_data, outfile_name)


def preprocessDebiasedFile(str1):
    '''

    :param str1:
    :return: this funciton appears to change the dewbiased embedding file into the format we're expecting for a word vector file to be in (i.e. the word2vec default format)
     this is so we can use the converttoopennre format function
    '''
    #wordvec = open(file_name, 'r')
    new_str = ""
    counter = 0
    for vec in str1.split(']'):
        #print(str(counter))
        counter += 1
        vec = vec.replace('\n', ' ')
        curr_vec = vec.replace('[', ' ')
        #words = nltk.word_tokenize(vec)
        #curr_vec = ' '.join(words[1:])
        new_str = '\n'.join([new_str,curr_vec])

    out_file = open('debiased_prepr.txt', 'w')
    out_file.write(new_str)
    return new_str

def convertHardDebiasedEmbeddingFileToOpenNREFormat(file_name):
    '''

    :param file_name: the debiased embedding file name (e.g. wordvec_debiased.txt)
    :return: creates a file called debiased_prepr_formatted.txt
    '''
    wordvecstr = open(file_name, 'r').read()
    x = preprocessDebiasedFile(wordvecstr)
    #print(x)
    formatWordVectorFile('debiased_prepr.txt')


def createArgsString(args):
    args_string = ""
    if(args.name_anonymize):
        args_string += '-na '
    if(args.gender_swap):
        args_string += '-gs '
    if(args.equalized_gender_mentions):
        args_string += '-egm '
    if(args.swap_names):
        args_string += '-sn '
    if(args.bootstrapped):
        args_string += '-bs '
    if(args.neutralize):
        args_string += "-nt"

    return args_string






def trainWord2VecEmbedding(dataset_name, suffix, args_for_debias_script="", boostrapping=False):
    '''

    :param dataset_name: name of dataset to train iton
    :param args_for_debias_script: the arguments for the debiasing script (if there are any)
    :param boostrapping: true if we're using boostrapping
    :return: creates word_vec_formatted.txt, a file with opennre formatted embeddings
    '''
    # get file names
    wordvec_file_name = getWordEmbeddingFileName(suffix, '.txt')
    wordvec_json_file_name = getWordEmbeddingFileName(suffix, '.json')
    wordvec_path = os.path.join('embeddings', wordvec_file_name)
    wordvec_json_path = os.path.join('embeddings', wordvec_json_file_name)
    debiased_file_path = os.path.join('embeddings', 'debiased_' + wordvec_file_name)
    debiased_json_path = os.path.join('embeddings', 'debiased_' + wordvec_json_file_name)
    opennre_data_file_path = '../Models/OpenNRE/data/Wikigender/'

    model = None
    try:
        if(boostrapping or not os.path.exists(wordvec_path)):
            # get sentences
            data = readFromJsonFile(dataset_name)
            sentences = getAllSentences(data)

            # train model
            model = Word2Vec(sentences, min_count=1)

            # save model
            model.wv.save_word2vec_format(os.path.join('embeddings', wordvec_file_name))

            # we need to do this to remove the line that word2vec adds that debiaswe doesn't want
            os.chdir('./embeddings/')
            os.system('sed -i.bak -e \'1d\' ' + wordvec_file_name + ' ; rm *.bak')
            os.chdir('../')
        if(boostrapping or not os.path.exists(wordvec_json_path) and os.path.exists(wordvec_path)):
            formatWordVectorFile(wordvec_path, wordvec_json_path)
        if (boostrapping or not os.path.exists(debiased_file_path) and os.path.exists(wordvec_path)):
            os.system('python3 DebiasEmbeddings.py ' + args_for_debias_script) # create those debiased embeddings
        if(boostrapping or not os.path.exists(debiased_json_path)):
            formatWordVectorFile(debiased_file_path, debiased_json_path)

        # copy the files over to the model
        os.system('cp ' + wordvec_json_path  + ' ' + opennre_data_file_path)
        os.system('cp ' + debiased_json_path + ' ' + opennre_data_file_path)
    except Exception as e:
        print(e)

    return model

'''
def formatWord2VecEmbedding(model, debiased_model, dataset_name, suffix):
    :param model: the word2vec model that you want formatted for opennre
    :param dataset_name:
    :param suffix: name suffix
    :return: creates file with opennre formatted word vectos for model
    wordvec_file_name = getWordEmbeddingFileName(suffix, '.txt')
    wordvec_json_file_name = getWordEmbeddingFileName(suffix, '.json')
    wordvec_path = os.path.join('embeddings', wordvec_file_name)
    wordvec_json_path = os.path.join('embeddings', wordvec_json_file_name)
    debiased_file_path = os.path.join('embeddings', 'debiased_' + wordvec_file_name)
    debiased_json_path = os.path.join('embeddings', 'debiased_' + wordvec_json_file_name)

    if model == None:
        if(os.path.exists(wordvec_path)):
            Word2Vec.load_word2vec_format(wordvec_path)
        else:
            return


    if (not os.path.exists(wordvec_json_path)):
        # convert to OpenNRE Format
'''


def trainWord2VecEmbeddings():
    true_false_combos = getTrueFalseCombos(2)

    for combo in true_false_combos:
        print('training')
        suffix =  getNameSuffixSimulateArgs(equalized_gender_mentions=False,
                                                                name_anonymized=combo[0],
                                                                gender_swapped=combo[1], swap_names=False)
        dataset_name = DATASET_NAME + suffix + '.json'

        model = trainWord2VecEmbedding(dataset_name, suffix)

def trainWord2VecEmbeddingsWithArgs():
    args = getCommandLineArgs()

    suffix = getNameSuffix(args)
    dataset_name = DATASET_NAME + suffix + '.json'

    model = trainWord2VecEmbedding(dataset_name, suffix, createArgsString(args), args.bootstrapped)





if __name__=='__main__':
    #trainWord2VecEmbeddings()
    os.chdir('./WordEmbeddings')
    trainWord2VecEmbeddingsWithArgs()
    os.chdir('../')
