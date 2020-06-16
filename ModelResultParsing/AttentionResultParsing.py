'''In this script, we want to find all the attention scores using the regex below'''

import re
import matplotlib.pyplot as plt
import numpy as np
import os
from Utility import readFromJsonFile

#REGEX = '###### Epoch \\d ######'
#REGEX = 'ATTSCOREPERISNTANCE \[((?:\s*\d*.\s*\d*[e-]?\d*\s*)*)\]' #gets all att vectors
#REGEX_GETALL = r'ATTSCOREPERISNTANCE \[((?:\s*\d*.\s*\d*[e-]?\d+\s*)*)\]\nname b\'([\/|a-z|\d|#]*)\'\nsentences:(?:^|\s+)([^@]*)\n[A|e|\n]'
NREDATA_PATH = '../WikigenderJsonParsing/OpenNREData/'
REG_DATA_PATH = os.path.join(NREDATA_PATH, 'train.json')
NA_DATA_PATH = os.path.join(NREDATA_PATH, 'train_NA_NoEq_NoGS_bootstrapped.json')
EGM_DATA_PATH = os.path.join(NREDATA_PATH, 'train_NoNA_Eq_NoGS_bootstrapped.json')
GS_DATA_PATH = os.path.join(NREDATA_PATH, 'train_NoNA_NoEq_GS_NoNS_bootstrapped.json')

REGEX = 'ATTSCOREPERISNTANCE \[((?:\s*\d*.\s*\d*[e-]?\d+\s*)*)\]' # only gets non-empty att vectors



def getAttScoreStringsPerEpoch(infile_name):
    '''

    :param infile_name: input file iwth attention data
    :return: string values represneting attention data (i.e. a list where each element is some string '0.220 0.10 0.110 0.43'
    '''
    with open(infile_name) as infile:
        matches = list()
        for line in infile.readlines():
            matches.extend(re.findall(REGEX, line))


        for i in range(len(matches)):
            print("EPOCH {}".format(i))
            print(matches[i])

    return matches

def getAttScoresPerEpoch(infile_name):
    '''

    :param infile_name: file to read att values from
    :return: a list of att values per epoch. a list of lists, where each element is al ist of floats where list[i][j] is the att num for the jth sentence on the ith epoch
    '''
    att_score_strings = getAttScoreStringsPerEpoch(infile_name)
    att_scores_per_epoch = list()
    for att_score_string in att_score_strings:
        att_scores_per_epoch.append(list())

        scores = att_score_string.split()

        for score in scores:
            att_scores_per_epoch[-1].append(float(score))

    return att_scores_per_epoch


def plotDataForSentence(sent_index, att_scores):
    '''

    :param sent_index:
    :return:
    '''

    data = dict()
    data['epoch'] = list()
    data['scores'] = list()
    for i in range(len(att_scores)):
        print(i)
        epoch = i
        score_vector = att_scores[i]

        data['epoch'].append(i)
        data['scores'].append(score_vector[sent_index])

    plt.plot('epoch', 'scores', data=data)
    plt.show()

    return data

def plotGroupedBarChart(labels, reg_att_scores, gs_att_scores):
    '''

    :param labels:
    :param reg_att_scores: the reg att scores IN ONE EPOCH!!!
    :param gs_att_scores: the gs att scores IN ONE EPOCH!!!  (usually, these should be for the last epoch since that's what we prbably care about)
    :return:
    '''
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, reg_att_scores, width, label='Regular')
    rects2 = ax.bar(x + width / 2, gs_att_scores, width, label='Genderswapped')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Scores by group and gender')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.show()


def getEntity2ID(dataset_name):
    '''

    :param dataset_name: the name of the training set to read from; must be OpenNRE data!
    :return: an entity id --> word (name) mapping from the training set
    '''
    entity2id = dict()

    data = readFromJsonFile(dataset_name)
    for entry in data:
        entity2id[entry['head']['id']] = entry['head']['word']
        entity2id[entry['tail']['id']] = entry['tail']['word']

    return entity2id


def getBags(infile_name, entity2id):
    with open(infile_name) as infile:
        entries = list()
        lines = infile.readlines()
        i = 0
        while i < len(lines):
            line = lines[i]

            if 'ATTSCOREPERISNTANCE' in line:
                entry = dict()
                entry['att_scores'] = list()

                scores = line.split('ATTSCOREPERISNTANCE')[1]
                scores = scores.split('[')[1].split(']')[0]

                for score in scores.split():
                    entry['att_scores'].append(float(score))

                i+=1
                line = lines[i]
                if 'name' in line:
                    names = line.split('#')
                    names[0] = names[0].split('#')[0][7:]
                    del names[-1]

                    #names[0] = entity2id[names[0]]
                    #names[1] = entity2id[names[1]]
                    entry['names'] = names

                i+=1
                line = lines[i]
                if 'sentences' in line:
                    pass
                    #sents = re.split('[b\']|[b\"]', line)
                    #entry['sentences'] = line

                entries.append(entry)




        for i in range(len(entries)):
            print(entries[i])





if __name__ == '__main__':
    e2id = getEntity2ID(GS_DATA_PATH)
    getBags('Attention_Results/gs.txt', e2id)
    '''
    #na_data = getAttScoreStringsPerEpoch('Attention_Results/na.txt')
    reg_data = getAttScoresPerEpoch('Attention_Results/reg.txt')
    gs_data = getAttScoresPerEpoch('Attention_Results/gs.txt')
    plotGroupedBarChart(['s1', 's2', 's3', 's4'], reg_data[-1], gs_data[-1])
    plotDataForSentence(0, reg_data)
    plotDataForSentence(1, gs_data)
    #plotDataForSentence(0, na_data)
    #getAttScoreStrings('Attention_Results/reg.txt')
    '''


