import argparse
import os
import json
import itertools
import sys

from Utility import *


def getGenderDifferencesResults(male_results, female_results):
    '''

    :param male_results:
    :param female_results:
    :return:
    '''
    gender_differences_results = dict()

    for relation in RELATIONS:
        gender_differences_results[relation] = dict()
        for metric in METRICS:
            gender_differences_results[relation][metric] = male_results[relation][metric] - female_results[relation][metric]

    return gender_differences_results



def createGenderDifferencesResultsFile(equalized_gender_mentions=False, gender_swapped=False, name_anonymized=False, debiased_embeddings=False, encoder='pcnn', selector='att'):
    '''

    :param egm:
    :param gs:
    :param na:
    :param de:
    :return:
    '''
    # mimic parser from OpenNRE code so we can directly use functions from that file
    # we can just make adjustments by assigning values
    # e.g. args.gender_swap = False
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', nargs='?', default='Wikigender')
    parser.add_argument('--encoder', nargs='?', default='pcnn')
    parser.add_argument('--selector', nargs='?', default='att')
    parser.add_argument("--gender_swap", "-gs", action="store_true")
    parser.add_argument("--equalized_gender_mentions", "-egm", action="store_true")
    parser.add_argument("--swap_names", "-sn", action="store_true")
    parser.add_argument("--name_anonymize", "-na", action="store_true")
    parser.add_argument("--debiased_embeddings", "-de", action="store_true")
    args = parser.parse_args()

    args.name_anonymize = name_anonymized
    args.gender_swap = gender_swapped
    args.equalized_gender_mentions = equalized_gender_mentions
    args.debiased_embeddings = debiased_embeddings
    args.encoder = encoder
    args.selector = selector

    # get the model name
    model_name = getModelName(args, DATASET_NAME, args.encoder, args.selector)

    # get the male and female results
    male_json_results, female_json_results = getTestFiles(TEST_RESULTS_DIRECTORY, model_name)

    # get the differences in those results
    gender_differences_json_results = getGenderDifferencesResults(male_json_results, female_json_results)

    # write the differences to a file
    outfile_path = os.path.join(GENDER_DIFFERENCES_RESULTS_DIRECTORY, model_name + "gender_differences.json")
    writeToJsonFile(gender_differences_json_results, outfile_path, True)




if __name__ == '__main__':
    items = [True, False]
    true_false_combos = [list(i) for i in itertools.product(items,repeat=4)]

    for combo in true_false_combos:
        for encoder in ENCODERS:
            for selector in SELECTORS:
                try:
                    createGenderDifferencesResultsFile(combo[0], combo[1], combo[2], combo[3], encoder, selector)
                except FileNotFoundError as e:
                    print(e)
                    continue
