import argparse
import json
import os

TEST_RESULTS_DIRECTORY = './test_results/'
GENDER_DIFFERENCES_RESULTS_DIRECTORY = './gender_differences_results/'
ABSOLUTE_SCORE_RESULTS_DIRECTORY = './absolute_score_results/'
BOOTSTRAPPED_DIR = "./bootstrapped_test_results"
BOOTSTRAPPED_PARSEDRESULTS_DIR = "./bootstrapped_parsed_results"
BOOTSTRAPPED_SHEETS_DIR = './bootstrapped_excel_sheets'

DATASET_NAME = 'Wikigender'

RELATIONS = ['spouse', 'birthdate', 'birthplace', 'hypernym']

RAW_METRICS = ['num_correct', 'num_predicted', 'num_actual']
METRICS = ['f1_score', 'precision', 'recall']
RESULTS_TYPES = ['absolute', 'gender_diffs']

ENCODERS = ['pcnn', 'cnn', 'rnn', 'birnn']
SELECTORS = ['att', 'ave']

TOTAL_INSTANCES_IN_TESTSET_PER_RELATION_FEMALE = 268
TOTAL_INSTANCES_IN_TESTSET_FEMALE = 1072

TOTAL_INSTANCES_IN_TESTSET_PER_RELATION_MALE = 255
TOTAL_INSTANCES_IN_TESTSET_MALE = 1020

BOOTSTRAPPED_SAMPLE_NUMS = [0,1,2,3,4]
#BOOTSTRAPPED_SAMPLE_NUMS = [0,1,2,3,4, 5, 6, 7, 8, 9]
#BOOTSTRAPPED_SAMPLE_NUMS = [5, 6, 7, 8, 9]







def readFromJsonFile(infile_name):
    with open(infile_name, 'r') as infile:
        return json.load(infile)
    return ""
def writeToJsonFile(data, outfile_name, prettify=False):
    with open(outfile_name, 'w') as outfile:
        if(prettify):
            json.dump(data, outfile, indent=4, sort_keys=True)
        else:
            json.dump(data, outfile)


def getTestFiles(dir_name, model_name, test_num=None):
    '''
    :param dir_name:
    :param model_name:
    :param test_num: this is for bootstrapping (bootstrapped samples have a number appended)
    :return: returns (female test file, male test file) tuple
    '''

    def makeResultsCompatible(results, file_path, overwrite_old_file = False, female=True):
        '''

        :param results: some json dict of results such that dict[relation] --> a dictionary of metrics and/or raw metrics
        :param file_name: the name of the file from which we got the results. This is so we can overwrite that file with the new information
        :param overwrite_old_file: boolean. if True, overwrite the old file. Else don't
        :param female: True if this is for female rsults, false if for male results
        :return: the results. Either return the old results (if they're correct, and compatible with the old code) or
        return the new results (created in this functino) if the old results were NOT compatible with new code

         THIS FUNCTION IS FOR ENSURING COMPATIBILITY BETWEEN THE NEW OUTPUT FORMAT AND THE OLD OUTPUT FORMAT
         this is because the result parsing functions are built off the old results format for OpenNRE.
        '''
        new_file = False
        for relation in RELATIONS:
            for raw_metric in RAW_METRICS:
                if raw_metric not in results[relation]:
                    new_file = True
                    break

        if new_file:
            # calculate raw_results metrics
            new_results = dict()
            for relation in RELATIONS:
                if 'total' not in results[relation]:
                    if female:
                        results[relation]['total'] = TOTAL_INSTANCES_IN_TESTSET_FEMALE
                    else:
                        results[relation]['total'] = TOTAL_INSTANCES_IN_TESTSET_MALE
                total_pos_for_relation = results[relation]['total'] # this gives the TOTAL, TRUE NUMBER positives for relation using new results format
                true_pos_for_relation = int(results[relation]['recall'] *  total_pos_for_relation)
                predicted_pos_for_relation = int(total_pos_for_relation/ results[relation]['prec'])

                new_results[relation] = dict()
                new_results[relation]['num_actual'] = total_pos_for_relation
                new_results[relation]['num_correct'] = true_pos_for_relation
                new_results[relation]['num_predicted'] = predicted_pos_for_relation

                new_results[relation]['precision'] = results[relation]['prec']
                new_results[relation]['f1_score'] = results[relation]['f1']
                for metric in METRICS:
                    if metric == 'precision' or metric == 'f1_score':
                        continue
                    else:
                        new_results[relation][metric] = results[relation][metric]


            # write these new results to a file
            if overwrite_old_file:
                writeToJsonFile(new_results, file_path)

            return new_results
        else:
            return results

    file_extension = ".json"

    test_suffix_female = "_FEmale_pred"
    test_suffix_male = "_male_pred"
    if(test_num != None):
        test_suffix_female += "_{}".format(test_num)
        test_suffix_male += "_{}".format(test_num)

    test_suffix_female += file_extension
    test_suffix_male += file_extension

    female_filename = model_name + test_suffix_female
    male_filename = model_name + test_suffix_male

    female_results = readFromJsonFile(os.path.join(dir_name, female_filename))
    male_results = readFromJsonFile(os.path.join(dir_name, male_filename))

    # make sure the results are compatible with the old code (the new results were created with newer code)
    female_results = makeResultsCompatible(female_results, os.path.join(dir_name, female_filename), female=True)
    male_results = makeResultsCompatible(male_results, os.path.join(dir_name, male_filename), female=False)

    return (male_results, female_results)


def getBootstrappedTestFiles(dir_name=BOOTSTRAPPED_PARSEDRESULTS_DIR, model_name='', males_only=False, females_only=False):
    '''
    :param dir_name:
    :param model_name:
    :param test_num: this is for bootstrapping (bootstrapped samples have a number appended)
    :return: returns (female test file, male test file) tuple
    '''
    if not males_only and not females_only:
        abs_scores = readFromJsonFile(os.path.join(dir_name, model_name, 'abs_scores.json'))
        gender_diffs = readFromJsonFile(os.path.join(dir_name, model_name, 'gender_diffs.json'))

    if males_only:
        abs_scores = readFromJsonFile(os.path.join(dir_name, model_name, 'male_abs_scores.json'))
        gender_diffs = None
    elif females_only:
        abs_scores = readFromJsonFile(os.path.join(dir_name, model_name, 'female_abs_scores.json'))
        gender_diffs = None

    return abs_scores, gender_diffs



import argparse

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



def getModelName(args, dataset_name=DATASET_NAME, encoder='pcnn', selector='att'):
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

def getTrueFalseCombos(num):
    items = [True, False]
    return [list(i) for i in itertools.product(items, repeat=num)]


def getWordEmbeddingFileName(args):
    if args.debiased_embeddings:
        return 'debiased_' + 'word_vec_' + getNameSuffix(args) + '.json'
    else:
        return 'word_vec_' + getNameSuffix(args) + '.json'



SINGLE_OPTIONS = [1,2,3,4]
def getNumForModelName(model_name):
    '''
    GS, NA, DE - 8
    GS,NA - 6
    GS,DE - 7
    GS - 4
    NA, DE - 5
    NA - 2
    DE - 3
    NONE - 1
    :param model_name:
    :return:
    '''
    gs = False
    na = False
    de = False
    num = 0
    if '_GS_' in model_name:
        gs = True
        if '_NA_' in model_name:
            na = True
            if '_DE' in model_name:
                num = 8
                de = True
            else:
                num = 6
        else:
            if '_DE' in model_name:
                num = 7
                de = True
            else:
                num = 4
    else:
        if '_NA_' in model_name:
            na = True
            if '_DE' in model_name:
                num = 5
                de = True
            else:
                num = 2
        else:
            if '_DE' in model_name:
                de = True
                num = 3
            else:
                num = 1

    return num













'''

deboptions[0]='-na'
deboptions[1]='-egm'
deboptions[2]='-na -egm'
deboptions[3]='-na -egm -de'
deboptions[4]='-egm -de'
deboptions[5]='-na -de'
deboptions[6]=''
deboptions[7]='-de'
deboptions[8]='-na -gs'
deboptions[9]='-gs'
deboptions[10]='-na -egm -gs -de'
deboptions[11]='-na -egm -gs'
deboptions[12]='-egm -gs'
deboptions[13]='-egm -gs -de'
deboptions[14]='-na -gs -de'
deboptions[15]='-gs -de'
deboptions[16]='-gs -nt'
deboptions[17]='-na -gs -nt'
'''