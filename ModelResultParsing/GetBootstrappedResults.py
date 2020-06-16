import numpy as np
import os
import itertools
import math
import xlwt


from ParseResults import getGenderDifferencesResults
from GetAbsoluteScores import getAbsoluteResults
from Utility import *

MIN = -1000000
MAX = 100000000
CONFIDENCE_INTERVAL = 0.95


def getUpperAndLowerEstimatesForInterval(confidence_interval, metric_results):
    '''
    obtained from https://machinelearningmastery.com/calculate-bootstrap-confidence-intervals-machine-learning-results-python/
    :param confidence_interval: the confidence interval
    :param metric_results: a vector of results for ONE metric (e.g., just for recall)
    :return:
    '''
    p = ((1.0-confidence_interval)/2.0) * 100
    lower = max(0.0, np.percentile(metric_results, p))
    p = (confidence_interval+((1.0-confidence_interval)/2.0)) * 100
    upper = min(1.0, np.percentile(metric_results, p))

    return lower, upper

def updateSumsAndRanges(new_scores, scores_means, scores_ranges):
    '''
    This funciton updates the scores in an object with scores for each relation and each metric

    :param new_scores: This is the object with then ew scores; it should be set up like so: obj[relation][metric] = score.
    :param scores_sums: This object has sums for each metric per relation. Set up like so: obj[relation][metric] = sum of scors thus far encountered
    :param scores_ranges: This object has ranges for each metric per relation. Set up like so: obj[relation][metric] = (max, min) of scors thus far encountered
    :return: the updated sums, ranges objects
    '''
    all_metrics = list(RAW_METRICS)
    all_metrics.extend(METRICS)

    for relation in RELATIONS:
        for metric in all_metrics:
            if metric in new_scores[relation]:

                if relation not in scores_means:
                    scores_means[relation] = dict()
                if metric not in scores_means[relation]:
                    scores_means[relation][metric] = 0
                if relation not in scores_ranges:
                    scores_ranges[relation] = dict()
                if metric not in scores_ranges[relation]:
                    scores_ranges[relation][metric] = [MIN, MAX]

                scores_means[relation][metric] += (float(new_scores[relation][metric]) / len(BOOTSTRAPPED_SAMPLE_NUMS))
                if new_scores[relation][metric] > scores_ranges[relation][metric][0]:
                    scores_ranges[relation][metric][0] = new_scores[relation][metric]
                if new_scores[relation][metric] < scores_ranges[relation][metric][1]:
                    scores_ranges[relation][metric][1] = new_scores[relation][metric]

    return scores_means, scores_ranges


def aggregate_objects(means, ranges, stddevs):
    ret_obj = dict()
    ret_obj['means'] = means
    ret_obj['ranges'] = ranges
    ret_obj['stddevs'] = stddevs

    return ret_obj

relation2num = {'birthplace': 2, 'birthdate':1, 'spouse':3, 'hypernym':4}
def writeAbsAndGenderDiffsToSheet(dict_for_sheet, use_egm_data=False, males_only=False, females_only=False):
    def writeNewResultsSheet(statistic, sheet_name, book, absolute_scores_used = False):
        '''

        :param statistic: means or stddevs
        :param sheet_name: the anme of the sheet
        :param book: the book to which the sheet will be added
        :param absolute_scores_used: if True, use absolute scorse. Else, use gender diffs scores
        :return: adds a asheet to book with name sheet_name that has values we can use for charts
        '''
        results_sheet = book.add_sheet(sheet_name)
        results_sheet.write(0, 1, "No Debiasing Option (Default)")
        results_sheet.write(0, 2, "Name Anonymization")
        results_sheet.write(0, 3, "Debiased Embeddings")
        results_sheet.write(0, 4, "Gender Swapping")

        results_sheet.write(1, 0, 'birthdate')
        results_sheet.write(2, 0, 'birthplace')
        results_sheet.write(3, 0, 'spouse')
        results_sheet.write(4, 0, 'hypernym')

        for model_name in dict_for_sheet:
            if use_egm_data:
                if ('_NoEq_') in model_name:
                    continue
            else:
                if ("_Eq_") in model_name:
                    continue
            model_num = getNumForModelName(model_name)
            if (model_num not in SINGLE_OPTIONS):
                continue
            for relation in RELATIONS:
                try:
                    rel_num = relation2num[relation]

                    # for the abs scores
                    index = 1
                    if absolute_scores_used:
                        index = 0

                    test  = dict_for_sheet[model_name]
                    test2 = test[index]
                    test3 = test2[statistic]
                    test4 = test3[relation]['f1_score']
                    results_sheet.write(rel_num, model_num,
                                        test4)
                except Exception as e:
                    print('Exception writing to sheet: {}'.format(e))

    name_prefix = ''
    if males_only:
        name_prefix = 'male_'
    elif females_only:
        name_prefix = 'female_'


    results_book = xlwt.Workbook()
    writeNewResultsSheet('means', 'Means_GenderDiffs', results_book)
    writeNewResultsSheet('stddevs', 'StdDevs_GenderDiffs', results_book)
    writeNewResultsSheet('means', 'Means_AbsoluteScores', results_book, True)
    writeNewResultsSheet('stddevs', 'StdDevs_AbsoluteScores', results_book, True)

    out_name = name_prefix + 'sheet.xls'
    if use_egm_data:
        out_name = 'equalized_' + out_name
    results_book.save(os.path.join(BOOTSTRAPPED_SHEETS_DIR, out_name))





def writeAbsAndGenderDiffsToFile(absolute_scores, gender_diff_scores, males_only=False, females_only=False):
    if(not absolute_scores['means'] and not gender_diff_scores['means']):
        return #don't print out blank stuff

    name_prefix = ''
    if males_only:
        name_prefix = 'male_'
    elif females_only:
        name_prefix = 'female_'
    abs_file_name = name_prefix + 'abs_scores.json'
    gender_diffs_file_name = name_prefix + 'gender_diffs.json'

    write_dir_path = os.path.join(BOOTSTRAPPED_PARSEDRESULTS_DIR, model_name)
    abs_outfile_path = os.path.join(write_dir_path, abs_file_name)
    gender_diffs_outfile_path = os.path.join(write_dir_path, gender_diffs_file_name)

    if not os.path.exists(write_dir_path):
        os.makedirs(write_dir_path)
    writeToJsonFile(absolute_scores, abs_outfile_path, True)
    writeToJsonFile(gender_diff_scores, gender_diffs_outfile_path, True)

def updateVariances(new_scores, scores_variances, scores_means):
    '''
    Although this function is used to create standard deviations, they are square rooted after this function is used.
    Thus, this function actually calculates variances.
    :param new_scores:
    :param scores_variances:
    :param scores_means:
    :return:
    '''
    all_metrics = list(RAW_METRICS)
    all_metrics.extend(METRICS)

    for relation in RELATIONS:
        for metric in all_metrics:
            if metric in new_scores[relation]:
                if relation not in scores_variances:
                    scores_variances[relation] = dict()
                if metric not in scores_variances[relation]:
                    scores_variances[relation][metric] = 0

                scores_variances[relation][metric] += (float(new_scores[relation][metric] - scores_means[relation][metric])**2) / (len(BOOTSTRAPPED_SAMPLE_NUMS) - 1)

    return scores_variances


def squareRootVariances(variance_scores):
    all_metrics = list(RAW_METRICS)
    all_metrics.extend(METRICS)

    for relation in RELATIONS:
        if relation in variance_scores:
            for metric in all_metrics:
                if metric in variance_scores[relation]:
                    variance_scores[relation][metric] = math.sqrt(variance_scores[relation][metric])

    return variance_scores

def getStandardDeviations(model_name, absolute_scores_means, gender_diff_scores_means, males_only=False, females_only=False):
    absolute_scores_variances = dict()
    gender_diff_scores_variances = dict()

    for i in BOOTSTRAPPED_SAMPLE_NUMS:
        try:
            if not males_only and not females_only:
                male_test_json_results, female_test_json_results = getTestFiles(BOOTSTRAPPED_DIR, model_name, i)
                absolute_scores = getAbsoluteResults(male_test_json_results, female_test_json_results)
                gender_diff_scores = getGenderDifferencesResults(male_test_json_results, female_test_json_results)

                absolute_scores_variances = updateVariances(absolute_scores, absolute_scores_variances,
                                                            absolute_scores_means)
                gender_diff_scores_variances = updateVariances(gender_diff_scores, gender_diff_scores_variances,
                                                               gender_diff_scores_means)
            if males_only:
                male_test_json_results, _ = getTestFiles(BOOTSTRAPPED_DIR, model_name, i)

                absolute_scores_variances = updateVariances(male_test_json_results, absolute_scores_variances,
                                                            absolute_scores_means)
            elif females_only:
                _, female_test_json_results = getTestFiles(BOOTSTRAPPED_DIR, model_name, i)

                absolute_scores_variances = updateVariances(female_test_json_results, absolute_scores_variances,
                                                            absolute_scores_means)

        except Exception as e:
            print(e)
            continue

    absolute_scores_stddevs = squareRootVariances(absolute_scores_variances)
    gender_diff_scores_stddevs = dict()
    if not males_only and not females_only:
        gender_diff_scores_stddevs = squareRootVariances(gender_diff_scores_variances)

    return absolute_scores_stddevs, gender_diff_scores_stddevs

def getMeansAndRanges(model_name, males_only=False, females_only=False):
    '''

    :param model_name:
    :return: the means and the ranges for every relation and metric
    '''
    absolute_scores_means = dict()
    absolute_scores_ranges = dict()

    gender_diff_scores_means = dict()
    gender_diff_scores_ranges = dict()
    for i in BOOTSTRAPPED_SAMPLE_NUMS:
        try:
            if not males_only and not females_only:
                male_test_json_results, female_test_json_results = getTestFiles(BOOTSTRAPPED_DIR, model_name, i)
                absolute_scores = getAbsoluteResults(male_test_json_results, female_test_json_results)
                gender_diff_scores = getGenderDifferencesResults(male_test_json_results, female_test_json_results)

                absolute_scores_means, absolute_scores_ranges = updateSumsAndRanges(absolute_scores, absolute_scores_means,
                                                                                    absolute_scores_ranges)
                gender_diff_scores_sums, gender_diff_scores_ranges = updateSumsAndRanges(gender_diff_scores,
                                                                                         gender_diff_scores_means,
                                                                                         gender_diff_scores_ranges)
            if males_only:
                male_test_json_results, _ = getTestFiles(BOOTSTRAPPED_DIR, model_name, i)

                absolute_scores_means, absolute_scores_ranges = updateSumsAndRanges(male_test_json_results,
                                                                                    absolute_scores_means,
                                                                                    absolute_scores_ranges)
            elif females_only:
                _, female_test_json_results = getTestFiles(BOOTSTRAPPED_DIR, model_name, i)

                absolute_scores_means, absolute_scores_ranges = updateSumsAndRanges(female_test_json_results,
                                                                                    absolute_scores_means,
                                                                                    absolute_scores_ranges)
        except Exception as e:
            print(e)
            continue
    return absolute_scores_means, absolute_scores_ranges, gender_diff_scores_means, gender_diff_scores_ranges

def getBootstrappedMetricScores(model_name, males_only=False, females_only=False):
    # first, get means and ranges
    absolute_scores_means, absolute_scores_ranges, gender_diff_scores_means, gender_diff_scores_ranges = getMeansAndRanges(model_name, males_only=males_only, females_only=females_only)

    # now, we get the standard deviations
    absolute_scores_stddevs, gender_diff_scores_stddevs = getStandardDeviations(model_name, absolute_scores_means, gender_diff_scores_means, males_only=males_only, females_only=females_only)

    # aggregate objects
    absolute_scores = aggregate_objects(absolute_scores_means, absolute_scores_ranges, absolute_scores_stddevs)
    gender_diff_scores = aggregate_objects(gender_diff_scores_means, gender_diff_scores_ranges, gender_diff_scores_stddevs)

    return absolute_scores, gender_diff_scores





if __name__ == '__main__':
    items = [True, False]
    true_false_model_combos = [list(i) for i in itertools.product(items, repeat=4)]

    gender_combos = [list(i) for i in itertools.product(items, repeat=2)]

    dict_for_sheet = dict()


    for encoder in ENCODERS:
        for selector in SELECTORS:
            for combo in true_false_model_combos:
                try:
                    #model_name = getModelNameSimulateArgs(equalized_gender_mentions=combo[0], gender_swapped=combo[1], name_anonymized=combo[2], swap_names=False, debiased_embeddings=combo[3], neutralize=combo[4], boostrapped=True, encoder=encoder, selector=selector)
                    model_name = getModelNameSimulateArgs(equalized_gender_mentions=combo[0], gender_swapped=combo[1], name_anonymized=combo[2], swap_names=False, debiased_embeddings=combo[3], boostrapped=True, encoder=encoder, selector=selector)
                    for gender_combo in gender_combos:
                        males_only = gender_combo[0]
                        females_only = gender_combo[1]
                        if males_only == females_only == True:
                            continue

                        absolute_scores, gender_diff_scores = getBootstrappedMetricScores(model_name, males_only=males_only, females_only=females_only)

                        # for writing to sheet
                        dict_for_sheet[model_name] = (absolute_scores, gender_diff_scores)

                        #now, write these to files
                        writeAbsAndGenderDiffsToFile(absolute_scores, gender_diff_scores, males_only=males_only, females_only=females_only)



                except FileNotFoundError as e:
                    print(e)
                    continue


    writeAbsAndGenderDiffsToSheet(dict_for_sheet)
    writeAbsAndGenderDiffsToSheet(dict_for_sheet, use_egm_data=True)



'''
for _, _, files in os.walk(BOOTSTRAPPED_DIR):
    pass
    '''
'''
for _, dirs, _ in os.walk(BOOTSTRAPPED_DIR):
    for dir in dirs:
        for _, _, files in os.walk(dir):
            for file in files:
                pass #here, get the gender_differences, abs score results; then, get the percentiles and that's what we want NO -- get the ranges and use that as +/- value

'''