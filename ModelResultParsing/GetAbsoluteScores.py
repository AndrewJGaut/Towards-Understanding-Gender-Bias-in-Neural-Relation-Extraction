from Utility import *
import itertools

def calculate_f1_score(precision, recall):
    return 2 * precision * recall / (precision + recall)

def getAbsoluteResults(male_test_json_results, female_test_json_results):
    absolute_scores = dict()
    for relation in RELATIONS:
        absolute_scores[relation] = dict()
        for raw_metric in RAW_METRICS:
            absolute_scores[relation][raw_metric] = male_test_json_results[relation][raw_metric] + \
                                                    female_test_json_results[relation][raw_metric]

        absolute_scores[relation]['precision'] = absolute_scores[relation]['num_correct'] / absolute_scores[relation][
            'num_predicted']
        absolute_scores[relation]['recall'] = absolute_scores[relation]['num_correct'] / absolute_scores[relation][
            'num_actual']
        absolute_scores[relation]['f1_score'] = calculate_f1_score(absolute_scores[relation]['precision'],
                                                                   absolute_scores[relation]['recall'])

    return absolute_scores

if __name__ == '__main__':
    items = [True, False]
    true_false_combos = [list(i) for i in itertools.product(items, repeat=4)]



    for combo in true_false_combos:
        try:
            model_name = getModelNameSimulateArgs(combo[0], combo[1], combo[2], combo[3])

            male_test_json_results, female_test_json_results = getTestFiles(TEST_RESULTS_DIRECTORY, model_name)
            absolute_scores = getAbsoluteResults(male_test_json_results, female_test_json_results)

            # write the differences to a file
            outfile_path = os.path.join(ABSOLUTE_SCORE_RESULTS_DIRECTORY, model_name + "abs_scores.json")
            writeToJsonFile(absolute_scores, outfile_path, True)


        except FileNotFoundError as e:
            print(e)
            continue