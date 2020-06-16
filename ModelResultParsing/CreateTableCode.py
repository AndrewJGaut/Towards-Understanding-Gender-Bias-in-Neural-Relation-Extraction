from ModelResultParsing.Utility import *
from GetPPSScores import get_pps_scores, get_raw_results
import itertools

NUM_DECIMAL_PLACES=3


OLD_RESULTS_DIRECTORY = 'old_results'
FULL_DIR_PATH = os.path.join('./test_results/', OLD_RESULTS_DIRECTORY)
METRICS_IN_TABLE = ['f1', 'recall', 'pps']

def getTrueFalseCombos(num):
    items = [True, False]
    return [list(i) for i in itertools.product(items, repeat=num)]

def format_nums(mean, stdev=None, pm=False):
    if pm:
        return '{} $\pm$ {}'.format(format_num(mean), format_num(stdev))
    else:
        return '{}'.format(format_num(mean))

def format_num(num):
    '''

    :param num: the number to format
    :return: the number rounded to 3 decimal places and printed as a string
    '''
    num_str = "%.3f" % float(num)
    num_str_leading_zero_removed = '.' + num_str.split('.')[-1]
    if num < 0:
        num_str_leading_zero_removed = '-' + num_str_leading_zero_removed
    return num_str_leading_zero_removed

def checkmark(bool):
    if(bool):
        return "\checkmark"
    else:
        return " "

def getPPSScoreTable(egm=False):
    '''
       Get code for latex table automatically
       :param abs_scores: TRue if you want the scores to be the absolute scores / performance
       :param gender_diffs: true if you want the scores to reflect gap between genders
       :return:
       '''
    table_strs = dict()
    combos = getTrueFalseCombos(3)
    for encoder in ENCODERS:
        if encoder != 'pcnn':
            continue
        for selector in SELECTORS:
            if selector != 'att':
                continue
            for combo in combos:
                # print("{}, {}".format(encoder, selector))
                model_name = getModelNameSimulateArgs(equalized_gender_mentions=egm, boostrapped=True, encoder=encoder,
                                                      selector=selector, gender_swapped=combo[0], name_anonymized=combo[1], debiased_embeddings=combo[2])

                gs = False
                na = False
                de = False
                if'_GS_' in model_name:
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
                try:
                    pps, pps_dict = get_pps_scores(get_raw_results(model_name, bootstrapping=True), [0.5], [1])
                    print(pps_dict)
                    pps_dict = pps_dict['alpha=0.5']['beta=1']
                    # ret_str += "& {}".format(format_num(pps[0][0]))
                    table_strs[num] = "{} & {} & {} & {} & {} & {} & {} \\\\".format(num, checkmark(na), checkmark(de), checkmark(gs), format_num(pps_dict['pps_score']), format_num(pps_dict['f_score']), format_num(pps_dict['parity_score']))
                    #ret_str += "\\\\ \n \\hline \n"
                except Exception as e:
                    print(e)
                    continue

    for i in range(1, 9):
        print(table_strs[i])
        if i==1 or i==4 or i==7 or i==8:
            print("\\hline \n")
    #print(ret_str)


def getTableCodeForEncoderSelectorPairs(abs_scores_used=False, gender_diffs_used=True, egm=False):
    '''
    Get code for latex table automatically
    :param abs_scores: TRue if you want the scores to be the absolute scores / performance
    :param gender_diffs: true if you want the scores to reflect gap between genders
    :return:
    '''
    ret_str = ""
    for selector in SELECTORS:
        for encoder in ENCODERS:
            # print("{}, {}".format(encoder, selector))
            model_name = getModelNameSimulateArgs(equalized_gender_mentions=egm, boostrapped=True, encoder=encoder, selector=selector)
            try:
                abs_scores, gender_diffs = getBootstrappedTestFiles(BOOTSTRAPPED_PARSEDRESULTS_DIR, model_name)
                scores_to_use = gender_diffs
                if abs_scores_used:
                    scores_to_use = abs_scores

                ret_str += encoder.upper() + "," + selector.upper() + " "

                for relation in RELATIONS:
                    f1_mean = scores_to_use['means'][relation]['f1_score']
                    f1_stdev = scores_to_use['stddevs'][relation]['f1_score']

                    recall_mean = scores_to_use['means'][relation]['recall']
                    recall_stdev = scores_to_use['stddevs'][relation]['recall']



                    # male_result = round(male_file[relation]['precision'], NUM_DECIMAL_PLACES)
                    # female_result = round(female_file[relation]['precision'], NUM_DECIMAL_PLACES)
                    ret_str += "& {} & {}".format(format_nums(f1_mean, f1_stdev), format_nums(recall_mean, recall_stdev))
                pps, pps_dict = get_pps_scores(get_raw_results(model_name, bootstrapping=True), [0.5], [1])
                parity = pps_dict['alpha=0.5']['beta=1']['parity_score']
                f1 = pps_dict['alpha=0.5']['beta=1']['f_score']
                ret_str += "& {} & {} & {}".format(format_num(f1), format_num(parity), format_num(pps[0][0]))
                ret_str += "\\\\ \n"
                #ret_str += "\\\\ \n \\hline \n"
            except Exception as e:
                print(e)
                continue
        ret_str += "\\hline \n"

    print(ret_str)


if __name__ == '__main__':
    getPPSScoreTable(egm=False)
    #getTableCodeForEncoderSelectorPairs(egm=False)
    #getPPSScoreTable(egm=False)
    #getTableCodeForEncoderSelectorPairs(egm=True)
    #getTableCodeForEncoderSelectorPairs(egm=True)
    #getPPSScoreTable()
