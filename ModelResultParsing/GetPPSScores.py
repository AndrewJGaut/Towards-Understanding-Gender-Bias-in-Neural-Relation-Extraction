from __future__ import division
import itertools
import random
from Utility import *

# from Utility import *

# # for the plotting
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 2 groups: M, F
NUM_GROUPS = 2


def calc_fbeta_score(num_correct, num_predicted, num_actual, beta):
    precision = num_correct / num_predicted
    recall = num_correct / num_actual

    beta_squared = beta * beta

    return (1 + beta_squared) * ((precision * recall) / ((beta_squared * precision) + recall))


def getRawMetricAbsoluteScores(male_results, female_results):
    raw_metric_abs_scores = dict()
    for relation in RELATIONS:
        raw_metric_abs_scores[relation] = dict()

        for raw_metric in RAW_METRICS:
            if raw_metric in raw_metric_abs_scores[relation]:
                raw_metric_abs_scores[relation][raw_metric]['total'] += male_results[relation][raw_metric] + \
                                                                        female_results[relation][raw_metric]
                raw_metric_abs_scores[relation][raw_metric]['male'] += male_results[relation][raw_metric]
                raw_metric_abs_scores[relation][raw_metric]['female'] += female_results[relation][raw_metric]
            else:
                raw_metric_abs_scores[relation][raw_metric] = dict()

                raw_metric_abs_scores[relation][raw_metric]['total'] = male_results[relation][raw_metric] + \
                                                                       female_results[relation][raw_metric]
                raw_metric_abs_scores[relation][raw_metric]['male'] = male_results[relation][raw_metric]
                raw_metric_abs_scores[relation][raw_metric]['female'] = female_results[relation][raw_metric]

    return raw_metric_abs_scores


def get_raw_results(model_name, dir=TEST_RESULTS_DIRECTORY, bootstrapping=True):
    '''get the desired raw results (num_correct, num_actual, num_predicted for each relation and male,
    female predictions for each relation'''
    male_results = None
    female_results = None
    if bootstrapping:
        # then, we need to get the results differently
        male_results, _ = getBootstrappedTestFiles(model_name=model_name,
                                                   males_only=True)
        female_results, _ = getBootstrappedTestFiles(model_name=model_name,
                                                     females_only=True)

        male_results = male_results['means']
        female_results = female_results['means']
    else:
        male_results, female_results = getTestFiles(dir, model_name)

    if male_results == None or female_results == None:
        raise Exception("Results not obtained correctly")
    raw_metric_abs_scores = getRawMetricAbsoluteScores(male_results, female_results)

    return raw_metric_abs_scores


def calculate_scores(raw_results, alpha, beta=1):
    # get the sum of these over the relations
    disparity_score_sum = 0
    f_score_sum = 0
    male_scores, female_scores = [], []
    for relation in RELATIONS:
        male_fbeta = calc_fbeta_score(raw_results[relation]['num_correct']['male'],
                                      raw_results[relation]['num_predicted']['male'],
                                      raw_results[relation]['num_actual']['male'],
                                      beta)
        female_fbeta = calc_fbeta_score(raw_results[relation]['num_correct']['female'],
                                        raw_results[relation]['num_predicted']['female'],
                                        raw_results[relation]['num_actual']['female'],
                                        beta)

        male_scores.append(male_fbeta)
        female_scores.append(female_fbeta)

        # we're summing these over all the relations!
        diff = abs(male_fbeta - female_fbeta)

        disparity_score_sum += diff

        f_score_sum += (calc_fbeta_score(raw_results[relation]['num_correct']['total'],
                                        raw_results[relation]['num_predicted']['total'],
                                        raw_results[relation]['num_actual']['total'],
                                        beta))

    f_score = f_score_sum / len(RELATIONS)
    disparity_score = disparity_score_sum / len(RELATIONS)
    pps_score = f_score - disparity_score

    print("alpha = ", alpha, ". beta = ", beta)
    print("pps score: ", pps_score)
    print("f score: ", f_score)
    print("parity score: ", disparity_score)
    print("\n---\n")

    return pps_score, f_score, disparity_score


def get_pps_scores(raw_results, alpha_vals, beta_vals):
    '''
    this function obtains the pps scores for a LIST of beta values and alpha values
    :param male_results: 
    :param female_results: 
    :param beta_vals: a list giving beta values
    :param alpha_vals: a list giving alpha values
    :return: A matrix of pps scores where pps[i][j] corresponds to beta_vals[j] and alpha_vals[i], a dictionary which can be written toa  file giving [absolute][beta_val][alpha_val] = absolute_val or [parity][beta_val]{alhpa_val} = parity_val
    '''

    pps_scores = list()
    pps_dict = dict()

    for alpha_index in range(len(alpha_vals)):
        alpha = alpha_vals[alpha_index]

        # update structures for this round of alpha vals
        pps_scores.append(list())
        pps_dict['alpha={}'.format(alpha)] = dict()

        for beta_index in range(len(beta_vals)):
            beta = beta_vals[beta_index]

            # update dict for this round of beta vals
            pps_dict['alpha={}'.format(alpha)]['beta={}'.format(beta)] = dict()

            pps_score, f_score, parity_score = calculate_scores(raw_results, alpha, beta)

            # add to the data structures
            pps_scores[alpha_index].append(pps_score)
            pps_dict['alpha={}'.format(alpha)]['beta={}'.format(beta)]['f_score'] = f_score
            pps_dict['alpha={}'.format(alpha)]['beta={}'.format(beta)]['parity_score'] = parity_score
            pps_dict['alpha={}'.format(alpha)]['beta={}'.format(beta)]['pps_score'] = pps_score

    return pps_scores, pps_dict


def get_pps_scores_for_model(combo, alpha_vals, beta_vals, bootstrap=True):
    '''

    :param combo: gives the debiasing configuration of the model so we know what its name is
    :param alpha_vals: alpha values for pps score
    :param beta_vals: beta values for pps score
    :return: pps scores for this model
    '''
    # get results files for this model name
    model_name = getModelNameSimulateArgs(gender_swapped=combo[0], name_anonymized=combo[1], equalized_gender_mentions=combo[2], debiased_embeddings=combo[3], boostrapped=bootstrap)
    raw_results = get_raw_results(model_name)

    # get the pps results
    pps_scores, pps_dict = get_pps_scores(raw_results, alpha_vals, beta_vals)

    # write the dict object to a file
    # outfile_path = os.path.join('pps_scores', model_name + "pps_scores.json")
    # writeToJsonFile(pps_dict, outfile_path, True)

    return pps_scores


# def plot_3d(alpha_vals, beta_vals):
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#
#     # Make data.
#     alpha_matrix, beta_matrix = np.meshgrid(alpha_vals, beta_vals, indexing='ij')
#
#     model_name = getModelNameSimulateArgs(False, True, False, False)
#     raw_results = get_raw_results(model_name)
#     Z = []
#     for alpha in alpha_vals:
#         temp_array = []
#         for beta in beta_vals:
#             pps_score, f_score, parity_score = calculate_scores(raw_results, alpha, beta)
#             temp_array.append(pps_score)
#         Z.append(temp_array)
#
#     # print(Z)
#
#     Z = np.asmatrix(Z)
#
#     # Plot the surface.
#     surf = ax.plot_surface(alpha_matrix, beta_matrix, Z, cmap=cm.coolwarm,
#                            linewidth=0, antialiased=False)
#
#     # Customize the z axis.
#     ax.set_zlim(-1.01, 1.01)
#     ax.zaxis.set_major_locator(LinearLocator(10))
#     ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#
#     # Add a color bar which maps values to colors.
#     fig.colorbar(surf, shrink=0.5, aspect=5)
#
#     plt.show()


if __name__ == '__main__':
    items = [True, False]
    true_false_combos = [list(i) for i in itertools.product(items, repeat=4)]

    for combo in true_false_combos:
        try:
            print(get_pps_scores_for_model(combo, [0.5], [1]))
        except IOError as e:
            print(e)
            continue

    # 3D plot the values for the PPS scores
    alpha_vals = np.arange(0,1.10,0.1) # get alpha values with step value of 0.01
    beta_vals = np.arange(0,2,0.2)
    # equalized gender mentions, gender swapping, name anonymization, debiased embeddings
    # pps_scores = get_pps_scores_for_model([False, True, False, False], alpha_vals, beta_vals)
    # z_vals = np.asarray(pps_scores)
    #plot_3d(alpha_vals, beta_vals)
