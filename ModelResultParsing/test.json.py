from Utility import *
from GetAbsoluteScores import getAbsoluteResults
from ParseResults import getGenderDifferencesResults

if __name__ == '__main__':
    male_results, female_results = getTestFiles(BOOTSTRAPPED_DIR, 'Wikigender_pcnn_att__NoNA_NoEq_GS_NoNSNT_bootstrapped_NoDE', 0)
    abs_results = getAbsoluteResults(male_results, female_results)
    gender_diff_results = getGenderDifferencesResults(male_results, female_results)

    print(abs_results)

    print(gender_diff_results)
