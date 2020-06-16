

import os

print(os.path.exists('./testmatrix.py'))
print(os.path.exists('./testmatrix2.py'))

def getModelName(args, dataset_name, encoder, selector):
    name_suffix = getNameSuffix(args)
    if args.debiased_embeddings:
        name_suffix += "_DE"
    else:
        name_suffix += "_NoDE"

    return dataset_name + "_" + encoder + "_" + selector + name_suffix

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
    else:
        name_suffix += "_NoGS"

    return "_" + name_suffix
