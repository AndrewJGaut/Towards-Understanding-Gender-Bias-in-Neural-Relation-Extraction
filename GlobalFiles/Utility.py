
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



def getModelName(args, dataset_name, encoder, selector):
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



def getWordEmbeddingFileName(args):
    if args.debiased_embeddings:
        return 'debiased_' + 'word_vec_' + getNameSuffix(args) + '.json'
    else:
        return 'word_vec_' + getNameSuffix(args) + '.json'


