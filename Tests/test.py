def getEqualizedEntriesThroughDownSampling(entries, male_names, female_names):
    '''
    :param entries: the list of JSON dictionary entries in the dataset; includes entity1, and all entity2,sentences pairs for each attribute
    :param male_names: hashset of male names from Wikipedia
    :param female_names: hashset of female names from wikipedia
    :return: entries, but with equalized gender numbers in train and dev set THROUGH REMOVING DATAPOINTS FROM LESSER CLASS
                here, for example, if there are less male datapoints than female, then we duplicate the male datapoints until there are as many male as female datapoints
                note that we randomly sample to do this duplication, so the order is randomized
    '''
    # get male and female entries
    male_entries = getSpecificEntries(entries, male_names)
    female_entries = getSpecificEntries(entries, female_names)

    # find lesser of the two
    smaller_entries = male_entries
    larger_entries = female_entries
    if(getLenEntries(female_entries) < getLenEntries(male_entries)):
        smaller_entries = female_entries
        larger_entries = male_entries

    # now, randomly sample until the smaller thing has as many datapoints as the larger one
    while(getLenEntries(smaller_entries) <= getLenEntries(larger_entries)):
        entry_index = getRandomArrayIndex(larger_entries)
        if entry_index == None:
            continue
        entry = larger_entries[entry_index]

        # now, choose a random relation
        relations = entry['relations']
        rel_index = getRandomArrayIndex(relations)
        if rel_index == None:
            continue
        relation = relations[rel_index]

        # now, choose random sentence
        sentences = relation['sentences']
        sent_index = getRandomArrayIndex(sentences)
        print('{}, {}, {}'.format(entry_index, rel_index, sent_index))
        print('{}, {}'.format(getLenEntries(smaller_entries), getLenEntries(larger_entries)))
        if sent_index != None:
            entry['relations'][rel_index]['sentences'].pop(sent_index)

        entries[entry_index] = entry

    return entries