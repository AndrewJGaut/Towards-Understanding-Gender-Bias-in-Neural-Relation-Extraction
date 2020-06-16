from Utility import *
from WikigenderJsonParsing.Utility import *
import copy


def removeDuplicates():
    data = readFromJsonFile('JsonData/Wikigender.json')

    new_dataset = dict()

    for type in DataTypes:
        data_split = data[type]

        seen_before = set()

        new_dataset[type] = list()

        for entry in data_split:
            name = entry['entity1']
            if name in seen_before:
                continue
            else:
                seen_before.add(name)
                new_dataset[type].append(copy.deepcopy(entry))


    writeToJsonFile(new_dataset, 'JsonData/new_Wikigender.json')



def oldcountInstances(entries):
    '''

    :param entries: the entries you're looking at
    if relation != None, then we want to only count instances for that relation.
    :return: the total number of sentences we have
    '''
    instance_count = 0
    for entry in entries:
        try:
            for relation in entry['relations']:
                for sentence in relation['sentences']:
                    instance_count += 1
        except Exception as e:
            print(e)

    return instance_count


def countInstances(entries):
    '''

    :param entries: the entries you're looking at
    if relation != None, then we want to only count instances for that relation.
    :return: the total number of sentences we have
    '''
    instance_count = 0
    seen_before = set()
    for entry in entries:
        name = entry['entity1']
        if name in seen_before:
            x = 5
        else:
            seen_before.add(name)

        try:
            for relation in entry['relations']:
                for sentence in relation['sentences']:
                    instance_count += 1
        except Exception as e:
            print(e)

    return instance_count

def countEntityPairs(entries):
    entpair_set = set()
    for entry in entries:
        e1 = entry['entity1']
        for relation in entry['relations']:
            if len(relation['sentences']) > 0:
                e2 = relation['entity2']
                if (e1, e2) not in entpair_set:
                    entpair_set.add((e1,e2))

    return len(entpair_set)

def printOutMaleFemaleDatasetNumbers():
    # get the male and female names
    male_names, female_names = getMaleAndFemaleNames()

    # get the dataset
    #data = readFromJsonFile('JsonData/Wikigender_NoNA_Eq_NoGS.json')
    data = readFromJsonFile('JsonData/new_Wikigender.json')
    equalized_data = readFromJsonFile('JsonData/Wikigender_NoNA_Eq_NoGS.json')
    ret_str = ""
    for type in DataTypes:
        data_split = data[type]

        male_entries = getSpecificEntries(data_split, male_names)
        female_entries = getSpecificEntries(data_split, female_names)

        ret_str += "{} & {} & {} & {} & {}".format(type, countEntityPairs(male_entries), countEntityPairs(female_entries),
                                              countInstances(male_entries), countInstances(female_entries))
        data_split = equalized_data[type]
        male_entries = getSpecificEntries(data_split, male_names)
        female_entries = getSpecificEntries(data_split, female_names)
        ret_str += "& {} & {} & {} & {}".format(countEntityPairs(male_entries),
                                                   countEntityPairs(female_entries),
                                                   countInstances(male_entries), countInstances(female_entries))
        ret_str += "\\\\ \n \\hline \n"

        print("For {} dataset, \n\n".format(type))
        print("For Male entries, \n num instances: {}\n num entpairs: {}".format(countInstances(male_entries),
                                                                                 countEntityPairs(male_entries)))
        print("\n")
        print("For Female entries, \n num instances: {}\n num entpairs: {}".format(countInstances(female_entries),
                                                                                   countEntityPairs(female_entries)))
        print("------------------------------------------------------------------------------------")

    print(ret_str)

def getAverageSentencesPerArticle(entries):
    '''
    NOTE that each entry is represents ONE article (each entry is a collection of distantly supervised sentences form that article)
    :param entries:
    :return:
    '''
    test = countInstances(entries)
    test2= len(entries)
    return countInstances(entries) / len(entries)


def initRelationDict():
    ret_dict = dict()

    for relation in RELATIONS:
        ret_dict[relation] = list()

    return ret_dict


def getProportionsOfSentencesPerRelationPerGender():
    '''

    :return: what percentage of 'spouse' sentences come from male vs female datapoints from wikigender
     and tehe saem fr all other elations
    '''
    relations_plus_NA = copy.deepcopy(RELATIONS)
    relations_plus_NA.append('NA')

    male_names, female_names = getMaleAndFemaleNames()

    # get the dataset
    data = readFromJsonFile('JsonData/Wikigender.json')
    ret_str = ""
    male_entries = list()
    female_entries = list()

    male_entries_per_relation = initRelationDict()
    male_entries_per_relation['NA'] = list()
    female_entries_per_relation = initRelationDict()
    female_entries_per_relation['NA'] = list()

    for type in DataTypes:
        data_split = data[type]

        male_entries.extend(getSpecificEntries(data_split, male_names))
        female_entries.extend(getSpecificEntries(data_split, female_names))

        for relation in relations_plus_NA:
            male_entries_per_relation[relation].extend(getSpecificEntries(data_split, male_names, relation=relation))
            female_entries_per_relation[relation].extend(
                getSpecificEntries(data_split, female_names, relation=relation))


    total_female_instances = countInstances(female_entries)
    total_male_instances = countInstances(male_entries)
    for relation in relations_plus_NA:
        male_entries_per_relation[relation] = countInstances(male_entries_per_relation[relation])
        female_entries_per_relation[relation] = countInstances(female_entries_per_relation[relation])
    ret_dict = initRelationDict()
    ret_dict['NA'] = list()
    for rel in ret_dict:
        total = male_entries_per_relation[rel] + female_entries_per_relation[rel]
        ret_dict[rel] = dict()
        ret_dict[rel]['female_percentage'] =  female_entries_per_relation[rel] / total_female_instances
        ret_dict[rel]['male_percentage'] = male_entries_per_relation[rel] / total_male_instances

    print(ret_dict)
    return ret_dict


def getMaleFemaleAverageInstancePerArticle2():
    male_names, female_names = getMaleAndFemaleNames()
    #data = readFromJsonFile('JsonData/Wikigender_NoNA_Eq_NoGS.json')
    data = readFromJsonFile('JsonData/new_Wikigender.json')

    male_rel_instance_totals = dict()
    female_rel_instance_totals = dict()

    total_male_articles = 0
    total_female_articles = 0

    for relation in RELATIONS:
        male_rel_instance_totals[relation] = 0
        female_rel_instance_totals[relation] = 0
    male_rel_instance_totals['NA'] = 0
    female_rel_instance_totals['NA'] = 0

    for type in DataTypes:
        data_split = data[type]

        for entry in data_split:
            male = False
            if entry['entity1'] in male_names:
                male = True
                total_male_articles += 1
            else:
                total_female_articles += 1
            for relation in entry['relations']:
                if len(relation['sentences']) == 0:
                    x = 6
                if male:
                    male_rel_instance_totals[relation['relation_name']] += len(relation['sentences'])
                else:
                    female_rel_instance_totals[relation['relation_name']] += len(relation['sentences'])


    total_male_instances = 0
    total_female_instances = 0
    for relation in RELATIONS:
        total_male_instances += male_rel_instance_totals[relation]
        total_female_instances += female_rel_instance_totals[relation]

    ave_per_article = dict()
    ave_per_article['male'] = dict()
    ave_per_article['female'] = dict()
    for relation in RELATIONS:
        ave_per_article['male'][relation] = male_rel_instance_totals[relation] / total_male_articles
        ave_per_article['female'][relation] = male_rel_instance_totals[relation] / total_female_articles


    print(ave_per_article)



def getMaleFemaleAverageInstancePerArticle():
    male_names, female_names = getMaleAndFemaleNames()

    # get the dataset
    data = readFromJsonFile('JsonData/Wikigender_NoNA_Eq_NoGS.json')
    ret_str = ""
    male_entries = list()
    female_entries = list()

    male_entries_per_relation = initRelationDict()
    female_entries_per_relation = initRelationDict()



    for type in DataTypes:
        data_split = data[type]

        male_entries.extend(getSpecificEntries(data_split, male_names))
        female_entries.extend(getSpecificEntries(data_split, female_names))

        for relation in RELATIONS:
            male_entries_per_relation[relation].extend(getSpecificEntries(data_split, male_names, relation=relation))
            female_entries_per_relation[relation].extend(getSpecificEntries(data_split, female_names, relation=relation))

    male_ave = getAverageSentencesPerArticle(male_entries)
    female_ave = getAverageSentencesPerArticle(female_entries)

    male_ave_per_relation = dict()
    female_ave_per_relation = dict()

    male_perc_per_relation = dict()
    female_perc_per_relation = dict()
    for relation in RELATIONS:
        male_ave_per_relation[relation] = getAverageSentencesPerArticle(male_entries_per_relation[relation])
        female_ave_per_relation[relation] = getAverageSentencesPerArticle(female_entries_per_relation[relation])


        #total = male_ave_per_relation[relation] + female_ave_per_relation[relation]
        #male_perc_per_relation[relation] = float(male_ave_per_relation[relation]/total)
        #female_perc_per_relation[relation] = float(female_ave_per_relation[relation]/total)




        #total_instances = countInstances(male_entries_per_relation[relation]) + countInstances(female_entries_per_relation[relation])
        #male_perc_per_relation[relation] = float(countInstances(male_entries_per_relation[relation]) / total_instances)
        #female_perc_per_relation[relation] = float(countInstances(female_entries_per_relation[relation]) / total_instances)






    print("Average sentences per article:\nMALE:{}\t\tFEMALE:{}\n".format(male_ave, female_ave))
    print("\n\nAVERAGE PER RELATION\n\nMALE:{}\n\nFEMALE:{}\n".format(male_ave_per_relation, female_ave_per_relation))
    print("\n\nPERCENT PER RELATION\n\nMALE:{}\n\nFEMALE:{}\n".format(male_perc_per_relation, female_perc_per_relation))

    return male_ave, female_ave



if __name__ == '__main__':
    #removeDuplicates()
    #data = readFromJsonFile('JsonData/new_Wikigender.json')
    #entries = getEntriesForDataType(data, 'female_test')
    #print(countEntityPairs(entries))
    #print(countInstances(entries))

    #printOutMaleFemaleDatasetNumbers()
    #printOutMaleFemaleDatasetNumbers()
    #getMaleFemaleAverageInstancePerArticle2()
    getProportionsOfSentencesPerRelationPerGender()