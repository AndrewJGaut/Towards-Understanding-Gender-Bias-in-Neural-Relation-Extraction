from genderSwap import *
from Utility import *
from DebiasJsonDataset import createNameAnonymizationDict, nameAnonymizeJson, createGenderSwappedDatasetEntries
import nltk

swapdict = createSwapDict()

def prettify(file_name):
    data = readFromJsonFile(file_name)
    writeToJsonFile(data, 'pretty2.json', True)



#print(genderSwap("In early 2010 , Michelle spoke about her husband's smoking habit and said Barack had quit smoking ."))

'''
[{"sentence": "In early 2010 , Michelle spoke about her husband's smoking habit and said Barack had quit smoking .", "relation": "spouse", "head": {"word": "Barack", "id": "/guid/9202a8c04000641f8000000000011111"}, "tail": {"word": "Michelle", "id": "/guid/9202a8c04000641f80000000000abcde"}},
[{"sentence": "In early 2010 , Michelle spoke about his wife 's smoking habit and said Barack had quit smoking .", "relation": "spouse", "head": {"word": "Barack", "id": "/guid/9202a8c04000641f8000000000011111"}, "tail": {"word": "Michelle", "id": "/guid/9202a8c04000641f80000000000abcde"}},
'''

'''
[{"sentence": "On December 4 , 1982 , he married actress Daphne Maxwell Reid .", "relation": "spouse", "head": {"word": "Tim Reid", "id": "/guid/9202a8c04000641f800000000007u1f8"}, "tail": {"word": "Daphne Reid", "id": "/guid/9202a8c04000641f800000000008riyb"}},
{"sentence": "On December 4 , 1982 , she married actor Daphne Maxwell Reid .", "relation": "spouse", "head": {"word": "Tim Reid", "id": "/guid/9202a8c04000641f800000000007u1f8"}, "tail": {"word": "Daphne Reid", "id": "/guid/9202a8c04000641f800000000008riyb"}}]
'''


swapdict = createSwapDict()
male_set, female_set = createGenderedSets()

def filterResults(weight_length = True, weight_names = False, relation_to_look_for='spouse', max_len=80):
    #train_data = readFromJsonFile('OpenNREData/train_NoNA_NoEq_NoGS.json')
    train_data = readFromJsonFile('pretty.json')

    instance_scores = list()
    instance_dict = dict()
    for instance in train_data:
        if instance['relation'] != relation_to_look_for:
            continue
        sentence = instance['sentence']
        if(len(sentence) > max_len):
            continue
        score = 0
        words = nltk.word_tokenize(sentence)
        for word in words:
            if word in swapdict:
                score += 1

                if weight_names:
                    if word.lower() in male_set or word.lower() in female_set:
                        score += 1000 * (1/len(sentence))
        if weight_length:
            score /= len(sentence)
        instance_scores.append(score)
        if score not in instance_dict:
            instance_dict[score] = list()
        instance_dict[score].append(instance)

    instance_scores = sorted(instance_scores)
    ins = list()
    seen = set()
    for score in instance_scores[-150:]:
        items = instance_dict[score]
        for item in items:

            if item['sentence'] in seen:
                continue
            seen.add(item['sentence'])
            print(item['sentence'])


def bidirectionalRelations(relation):
    train_data = readFromJsonFile('OpenNREData/train_NoNA_NoEq_NoGS.json')
    bidir_entries = set()
    entries = dict()

    for instance in train_data:
        if(instance['relation'] == relation):
            tail = instance['tail']['word']
            head = instance['head']['word']
            if tail in entries:
                if entries[tail] == head:
                    bidir_entries.add(head,tail)
                else:
                    entries[head] = tail

    print(bidir_entries)









def getInstancesForSentences(sentences):
    train_data = readFromJsonFile('OpenNREData/train_NA_NoEq_NoGS.json')

    train_sents = set()
    sent2ins = dict()
    for ins in train_data:
        train_sents.add(ins['sentence'])
        sent2ins[ins['sentence']] = ins

    ins_list = list()
    for sentence in sentences:
        if sentence in train_sents:
            ins_list.append(sent2ins[sentence])

    print(ins_list)
    writeToJsonFile(ins_list, 'regular.json', True)

    gs_ins_list = list()
    for i in range(len(ins_list)):
        ins = ins_list[i]
        sent = ins['sentence']
        gs_sent = genderSwap(sent)

        ins['sentence'] = gs_sent

        gs_ins_list.append(ins)

    writeToJsonFile(gs_ins_list, 'gs.json', True)

    '''
    anon_dict = createNameAnonymizationDict(ins_list)
    for i in range(len(ins_list)):
        ins = ins_list[i]
        sent = ins['sentence']
        gs_sent = genderSwap(sent)

        ins['sentence'] = gs_sent

        ins_list[i] = ins
    '''

#print(genderSwap('Marge was with the actress, the boy, the girl, and the businessman Mr. Johnson, she told my aunt.', neutralize=True))
input_data = {
    'train': [
        {
            'entity1': 'test1',
            'relations': [
                {
                    'relation_name': 'spouse',
                    'entity2': 'Johnny',
                    'sentences': [
                        'Johnny and his mother were talking to her father on the sister.',
                        'Marge was with the actress, the boy, the girl, and the businessman Mr. Johnson, she told my aunt.',

                    ]
                }

            ]
        }
    ],
    'dev': [

    ],
    'male_test': [

    ],
    'female_test': [

    ]
}
#print(createGenderSwappedDatasetEntries(input_data, neutralized=True))
# bidirectionalRelations('spouse')
prettify('OpenNREData/female_test.json')
#filterResults(relation_to_look_for='spouse')


#getInstancesForSentences(["[ 7 ] Kuhn is married to his wife Denise and together they have two sons : Mason and Alex .", "[ 5 ] He was survived by his wife , Florence Alice Fairchild , and his daughter , Betsy Anne Calvert .", "[ 1 ] Brower lives in Reno , Nevada , with his wife Loren .", "[ 7 ] Kuhn is married to his wife Denise and together they have two sons : Mason and Alex .", "Johnson was survived by his son Paul , his daughter Judy , and his wife Lola of 63 years ."])
#getInstancesForSentences(["Dodd said that he called Mrs. Long to inform her of his intentions to assist Earl , and she thanked him for helping her husband .", "Nawazuddin is married to Anjali and they have a daughter , Shora , and a son who was born on the actor 's 41st birthday .", "[ 1 ] He married twice ; his second wife was Naomi Carry .", "[ 46 ] Yoho and his wife , Carolyn , have three children .", "[ 2 ] Cairoli had three children with his wife , Violetta .", "[ 33 ] That year , he married Tao Li , with whom he had a son named Liu Tao in 1985 .", "[ 21 ] George W. Bush credited his wife with his decision to stop drinking in 1986 .", "This is where she met her husband , George William Andrews ."])

# george bush
#getInstancesForSentences(["`` [ 9 ] She met George W. Bush in July 1977 when mutual friends Joe and Jan O'Neill invited her and Bush to a backyard barbecue at their home .", "[ 21 ] George W. Bush credited his wife with his decision to stop drinking in 1986 .", "According to George Bush , when he asked her to marry him , she had said , `` Yes .", "She was to arrive there with her daughter Barbara Pierce Bush , her husband George W. Bush , and Soledad O'Brien , a journalist ."])
#getInstancesForSentences(["E30 E30 E30 E30 E5316 E22643 E377 E61 E423 E14794 E3115 E30 when mutual friends E940 E4 E1688 E3159 invited E4587 E4 E423 E22714 E375 backyard barbecue at their home E30", "E30 E30 E30 E377 E61 E423 credited E1725 E15873 E20638 E1725 decision E22714 stop drinking E14794 E30 E30", "According E22714 E377 E423 E30 when E1540 asked E4587 E22714 marry her E30 E5316 had E5366 E30 E30 Yes E30", "E5316 was E22714 arrive there E20638 E4587 E2432 E36 E2621 E423 E30 E4587 E10754 E377 E61 E423 E30 E4 E7392 E2915 E30 E375 journalist E30"])

'''
NOTE:!!! A LOT OF THESE ARE USELESS !!!  Here's why: the attention will always be 1 unless there are multiple sentences in the bag. In this case, for all of these, there is only one sentence per entityrel pair, so there will always only be one sentence int he bag. That's the motivation for using George Bush above
Name heavy
["[ 7 ] Kuhn is married to his wife Denise and together they have two sons : Mason and Alex .", [ 5 ] He was survived by his wife , Florence Alice Fairchild , and his daughter , Betsy Anne Calvert ."]

[ 47 ] Tan was survived by his wife Mrs. Tan Sook Yee , his son Pip Tan Seng Hin and daughter Tan Sui Lin , and five grandchildren .

[ 1 ] Brower lives in Reno , Nevada , with his wife Loren .

[ 7 ] Kuhn is married to his wife Denise and together they have two sons : Mason and Alex .

Johnson was survived by his son Paul , his daughter Judy , and his wife Lola of 63 years .


less complex
Dodd said that he called Mrs. Long to inform her of his intentions to assist Earl , and she thanked him for helping her husband .

]In 1993 , he married journalist Nandita Puri , with whom he had a son named Ishaan .

Nawazuddin is married to Anjali and they have a daughter , Shora , and a son who was born on the actor 's 41st birthday .
 
[ 1 ] He married twice ; his second wife was Naomi Carry .

[ 46 ] Yoho and his wife , Carolyn , have three children .
 
[ 2 ] Cairoli had three children with his wife , Violetta .

[ 33 ] That year , he married Tao Li , with whom he had a son named Liu Tao in 1985 .

[ 21 ] George W. Bush credited his wife with his decision to stop drinking in 1986 .

This is where she met her husband , George William Andrews .

ageorge bush sentences annonned:

"E30 E30 E30 E30 E5316 E22643 E377 E61 E423 E14794 E3115 E30 when mutual friends E940 E4 E1688 E3159 invited E4587 E4 E423 E22714 E375 backyard barbecue at their home E30"

"E30 E30 E30 E377 E61 E423 credited E1725 E15873 E20638 E1725 decision E22714 stop drinking E14794 E30 E30"

"According E22714 E377 E423 E30 when E1540 asked E4587 E22714 marry her E30 E5316 had E5366 E30 E30 Yes E30"

"E5316 was E22714 arrive there E20638 E4587 E2432 E36 E2621 E423 E30 E4587 E10754 E377 E61 E423 E30 E4 E7392 E2915 E30 E375 journalist E30"'''


