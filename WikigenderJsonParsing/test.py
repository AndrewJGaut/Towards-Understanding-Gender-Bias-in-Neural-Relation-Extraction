from DebiasJsonDataset import getEqualizedEntriesThroughDownSampling
from Utility import writeToJsonFile


def getNamesFromFileToDict(filename):
    file = open(filename, 'r')
    namesDict = set()

    for line in file.readlines():
        namesDict.add(line.strip())

    return namesDict

#def getEqualizedEntriesThroughDownSampling(entries, male_names, female_names):
male_names = getNamesFromFileToDict('NamesAndSwapLists/male_names.txt')
female_names = getNamesFromFileToDict('NamesAndSwapLists/female_names.txt')
data =  [{
            "entity1": "Ross Mirkarimi",
            "relations": [
                {
                    "entity2": "Sheriff",
                    "relation_name": "hypernym",
                    "sentences": [
                        "Rostam Mirkarimi ( born August 4 , 1961 ) is an American politician and the former Sheriff of San Francisco .",
                        "[ 17 ] [ 18 ] Mirkarimi , in collaboration with Public Defender Jeff Adachi , District Attorney Kamala Harris and Sheriff Michael Hennessey , crafted the legislation to increase the effectiveness of City-wide efforts to reduce recidivism and violence , and promote safe and successful reentry into society for adults released from jails and prisons.In March 2007 , Mirkarimi introduced legislation that prohibits large supermarkets and drugstores from providing customers with non-biodegradable plastic bags , making San Francisco the first city to regulate such bags .",
                        "[ 27 ] Mirkarimi did not receive the endorsement of the San Francisco Deputy Sheriff 's Association , the union representing sheriffs .",
                        "On October 9 , 2012 , only seven supervisors voted to remove Mirkarimi as Sheriff , and he was duly reinstated .",
                        "[ 70 ] In November 2013 , Sheriff Mirkarimi publicly apologized for his department 's slow and incomplete search for Lynne Spalding , a San Francisco General Hospital patient whose body was found in a stairwell by a hospital engineer two weeks after she went missing from her hospital bed .",
                        "Mirkarimi was banned from carrying a firearm , and by definition could no longer be Sheriff since he was not qualified at the range .",
                        "[ 85 ] He ran against Vicki Hennessy , who served as interim sheriff when Mirkarimi was suspended from his post as Sheriff in 2012 .",
                        "[ 86 ] In March 2015 , Mirkarimi failed to receive the endorsement of the hundred-member San Francisco Sheriff 's Managers and Supervisors Association , only seven of whom voted to endorse him .",
                        "Sheriff Mirkarimi reeling from scandal over forced fights , '' the San Francisco Chronicle suggested that Mirkarimi would have difficulty being re-elected in light of recent scandals in the Sheriff 's Department \u2014 an escaped prisoner and a report that deputies in San Francisco County Jail had forced prisoners to fight each other for the guards ' amusement ."
                    ]
                },
                {
                    "entity2": "Eliana Lopez",
                    "relation_name": "spouse",
                    "sentences": [
                        "[ 4 ] The charges came five days after he was sworn in publicly as sheriff and resulted from an altercation Mirkarimi had with his wife , Eliana Lopez , before he became sheriff , on New Year 's Eve .",
                        "`` [ 38 ] [ 39 ] Mirkarimi 's wife , Eliana Lopez , repudiated the charges against her husband ."
                    ]
                },
                {
                    "entity2": "August 4, 1961",
                    "relation_name": "birthDate",
                    "sentences": [
                        "Rostam Mirkarimi ( born August 4 , 1961 ) is an American politician and the former Sheriff of San Francisco ."
                    ]
                },
                {
                    "entity2": "Chicago",
                    "relation_name": "birthPlace",
                    "sentences": [
                        "He lost his reelection bid to Vicki Hennessy in 2015.Mirkarimi was born in Chicago to Nancy Kolman , a 19-year-old descended from Russian Jews , and Hamid Mirkarimi , an Iranian immigrant ."
                    ]
                },
                {
                    "entity2": "",
                    "relation_name": "NA",
                    "sentences": []
                }
            ]
        },{
            "entity1": "Talitha Cummins",
            "relations": [
                {
                    "entity2": "Cummins",
                    "relation_name": "hypernym",
                    "sentences": [
                        "Talitha Cummins ( born 27 April 1980 ) is an Australian journalist.Cummins has previously been a news presenter on Weekend Sunrise , reporter for Seven News and weather presenter on Seven News Brisbane .",
                        "[ citation needed ] Cummins has worked in many of Seven \u2019 s Queensland \u2019 s bureaus , she started out at Maroochydore before moving onto Cairns .",
                        "However , after the bulletin continued to lag behind Nine News Queensland and Ten News Brisbane in the ratings , Cummins was relegated to weekend duties , replaced on weeknights by former Nine weatherman John Schluter from February 2007 .",
                        "She occasionally filled-in for the local Brisbane bulletin when either Kay McGrath or Sharyn Ghidella were on holiday or ill.In June 2007 , Cummins joined Weekend Sunrise as a news presenter on Sunday mornings .",
                        "She held the position until July 2008 where she was replaced by Sharyn Ghidella.In 2011 , Cummins moved to Sydney where she is a reporter for Seven News Sydney and a fill in presenter on Seven Morning News and Seven Afternoon News .",
                        "[ 1 ] In January 2014 , Cummins was appointed news presenter on Weekend Sunrise replacing Jessica Rowe .",
                        "Cummins remained in the role until she went on maternity leave in September 2016.In January 2017 , it was revealed that the Seven Network had dismissed Cummins whilst she was on maternity leave .",
                        "[ 3 ] Cummins is also a casual Triple M Sydney newsreader.Cummins was born on the Gold Coast on 27 April 1980 .",
                        "[ citation needed ] Cummins was engaged to former Olympic beach volleyball player Lee Zahner however they separated in 2010 .",
                        "[ 4 ] In May 2013 , Cummins announced that she is engaged to personal trainer Ben Lucas and in October 2013 , they married in New York .",
                        "[ 2 ] It was revealed on the ABC 's Australian Story ( 10 October 2016 ) that she has battled alcoholism.In January 2016 , Cummins announced she is pregnant with her first child .",
                        "In June 2018 , Cummins announced that she was pregnant with her second child ."
                    ]
                },
                {
                    "entity2": "Ben Lucas",
                    "relation_name": "spouse",
                    "sentences": [
                        "[ 4 ] In May 2013 , Cummins announced that she is engaged to personal trainer Ben Lucas and in October 2013 , they married in New York ."
                    ]
                },
                {
                    "entity2": "April 27, 1980",
                    "relation_name": "birthDate",
                    "sentences": [
                        "Talitha Cummins ( born 27 April 1980 ) is an Australian journalist.Cummins has previously been a news presenter on Weekend Sunrise , reporter for Seven News and weather presenter on Seven News Brisbane .",
                        "[ 3 ] Cummins is also a casual Triple M Sydney newsreader.Cummins was born on the Gold Coast on 27 April 1980 ."
                    ]
                },
                {
                    "entity2": "Gold Coast, Queensland",
                    "relation_name": "birthPlace",
                    "sentences": [
                        "[ 3 ] Cummins is also a casual Triple M Sydney newsreader.Cummins was born on the Gold Coast on 27 April 1980 ."
                    ]
                },
                {
                    "entity2": "",
                    "relation_name": "NA",
                    "sentences": []
                }
            ]
        }
    ]
entries = getEqualizedEntriesThroughDownSampling(data, male_names, female_names)
writeToJsonFile(entries, 'file.json', True)