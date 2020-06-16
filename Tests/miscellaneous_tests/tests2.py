import requests

'''
person_json = requests.get('http://dbpedia.org/data/Barack_Obama.json').json()
person_data = person_json['http://dbpedia.org/resource/Barack_Obama']
#person_attr = person_data['http://dbpedia.org/ontology/' + attribute][0]['value']
gender = person_data['http://xmlns.com/foaf/0.1/gender'][0]['value']
print(gender)


for thing in person_data:
    if 'gender' in thing or 'spouse' in thing:
        print(thing)



spears_json = requests.get('http://dbpedia.org/data/Britney_Spears.json').json()
spears_data = spears_json['http://dbpedia.org/resource/Britney_Spears']
for thing in spears_data:
    print(thing)
'''
'''
data = requests.get('http://dbpedia.org/data/')
for thing in data:
    pass
    #print(thing)

x = requests.get('http://dbpedia.org/resource/')
for thing in x:
    print(thing)
'''

'''
def formatName(name):
    words = name.split()
    name = ""
    for i in range(len(words)):
        name += words[i]
        if i != len(words) - 1:
             name += "_"
    return name


print(formatName("Andrew Gaut"))
'''

def getNameFromUrl(url):
    words = url.split('/')
    name = words[-1]
    if '(' in name:
        name = name[0:name.rindex('(') - 1]
    name = name.replace('_', ' ')
    return name

def getAttributeForPerson(person_name, attribute):
    person_name = formatName(person_name)
    person_json = requests.get('http://dbpedia.org/data/' + person_name + '.json').json()
    person_data = person_json['http://dbpedia.org/resource/' + person_name]
    try:
        person_attr = person_data['http://dbpedia.org/ontology/' + attribute][0]['value']
    except:
        try:
            person_attr = person_data['http://xmlns.com/foaf/0.1/' + attribute][0]['value']
        except:
            try:
                person_attr = person_data['http://dbpedia.org/property/' + attribute][0]['value']
            except:
                try:
                    person_attr = person_data['http://www.w3.org/1999/02/22-rdf-syntax-ns#' + attribute][0]['value']
                except:
                    try:
                        person_attr = person_data['http://purl.org/linguistics/gold/' + attribute][0]['value']
                    except:
                        return 'ERROR: could not find attribute'

    if('/' in person_attr or '_' in person_attr):
        person_attr = getNameFromUrl(person_attr)
    return person_attr

'''
def formatName(name):
    words = name.split()
    name = ""
    for i in range(len(words)):
        name += words[i]
        if i != len(words) - 1:
             name += "_"
    return name


def getGenderedLists():
    males = dict()
    females = dict()

    names = ['Barack Obama', 'Britney Spears', 'Hillary Clinton']

    for name in names:
        formattedName = formatName(name)
        try:
            gender = getAttributeForPerson(formattedName, 'gender')
        except:
            continue
        if(gender == 'male'):
            if name not in males:
                males[name] = name
        if(gender == 'female'):
            if name not in females:
                females[name] = name

    return males, females

def writeKeysToFiles(males, females):
    with open('male_names.txt', 'w') as file:
        for name in males:
            file.write(name+'\n')
    with open('female_names.txt', 'w') as file:
        for name in females:
            file.write(name+'\n')

males, females = getGenderedLists()

print(males)
print("********")
print(females)

writeKeysToFiles(males, females)
'''

def formatName(name):
    words = name.split()
    name = ""
    for i in range(len(words)):
        name += words[i]
        if i != len(words) - 1:
             name += "_"
    return name


def getAttribs(person_name):
    person_name = formatName(person_name)
    person_json = requests.get('http://dbpedia.org/data/' + person_name + '.json').json()
    person_data = person_json['http://dbpedia.org/resource/' + person_name]
    for attrib in person_data:
        #print(attrib)
        print(attrib)
        '''
        if 'entity' in attrib:
            print(attrib)
        '''

#getAttribs('Barack Obama')
#print(getAttributeForPerson('Barack Obama', 'hypernym'))
#print(getAttributeForPerson('Barack Obama', 'isPrimaryTopicOf'))

import nltk

def opennreFormatSentence(sentence):
    new_sentence = ""
    for word in nltk.word_tokenize(sentence):
        new_sentence += word.lower() + " "

    return new_sentence

'''
print(opennreFormatSentence("I like apples; but I also like you."))

print(opennreFormatSentence("Dr. Johnson, reporting for duty; at your service, SIr."))
print(opennreFormatSentence("Mr. Johnson, reporting for duty; at your service, SIr."))
print(opennreFormatSentence("facebook.com is a cool website?"))
'''


'''CHANGE SO THIS DONE ONE PASS THROUGH EACH ARTICLE'''
'''
Precondition:
    article is e1's WIkipedia article that is ALREADY LEMMATIZED
    relation is the relation between the two entities
    e1 is the entity from whose Wikipedia article we are taking sentences
    e2 is the entity that relates to e1 in the relation on DBPedia
    (e.g. Barack marriedTo Michelle --> relation:marriedTo, e1:Barack, e2:Michelle)
Postcondition:
    returns a list of tuples representing relations
'''
def getRelationTuples(article, relation, e1, e2):
    relations = list()
    e1 = lemmatize(e1)
    e2 = lemmatize(e2)
    for sentence in nltk.sent_tokenize(article):
        if e1 in sentence and e2 in sentence:
            # then we know we want this relation tuple
            sentence = opennreFormatSentence(sentence)
            relations.append((relation, e1, e2, sentence))

    return relations


def formatDate(date):
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
                 'August', 'September', 'October', 'November', 'December']

    year, month, day = date.split('-')

    return str(months[int(month) - 1]) + " " + str(day) + ", " + str(year)

print(formatDate('1981-6-15'))