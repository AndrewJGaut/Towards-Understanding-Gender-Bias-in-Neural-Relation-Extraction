from nltk.stem import WordNetLemmatizer
import nltk
import json, string
from stanfordcorenlp import StanfordCoreNLP

#from corenlp import StanfordCoreNLP

'''lemmatize'''
lemmatizer = WordNetLemmatizer()

def lemmatize(str):
    new_str = ""
    for word in nltk.word_tokenize(str):
        new_str += lemmatizer.lemmatize(word)
        if(word[0].isalpha()):
            new_str += " "
    return new_str


def lemmatize_corenlp(conn_nlp, sentence):
    props = {
        'annotators': 'pos,lemma',
        'pipelineLanguage': 'en',
        'outputFormat': 'json'
    }

    # tokenize into words
    sents = conn_nlp.word_tokenize(sentence)

    # remove punctuations from tokenised list
    sents_no_punct = [s for s in sents if s not in string.punctuation]

    # form sentence
    sentence2 = " ".join(sents_no_punct)

    # annotate to get lemma
    parsed_str = conn_nlp.annotate(sentence2, properties=props)
    parsed_dict = json.loads(parsed_str)

    # extract the lemma for each word
    lemma_list = [v for d in parsed_dict['sentences'][0]['tokens'] for k,v in d.items() if k == 'lemma']

    # form sentence and return it
    return " ".join(lemma_list)


''''other stuff'''

def get_jaccard_sim(str1, str2):
    str1 = lemmatize(str1)
    str2 = lemmatize(str2)

    a = set(str1.split())
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

#nlp = StanfordCoreNLP('~/Documents/JavaProjects/Eclipse/NLP/CoreNLPExample/stanford-corenlp-full-2018-10-05/JAR_FILES/')
#nlp = StanfordCoreNLP('~/Documents/JavaProjects/OpenIE-Tests/JAR_FILES')
#nlp = StanfordCoreNLP('./stanford-corenlp-full-2018-10-05')
#print(lemmatize_corenlp(conn_nlp=nlp, sentence="Players know the game, but I've been playing and I ain't the one"))

#print(get_jaccard_sim("AI is our friend and it has been friendly", "AI and humans have always been friendly"))


# TRY THIS!!!!
'''
from corenlp import StanfordCoreNLP
corenlp_dir = "/Users/agaut/Documents/JavaProjects/Eclipse/NLP/CoreNLPExample/stanford-corenlp-full-2018-10-05/JAR_FILES"
corenlp = StanfordCoreNLP(corenlp_dir)
corenlp.raw_parse("Several women told me I have lying eyes.")
'''

#lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(sentence) if w not in string.punctuation



def createInputData(sentence, relation, subj, obj):
    pass





from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.stem import LancasterStemmer
import contractions

stemmer = LancasterStemmer()


def lemmatize_nltk(sentence):
    sentence = contractions.fix(sentence)
    pos_tags = pos_tag(nltk.word_tokenize(sentence))
    new_sentence = ""

    for i in range(len(pos_tags)):
        if not pos_tags[i][0].isalpha():
            new_sentence += pos_tags[i][0]
        else:
            stem = stemmer.stem(pos_tags[i][0])
            tag = get_wordnet_pos(pos_tags[i][1])

            '''
            print("----")
            print(stem)
            print(tag)
            print("----")
            '''

            lemma = pos_tags[i][0]
            if(tag != ''):
                #lemma = lemmatizer.lemmatize(stem, pos=tag)
                lemma = lemmatizer.lemmatize(pos_tags[i][0], pos=tag)
            new_sentence += " "
            new_sentence += lemma.lower()


    return new_sentence

def get_positions(sentence):
    pos_tags = pos_tag(nltk.word_tokenize(sentence))
    return pos_tags # returns list [(word, pos_tag), (), ()]

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''

def get_jaccard_sim(str1, str2):
    str1 = lemmatize_nltk(str1)
    str2 = lemmatize_nltk(str2)

    a = set(str1.split())
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


print(lemmatize_nltk("I like apples; but I also like bananas"))
print(lemmatize_nltk("I'm playing ball, but I sure ain't the GOAT"))
print(lemmatize_nltk("AI is our friend and it has been friendly"))
print(lemmatize_nltk("AI and humans have always been friendly"))
print(get_jaccard_sim("AI is our friend and it has been friendly", "AI and humans have always been friendly"))
