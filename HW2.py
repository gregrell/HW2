import spacy
import os
from collections import Counter
import pickle
import math
import numpy as np





def Run():
    print('Using SpaCy version', spacy.__version__)
    print("Performing Segmentation")
    hockeyDirectory ="20news-bydate/20news-bydate-train/rec.sport.hockey"
    autosDirectory ="20news-bydate/20news-bydate-train/rec.autos"
    hockeyTestDirectory="20news-bydate/20news-bydate-test/rec.sport.hockey"
    autosTestDirectory ="20news-bydate/20news-bydate-test/rec.autos"
    pathNlpHockeyDocs="nlpHockeyDocs.txt"
    pathNlpAutosDocs="nlpAutosDocs.txt"
    pathNlpHockeyTestDocs="nlpHockeyTestDocs.txt"
    pathNlpAutosTestDocs="nlpAutosTestDocs.txt"
    pathTrainingFeatures="trainingFeatures.txt"
    pathHockeyFeatures="hockeyFeatures.txt"
    pathAutosFeatures="autosFeatures.txt"

    #Grab file data into doc list objects
    hockeyDocs=readDocs(hockeyDirectory)
    autosDocs=readDocs(autosDirectory)
    hockeyTestDocs=readDocs(hockeyTestDirectory)
    autosTestDocs=readDocs(autosTestDirectory)
    print('training data read')


    #load/save nlp docs to aide in speed of development
    nlpHockeyDocs,read=openNlpDoc(pathNlpHockeyDocs)
    if not read:
        print('Could not find saved data... performing nlp conversion on Hockey Docs')
        nlpHockeyDocs = nlpDocs(hockeyDocs)
        saveNlpDoc(pathNlpHockeyDocs,nlpHockeyDocs)

    nlpAutosDocs,read=openNlpDoc(pathNlpAutosDocs)
    if not read:
        print('Could not find saved data... performing nlp conversion on Autos Docs')
        nlpAutosDocs = nlpDocs(autosDocs)
        saveNlpDoc(pathNlpAutosDocs,nlpAutosDocs)

    nlpHockeyTestDocs,read=openNlpDoc(pathNlpHockeyTestDocs)
    if not read:
        print('Could not find saved data... performing nlp conversion on Hockey Testing Docs')
        nlpHockeyTestDocs = nlpDocs(hockeyTestDocs)
        saveNlpDoc(pathNlpHockeyTestDocs,nlpHockeyTestDocs)

    nlpAutosTestDocs, read = openNlpDoc(pathNlpAutosTestDocs)
    if not read:
        print('Could not find saved data... performing nlp conversion on Autos Testing Docs')
        nlpAutosTestDocs = nlpDocs(autosTestDocs)
        saveNlpDoc(pathNlpAutosTestDocs, nlpAutosTestDocs)




    print('Total Hockey Docs ',len(nlpHockeyDocs))
    print('Total Autos Docs', len(nlpAutosDocs))
    print('Total Hockey TESTING Docs ',len(nlpHockeyTestDocs))
    print('Total Autos TESTING Docs', len(nlpAutosTestDocs))

    #sentence segmentation
    sentences_Hockey=sentences(nlpHockeyDocs)
    sentences_Autos=sentences(nlpAutosDocs)

    print("Number of sentences in Hockey training",len(sentences_Hockey))
    print("Number of sentences in Autos training",len(sentences_Autos))
    print("Total number of sentences in training data ",len(sentences_Autos)+len(sentences_Hockey))



    #tokenize
    tokenizedHockeyDocs = tokenize(nlpHockeyDocs)
    tokenizedAutosDocs = tokenize(nlpAutosDocs)
    tokenizedHockeyTest = tokenize(nlpHockeyTestDocs)
    tokenizedAutosTest = tokenize(nlpAutosTestDocs)
    print('Tokenized  Docs')

    #group into bag of unique words returned as Counter objects
    counterHockey,bagHockey=bagOfWords(tokenizedHockeyDocs)
    counterAutos,bagAutos=bagOfWords(tokenizedAutosDocs)


    counterBoth=counterHockey+counterAutos
    totalDocs=len(nlpHockeyDocs)+len(nlpAutosDocs)
    priorHockey=len(hockeyDocs)/totalDocs
    priorAutos=len(autosDocs)/totalDocs

    #print(counterHockey)
    #print(counterAutos)
    #print(counterBoth)
    print('Unique words in Hockey Category ',len(counterHockey))
    print('Unique words in Autos Category ',len(counterAutos))
    print('Unique words in both categories ',len(counterBoth))
    print('Total Hockey Words', len(bagHockey))
    print('Total Autos Words', len(bagAutos))
    print('Prior of class Hockey',priorHockey)
    print('Prior of class Autos',priorAutos)

    #Create feature set based on bag of words model from both classes
    #first need the dictionary of all words
    dictAll=counterBoth.items()
    print(dictAll)

    trainingFeatures, read = openNlpDoc(pathTrainingFeatures)
    if not read:
        print('Could not find saved data... performing feature count on training features')
        featuresHockey = featureCount(dictAll, bagHockey)
        featuresAutos = featureCount(dictAll, bagAutos)
        trainingFeatures = np.row_stack((featuresHockey, featuresAutos))
        saveNlpDoc(pathTrainingFeatures, trainingFeatures)

    hockeyTestFeatures, read = openNlpDoc(pathHockeyFeatures)
    if not read:
        print('Could not find saved data... performing feature count on hockey test set')
        hockeyTestFeatures = []
        for doc in tokenizedHockeyTest:
            docFeatures = featureCount(dictAll, doc)
            hockeyTestFeatures.append(docFeatures)
        saveNlpDoc(pathHockeyFeatures, hockeyTestFeatures)

    autosTestFeatures, read = openNlpDoc(pathAutosFeatures)
    if not read:
        print('Could not find saved data... performing feature count on autos test set')
        autosTestFeatures = []
        for doc in tokenizedAutosTest:
            docFeatures = featureCount(dictAll, doc)
            autosTestFeatures.append(docFeatures)
        saveNlpDoc(pathAutosFeatures, autosTestFeatures)

    print('Feature Set Loaded')

    #Multinomial Naive Bayes Code
    from sklearn.naive_bayes import MultinomialNB
    print("sklearn ")
    X = trainingFeatures
    y = np.array(['Hockey', 'Autos',])  # classes
    clf = MultinomialNB()
    clf.fit(X, y)
    # MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
    print("line")
    print(X)
    #print(clf.predict(hockeyTestFeatures[0]))










    #Perform Naive Bayes Classification
    #pHockey=classDoc(nlpHockeyTestDocs,counterHockey,bagHockey,counterBoth,priorHockey)
    #print('probability that doc is in hockey ',pHockey)
    #pAutos=classDoc(nlpHockeyTestDocs,counterAutos,bagAutos,counterBoth,priorAutos)
    #print('probability that doc is in Autos ',pAutos)








#Read the document files return the raw text as a list of docs
def readDocs(path):
    docAsList=[]
    for doc in os.listdir(path):
        with open(os.path.join(path,doc)) as file:
            docAsList.append(file.read())
            file.close()
    return docAsList

#convert raw text docs into nlp 'spacy' docs return a list of spacy docs
def nlpDocs(docs):
    nlp = spacy.load("en_core_web_sm")
    converted=[]
    for doc in docs:
        spacy_doc=nlp(doc)
        converted.append(spacy_doc)
    return converted

#Sentence Segmenter, return list of sentences given a list of nlp docs
def sentences(nlpDocs):
    sentences=[]
    for nlpDoc in nlpDocs:
        for sent in nlpDoc.sents:
            sentences.append(sent)
    return sentences

#Tokenize document, remove stop words, apply filter, return a list of lists. List data is the doc tokens
def tokenize(docs):
    tokenizedDocs=[]
    for doc in docs:
        wordlist=[]
        for token in doc:
            if token.pos_ != 'SYM' and token.pos_ != 'NUM' and token.pos_ !='PUNCT' and token.pos_ !='X' and token.pos_ !='SPACE' and not token.is_stop and token.text!='_' and token.text!='|':
                wordlist.append(token.text.lower())
        tokenizedDocs.append(wordlist)
    return tokenizedDocs

#take in tokenized docs return a dictionary of the words as a counter object as well as all the text from the documents as a list
def bagOfWords(docs):
    bagofWords = []
    for doc in docs:
        bagofWords.extend(doc)
    cnt = Counter(bagofWords)
    return cnt, bagofWords
#Used to save NLP data for speed in development instead of performing each run
def saveNlpDoc(path,docs):
    fb = open(path, 'wb')
    pickle.dump(docs, fb)
    print('Saved data to ',path)
    fb.close()
#Used to save NLP data for speed in development instead of performing each run
def openNlpDoc(path):
    itemlist=[]
    if os.path.exists(path):
        fb = open(path, 'rb')
        itemlist=pickle.load(fb)
        print('Successfully read data from ',path)
        return itemlist,True
    else:
        print('Could not find ',path)
        return itemlist,False



def featureCount(dict, words):
    vector=[]
    for feature in dict:
        number=words.count(feature[0])
        vector.append(number)
    return vector

def docAsFeatureArray(tokens,dict):
    vector=featureCount(dict,tokens)
    return vector









#NB from scratch --- ABANDONNED
def classDoc(docs,counter,bag,uniqueWords,prior):
    tkDocs=tokenize(docs)
    overall_p=0
    for token in tkDocs[100]:
        overall_p+= likelihood(token,counter,bag)
    return overall_p+math.log10(prior)



def getOccurrences(token,counter):
    result=counter.get(token)
    if result == None:
        result=0
    return result




def likelihood(token,counter,bag):
    l=getOccurrences(token,counter)/len(bag)
    if l!=0:
        return math.log10(l)
    else:
        return 0



