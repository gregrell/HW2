import spacy
import os
from collections import Counter
import pickle





def perform():
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


    nlpAutosTestDocs,read=openNlpDoc(pathNlpAutosTestDocs)
    if not read:
        print('Could not find saved data... performing nlp conversion on Autos Testing Docs')
        nlpAutosTestDocs = nlpDocs(autosTestDocs)
        saveNlpDoc(pathNlpAutosTestDocs,nlpAutosTestDocs)

    print('Total Hockey Docs ',len(nlpHockeyDocs))
    print('Total Autos Docs', len(nlpAutosDocs))
    print('Total Hockey TESTING Docs ',len(nlpHockeyTestDocs))
    print('Total Autos TESTING Docs', len(nlpAutosTestDocs))


    #tokenize
    tokenizedHockeyDocs = tokenize(nlpHockeyDocs)
    tokenizedAutosDocs = tokenize(nlpAutosDocs)

    print('Tokenized Training Docs')

    #group into bag of unique words returned as Counter objects
    dictHockeyWords,bagofHockeyWords=bagOfWords(tokenizedHockeyDocs)
    dictAutosWords,bagofAutosWords=bagOfWords(tokenizedAutosDocs)
    print(dictHockeyWords)
    print(dictAutosWords)
    print('Unique words in Hockey Category ',len(dictHockeyWords))
    print('Unique words in Autos Category ',len(dictAutosWords))
    print('Total Hockey Words', len(bagofHockeyWords))
    print('Total Autos Words', len(bagofAutosWords))


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

def saveNlpDoc(path,docs):
    fb = open(path, 'wb')
    pickle.dump(docs, fb)
    print('Saved data to ',path)
    fb.close()

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








