import spacy
import os


def perform():
    print('Using SpaCy version', spacy.__version__)
    print("Performing Segmentation")
    hockeyDirectory ="20news-bydate/20news-bydate/20news-bydate-train/rec.sport.hockey"
    autosDirectory ="20news-bydate/20news-bydate/20news-bydate-train/rec.autos"

    #Grab file data into doc list objects
    hockeyDocs=readDocs(hockeyDirectory)
    autosDocs=readDocs(autosDirectory)
    print('training data read')

    #Tokenize the documents




    hockeyDocsAsTokens=docsAsTokens(hockeyDocs)
    #autosDocsAsTokens=docsAsTokens(autosDocs)

    print(hockeyDocsAsTokens[0])




def readDocs(path):
    docAsList=[]
    for doc in os.listdir(path):
        with open(os.path.join(path,doc)) as file:
            docAsList.append(file.read())
            file.close()
    return docAsList

def docsAsTokens(docs):
    nlp = spacy.load("en_core_web_sm")
    docAsToken=[]
    for doc in docs:
        tokens=[]
        spacy_doc=nlp(doc)
        for token in spacy_doc:
            #print(token.text)
            tokens.append(token)
        docAsToken.append(tokens)
    return docAsToken


