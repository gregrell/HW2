import spacy
import os


def perform():
    print('Using SpaCy version', spacy.__version__)
    print("Performing Segmentation")
    hockeyDirectory ="20news-bydate/20news-bydate/20news-bydate-train/rec.sport.hockey"
    autosDirectory ="20news-bydate/20news-bydate/20news-bydate-train/rec.autos"

    hockeyDocs=readDocs(hockeyDirectory)
    autosDocs=readDocs(autosDirectory)
    print('training data read')




def readDocs(path):
    docAsList=[]
    for doc in os.listdir(path):
        with open(os.path.join(path,doc)) as file:
            docAsList.append(file.read())
            file.close()
    return docAsList

