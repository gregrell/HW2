import spacy
import os


def perform():
    print('Using SpaCy version', spacy.__version__)
    print("Performing Segmentation")
    hockeyDirectory ="20news-bydate/20news-bydate/20news-bydate-train/rec.sport.hockey"
    autosDirectory ="20news-bydate/20news-bydate/20news-bydate-train/rec.autos"
    hockeyDocs=os.listdir(hockeyDirectory)
    autosDocs=os.listdir(autosDirectory)
    for doc in hockeyDocs:
        print(doc)
