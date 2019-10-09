import numpy as np
from sklearn.naive_bayes import MultinomialNB


def Run():
    print("sklearn ")
    X = np.random.randint(5,size=(6,10)) #6 lists of 100 integers between 0 and 4, this represents 6 classes of data
    y = np.array([1,2,3,4,5,6]) #classes
    clf = MultinomialNB()
    clf.fit(X,y)
    #MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
    print(X)
    print("line")
    print(X[2:3])
    print(clf.predict(X))
    #print(clf.predict_proba(X[2:3]))


