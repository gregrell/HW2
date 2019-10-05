import numpy as np
from linear_classifier import LinearClassifier


class MultinomialNaiveBayes(LinearClassifier):

    def __init__(self):
        LinearClassifier.__init__(self)
        self.trained = False
        self.likelihood = 0
        self.prior = 0
        self.smooth = False
        self.smooth_param = 1

    def train(self, x, y):
        # n_docs = no. of documents
        # n_words = no. of unique words
        n_docs, n_words = x.shape

        # classes = a list of possible classes
        classes = np.unique(y)

        # n_classes = no. of classes
        n_classes = np.unique(y).shape[0]

        # initialization of the prior and likelihood variables
        prior = np.zeros(n_classes)
        likelihood = np.zeros((n_words, n_classes))

        # TODO: This is where you have to write your code!
        # You need to compute the values of the prior and likelihood parameters
        # and place them in the variables called "prior" and "likelihood".
        # Examples:
        # prior[0] is the prior probability of a document being of class 0
        # likelihood[4, 0] is the likelihood of the fifth(*) feature being
        # active, given that the document is of class 0
        # (*) recall that Python starts indices at 0, so an index of 4
        # corresponds to the fifth feature!

        ###########################

        # YOUR CODE HERE
        # RELL - Numpy Working
        print
        'Classes: %s, Docs: %d, Words: %d' % (classes, n_docs, n_words)

        # Initialize Number of Docs In Class Array
        numDocsInClass = [0] * n_classes

        # for each class, find how many matches there are
        # in the y array, increment the numDocsInClass array at that
        # class index.

        for currentClass in classes:
            for i in y:
                if y[i][0] == currentClass:
                    numDocsInClass[currentClass] += 1
        print
        'there are ', numDocsInClass[0], ' docs in class 0'
        print
        'there are ', numDocsInClass[1], ' docs in class 1'

        # Set the prior parameters
        prior[0] = float(numDocsInClass[0]) / n_docs
        prior[1] = float(numDocsInClass[1]) / n_docs

        print
        'Prior is ', prior

        # print x.shape[1] # numpy shape = returns the size of an array by dimension array.shape[0] rows, array.shape[1]
        # columns
        # the x,y data is represented as follows: all the documents were dissected into a bag of words.
        # The vocabulary is 13989 words and there are 1600 total documents. These documents are either class
        # positive or negative and in those 1600 rows they are mixed at random.
        # Each row represents a document, and each column represents the number of times that particular word exists
        # in that document. That's it. It's a 2D array of statistics.

        # Count the number of words in the classes by word. This means for each word index count the
        # number of times that particular word exists within that class.
        countOfWordInClass = np.zeros((n_words, n_classes))
        i = 0
        for word in range(n_words):
            for doc in range(n_docs):
                countOfWordInClass[word, y[doc][0]] += x[doc, word]
                i += 1
                if i % 1000000 == 0:
                    print('\r',float(i) / (n_words * n_docs) * 100, '% done',end="")
        print("")


        # Count the total number of words within each class.
        totalWordsClass = [0, 0]

        for doc in range(n_docs):
            if y[doc][0] == 0:
                totalWordsClass[0] += np.sum(x[doc])
            elif y[doc][0] == 1:
                totalWordsClass[1] += np.sum(x[doc])
        print
        'The total number of words in class 0 is ', totalWordsClass[0], ' Total for class 1 is ', totalWordsClass[1]

        # Targets: Accuracy on training set: 0.985625, on test set: 0.687500.
        # In order to achieve EXACT accuracy as described in HW1 the number of words in the vocabulary
        # has been omitted from the denominator. This takes on equation 4.12 from the textbook
        # Only add the total number of words in dictionary in denominator when using laplace smoothing.
        for word in range(n_words):
            likelihood[word, 0] = float(countOfWordInClass[word, 0]) / (totalWordsClass[0])
            likelihood[word, 1] = float(countOfWordInClass[word, 1]) / (totalWordsClass[1])

        ###########################

        params = np.zeros((n_words + 1, n_classes))
        for i in range(n_classes):
            # log probabilities
            params[0, i] = np.log(prior[i])
            with np.errstate(divide='ignore'):  # ignore warnings
                params[1:, i] = np.nan_to_num(np.log(likelihood[:, i]))
        self.likelihood = likelihood
        self.prior = prior
        self.trained = True
        return params
