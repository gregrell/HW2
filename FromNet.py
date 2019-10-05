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
        # Calculating the Prior Probabilities for the classes
        cat, cat_cnt = np.unique(y, return_counts=True)
        prior[0] = float(cat_cnt[0]) / n_docs
        prior[1] = float(cat_cnt[1]) / n_docs
        prior[2] = float(cat_cnt[2]) / n_docs

        g_count, a_count, n_count = np.zeros(n_words), np.zeros(n_words), np.zeros(n_words)
        # examining each word and finding the above mentioned values
        for c_w in range(x.shape[1]):
            for c_d in range(x.shape[0]):
                if y[c_d] == 0:  # if graphics category
                    g_count[c_w] += x[c_d, c_w]
                elif y[c_d] == 1:  # if autos category
                    a_count[c_w] += x[c_d, c_w]
                else:
                    n_count[c_w] += x[c_d, c_w]
        # Finding likelihood for each word with respective to a class
        for c_w in range(x.shape[1]):
            likelihood[c_w, 0] = float(g_count[c_w] + 1) / (np.sum(g_count) + n_words)
            likelihood[c_w, 1] = float(a_count[c_w] + 1) / (np.sum(a_count) + n_words)
            likelihood[c_w, 2] = float(n_count[c_w] + 1) / (np.sum(n_count) + n_words)
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