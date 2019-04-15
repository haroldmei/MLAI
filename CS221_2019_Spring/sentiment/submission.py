#!/usr/bin/python

import random
import collections
import math
import sys
from util import *

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    return collections.Counter(x.split())
    # END_YOUR_CODE

############################################################
# Problem 3b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.
    You should implement stochastic gradient descent.
    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    '''
    weights = {}  # feature => weight
    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
    # Preprocess it first
    numExamples = len(trainExamples)
    X = []
    Y = []
    for i in range(numExamples):
        phi_x = featureExtractor(trainExamples[i % numExamples][0])
        y = trainExamples[i % numExamples][1]
        X.append(phi_x)
        Y.append(y)

    while numIters > 0:
        for i in range(numExamples):
            phi_x = X[i]
            y = Y[i]

            # w.dot(phi_x)
            score = dotProduct(weights, phi_x)#sum(weights[comp] * phi_x[comp] for comp in set(phi_x.keys()))
            margin = score * y

            if margin <= 1:
                increment(weights, y*eta, phi_x)
            
        print("Current loss ", max(0, 1 - margin), numIters)
        numIters -= 1
    # END_YOUR_CODE
    return weights

############################################################
# Problem 3c: generate test case

def generateDataset(numExamples, weights):
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)
    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a nonzero score under the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    def generateExample():
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        phi = {}
        for comp in set(weights.keys()): phi[comp] = random.randint(0,100)

        score = dotProduct(weights, phi)
        y = -1 if score < 0 else 1

        # END_YOUR_CODE
        return (phi, y)

    return [generateExample() for _ in range(numExamples)]

############################################################
# Problem 3e: character features

def extractCharacterFeatures(n):
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x):
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        xx = x.replace(' ', '')
        length = len(xx)
        features = {}
        for i in range(length - n + 1):
            cur = xx[i : i + n]
            features[cur] = features[cur] + 1 if cur in features else 1
            
        return features
        # END_YOUR_CODE
    return extract

############################################################
# Problem 4: k-means
############################################################


def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE (our solution is 32 lines of code, but don't worry if you deviate from this)
    numExamples = len(examples)
    random.seed(42)
    centroids = [item.copy() for item in random.sample(examples, K)] # examples[0: K]
    assignments = []
    theTuples = []
    loss = 0
    for ii in range(maxIters):
        norms = [[sum((centroids[i][comp] - examples[j][comp]) ** 2 for comp in (set(centroids[i].keys()) | set(examples[j].keys()))) for j in range(numExamples)] for i in range(K)]
        
        theTuples = zip(*norms)
        assignments = [theTuples[i].index(min(theTuples[i])) for i in range(numExamples)]
        idxs = [[j for j,x in enumerate(assignments) if x == i] for i in range(K)]

        loss_ = sum(min(theTuples[i]) for i in range(numExamples))
        if abs(loss - loss_)/numExamples < 1e-6:
            break
        loss = loss_

        #recalc centroids
        for i in range(K):
            if len(idxs[i]) > 0:
                for comp in centroids[i].keys(): 
                    total = sum(examples[j][comp] if comp in examples[j] else 0.0 for j in idxs[i]) + 0.0
                    centroids[i][comp] = total / len(idxs[i]) 

    return centroids, assignments, loss
    # END_YOUR_CODE


def test4b3():
    random.seed(42)
    K = 6
    bestCenters = None
    bestAssignments = None
    bestTotalCost = None
    examples = generateClusteringExamples(numExamples=10000, numWordsPerTopic=3, numFillerWords=10000)
    centers, assignments, totalCost = kmeans(examples, K, maxIters=100)

test4b3()

def test4b0():
    random.seed(42)
    x1 = {0:0, 1:0}
    x2 = {0:0, 1:1}
    x3 = {0:0, 1:2}
    x4 = {0:0, 1:3}
    x5 = {0:0, 1:4}
    x6 = {0:0, 1:5}
    x7 = {0:0, 1:0.5}
    examples = [x1, x2, x3, x4, x5, x6, x7]
    centers, assignments, totalCost = kmeans(examples, 2, maxIters=10)

#test4b0()