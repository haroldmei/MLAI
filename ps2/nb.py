import numpy as np
from heapq import nlargest

def readMatrix(file):
    fd = open(file, 'r')
    hdr = fd.readline()
    rows, cols = [int(s) for s in fd.readline().strip().split()]
    tokens = fd.readline().strip().split()
    matrix = np.zeros((rows, cols))
    Y = []
    for i, line in enumerate(fd):
        nums = [int(x) for x in line.strip().split()]
        Y.append(nums[0])
        kv = np.array(nums[1:])
        k = np.cumsum(kv[:-1:2])
        v = kv[1::2]
        matrix[i, k] = v
    return matrix, tokens, np.array(Y)

def nb_train(matrix, category):
    state = {}
    N = matrix.shape[1]
    ###################
    state['prior'] = sum(1 if x == 1 else 0 for x in category) / N
    state['post1'] = (sum(a if b == 1 else 0 for a, b in zip(matrix, category)) + 1) / \
        (sum(sum(a) if b == 1 else 0 for a, b in zip(matrix, category)) + N)
    state['post0'] = (sum(a if b == 0 else 0 for a, b in zip(matrix, category)) + 1) / \
        (sum(sum(a) if b == 0 else 0 for a, b in zip(matrix, category)) + N)
    ###################
    return state

def nb_test(matrix, state):
    output = np.zeros(matrix.shape[0])
    ###################
    prior1 = state['prior']
    logsum1 = np.sum(np.log(state['post1']) * matrix, axis = 1)
    logsum0 = np.sum(np.log(state['post0']) * matrix, axis = 1)
    p1 = prior1 / (prior1 + np.exp(logsum0 - logsum1) * (1 - prior1))
    output = np.array([1 if i > 0.5 else 0 for i in p1])
    ###################
    return output

def top_indicative(tokenlist, state, n):
    ss = state['post1']/state['post0']
    indexes = range(0, len(tokenlist))
    ids = nlargest(n, indexes, key=lambda i: ss[i])
    top = np.array([tokenlist[i] for i in ids])
    return top

def evaluate(output, label):
    error = (output != label).sum() * 1. / len(output)
    print('Error: %1.4f' % error)

def main():
    trainMatrix, tokenlist, trainCategory = readMatrix('MATRIX.TRAIN')
    testMatrix, tokenlist, testCategory = readMatrix('MATRIX.TEST')

    state = nb_train(trainMatrix, trainCategory)
    output = nb_test(testMatrix, state)

    evaluate(output, testCategory)

    ss = state['post1']/state['post0']
    indexes = range(0, trainMatrix.shape[1])
    ids = nlargest(3, indexes, key=lambda i: ss[i])
    top3 = np.array([tokenlist[i] for i in ids])
    return

if __name__ == '__main__':
    main()
