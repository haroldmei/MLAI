import numpy as np
import matplotlib.pyplot as plt

def readData(images_file, labels_file):
    x = np.loadtxt(images_file, delimiter=',')
    y = np.loadtxt(labels_file, delimiter=',')
    return x, y

def writeData(fname, X):
    np.savetxt(fname, X, delimiter=',')
    return

def softmax(x):
    """
    Compute softmax function for input. 
    Use tricks from previous assignment to avoid overflow
    """
	### YOUR CODE HERE
    b,n = x.shape
    s = np.array([1 / np.sum(np.exp(x.T - x.T[i]), axis = 0) for i in range(n)])
	### END YOUR CODE
    return s.T

def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    """
    ### YOUR CODE HERE
    s = np.zeros_like(x, dtype=float)
    pos = (x > 0)
    neg = (x <= 0)
    s[pos] = 1 / (1 + np.exp(-x[pos]))
    s[neg] = np.exp(x[neg]) / (1 + np.exp(x[neg]))
    ### END YOUR CODE
    return s

def forward_prop(data, labels, params):
    """
    return hidder layer, output(softmax) layer and loss
    """
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']
  
    ### YOUR CODE HERE
    z1 = data.dot(W1) + b1
    h = sigmoid(z1)
    z2 = h.dot(W2) + b2
    y = softmax(z2)
    cost = np.sum(labels * np.log(y), axis = 1)
    ### END YOUR CODE
    return h, y, cost

def backward_prop(data, labels, params):
    """
    return gradient of parameters
    """
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']
  
    ### YOUR CODE HERE
    batch,l = data.shape
    a1,yhat,cost = forward_prop(data, labels, params)
    # tricky place, the derivation of softmax(x) should be (yhat - y)
    gradb2 = [(yhat[i] - labels[i]) for i in range(batch)]                      # 10 * 1
    gradW2 = [np.matrix(gradb2[i]).T * np.matrix(a1[i]) for i in range(batch)]  # 10 * 300
    gradb1 = [W2.dot(gradb2[i]) * a1[i] * (1 - a1[i]) for i in range(batch)]    # 300 * 1
    gradW1 = [np.matrix(gradb1[i]).T * np.matrix(data[i]) for i in range(batch)]  # 300 * 784
  
    gradb2, gradW2, gradb1, gradW1 = np.mean(gradb2,axis=0), np.mean(gradW2,axis=0), np.mean(gradb1,axis=0), np.mean(gradW1,axis=0)
    ### END YOUR CODE
  
    grad = {}
    grad['W1'] = gradW1.T
    grad['W2'] = gradW2.T
    grad['b1'] = gradb1.T
    grad['b2'] = gradb2.T
  
    return grad

def nn_train(trainData, trainLabels, devData, devLabels):
    (m, n) = trainData.shape
    m, K = trainLabels.shape
    num_hidden = 300
    learning_rate = 5
    params = {}
  
    ### YOUR CODE HERE
    params['W1'] = np.random.standard_normal((n, num_hidden))
    params['b1'] = np.zeros((1, num_hidden), dtype=float)
    params['W2'] = np.random.standard_normal((num_hidden, K))
    params['b2'] = np.zeros((1, K), dtype=float)
  
    batch = 1000
    numEpoc = 30
    numIter = m / batch
    costTrain = np.zeros(numEpoc)
    costDev = np.zeros(numEpoc)
    accurateTrain = np.zeros(numEpoc)
    accurateDev = np.zeros(numEpoc)
    for i in range(numEpoc):
        print "Epoc %d"%i
        for j in range(numIter):
            print "Iter %d"%j
            batch_data = trainData[j * batch: (j + 1) * batch]
            batch_labels = trainLabels[j * batch: (j + 1) * batch]
            grad = backward_prop(batch_data, batch_labels, params)
            params['W1'] = params['W1'] - grad['W1'] * learning_rate
            params['b1'] = params['b1'] - grad['b1'] * learning_rate
            params['W2'] = params['W2'] - grad['W2'] * learning_rate
            params['b2'] = params['b2'] - grad['b2'] * learning_rate
        #
        trainCost = forward_prop(trainData, trainLabels, params)
        devCost = forward_prop(devData, devLabels, params)
        costTrain[i] = np.mean(trainCost[2])
        costDev[i] = np.mean(devCost[2])
        accurateTrain[i] = np.mean(nn_test(trainData, trainLabels, params))
        accurateDev[i] = np.mean(nn_test(devData, devLabels, params))
    plotData(costTrain, costDev, accurateTrain, accurateDev)
    ### END YOUR CODE
  
    return params

def plotData(tr_loss, dev_loss, tr_metric, dev_metric):
    num_epochs = 30
    xs = np.arange(num_epochs)
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=False, figsize=(12, 4))
    ax0, ax1 = axes.ravel()
  
    ax0.plot(xs, tr_loss, label='train loss')
    ax0.plot(xs, dev_loss, label='dev loss')
    ax0.legend()
    ax0.set_xlabel('# epoch')
    ax0.set_ylabel('CE loss')
  
    ax1.plot(xs, tr_metric, label='train acc')
    ax1.plot(xs, dev_metric, label='dev acc')
    ax1.legend()
    ax1.set_xlabel('# epoch')
    ax1.set_ylabel('Accuracy')
    plt.show()

def nn_test(data, labels, params):
    h, output, cost = forward_prop(data, labels, params)
    accuracy = compute_accuracy(output, labels)
    return accuracy

def compute_accuracy(output, labels):
    accuracy = (np.argmax(output,axis=1) == np.argmax(labels,axis=1)).sum() * 1. / labels.shape[0]
    return accuracy

def one_hot_labels(labels):
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size),labels.astype(int)] = 1
    return one_hot_labels

def main():
    np.random.seed(100)
    trainData, trainLabels = readData('images_train.csv', 'labels_train.csv')
    trainLabels = one_hot_labels(trainLabels)
    p = np.random.permutation(60000)
    trainData = trainData[p,:]
    trainLabels = trainLabels[p,:]

    devData = trainData[0:10000,:]
    devLabels = trainLabels[0:10000,:]
    trainData = trainData[10000:,:]
    trainLabels = trainLabels[10000:,:]

    mean = np.mean(trainData)
    std = np.std(trainData)
    trainData = (trainData - mean) / std
    devData = (devData - mean) / std

    testData, testLabels = readData('images_test.csv', 'labels_test.csv')
    testLabels = one_hot_labels(testLabels)
    testData = (testData - mean) / std
	
    params = nn_train(trainData, trainLabels, devData, devLabels)


    readyForTesting = True
    if readyForTesting:
        accuracy = nn_test(testData, testLabels, params)
	print 'Test accuracy: %f' % accuracy

if __name__ == '__main__':
    main()
