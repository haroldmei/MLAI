from matplotlib.image import imread
from heapq import nsmallest
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# reduce to n colors
def kmeans_compress(X, n = 16):
    X = X.reshape(-1, 3)
    centroids = X[np.random.choice(np.arange(X.shape[0]), size=n, replace=False)]
    centers = np.zeros([n, 3])
    #
    while not np.array_equal(centers, centroids):
        centers = centroids
        norms = [np.sqrt(np.sum((centers[i] - X) ** 2, axis=1)) for i in range(0, len(centers))]
        idxs = np.stack(norms).argmin(axis=0)
        #recalc centroids
        centroids = np.array([np.mean(X[np.where(idxs == i,True,False)], axis = 0).round() for i in range(n)])
    #you need the data type conversion, otherwise you are in big trouble!!
    return np.uint8(centroids), np.uint8(idxs)

def show_img(X, ax, s, size):
    ax.imshow(X)
    #size = pd.DataFrame(X.reshape(-1,3)).drop_duplicates().shape
    ax.set_title(s % np.product(size))

def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 4))
    axa1, axb1, axa2, axb2 = axes.ravel()

    A1 = imread('mandrill-small.tiff')
    centroids,idxs = kmeans_compress(A1)
    B1 = np.array([centroids[idxs[i]] for i in range(0, len(idxs))])
    B1 = B1.reshape(A1.shape)
    show_img(A1, axa1, 'original mandrill-small, size=%d', A1.size)
    show_img(B1, axb1,'compressed mandrill-small, size=%d', centroids.size + idxs.size)

    A2 = imread('mandrill-large.tiff')
    centroids,idxs = kmeans_compress(A2)
    B2 = np.array([centroids[idxs[i]] for i in range(0, len(idxs))])
    B2 = B2.reshape(A2.shape)
    show_img(A2, axa2, 'original mandrill-small, size=%d', A2.size)
    show_img(B2, axb2, 'compressed mandrill-small, size=%d', centroids.size + idxs.size)

    plt.show()

if __name__ == '__main__':
    main()