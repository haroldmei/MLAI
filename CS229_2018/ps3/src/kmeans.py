from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np

# reduce to n colors
def kmeans_compress(X, n = 16):
    X = X.reshape(-1, 3)
    centroids = X[np.random.choice(np.arange(X.shape[0]), size=n, replace=False)]
    centers = np.zeros([n, 3])
    #
    while not np.array_equal(centers, centroids):
        centers = centroids
        norms = [np.sqrt(np.sum((centers[i] - X) ** 2, axis=1)) for 
            i in range(0, len(centers))]
        idxs = np.stack(norms).argmin(axis=0)
        #recalc centroids
        centroids = np.array([np.mean(X[np.where(idxs == i,True,False)], 
            axis = 0).round() for i in range(n)])
    #you need the data type conversion, otherwise you are in big trouble!!
    return np.uint8(centroids), np.uint8(idxs)

def show_img(X, ax, s):
    ax.imshow(X)
    ax.set_title(s)

def main():
    fig, axes = plt.subplots(2, 1, figsize=(12, 4))
    axb1, axb2 = axes.ravel()

    A1 = imread('..\data\peppers-small.tiff')
    centroids,idxs = kmeans_compress(A1)
    B1 = np.array([centroids[idxs[i]] for i in range(0, len(idxs))])
    B1 = B1.reshape(A1.shape)
    show_img(B1, axb1,'compressed peppers-small')

    A2 = imread('..\data\peppers-large.tiff')
    centroids,idxs = kmeans_compress(A2)
    B2 = np.array([centroids[idxs[i]] for i in range(0, len(idxs))])
    B2 = B2.reshape(A2.shape)
    show_img(B2, axb2, 'compressed peppers-large')

    plt.show()

if __name__ == '__main__':
    main()