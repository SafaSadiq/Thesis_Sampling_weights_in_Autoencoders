from sklearn import datasets
from random import randint
import numpy as np
from sklearn import manifold
from sklearn import decomposition
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from swimnetworks import Dense, Linear
from sklearn.metrics import mean_squared_error
from keras.datasets import mnist, fashion_mnist
from sklearn import manifold


def create_test_train_sets(dataset="Swiss Roll", n_samples_train=2000, n_samples_test=400, hole=False, dimensions=100):
    if dataset == "mnist" or dataset == "fashion mnist":
        if dataset == "mnist":
            (points_train, color_train), (points_test, color_test) = mnist.load_data()
        else:
            (points_train, color_train), (points_test, color_test) = fashion_mnist.load_data()
        points_train = points_train.astype("float")

        points_train = points_train.reshape(len(points_train), np.prod(points_train.shape[1:]))
        points_test = points_test.reshape(len(points_test), np.prod(points_test.shape[1:]))

        mean_train_x = points_train.mean().astype(np.float32)
        std_train_x = points_train.std().astype(np.float32)
        points_train = (points_train - mean_train_x)/(std_train_x)
        points_test = (points_test - mean_train_x)/(std_train_x)

    if dataset == "Swiss Roll":
        points_train, color_train = datasets.make_swiss_roll(n_samples=n_samples_train, random_state=24, hole=hole)
        points_test, color_test = datasets.make_swiss_roll(n_samples=n_samples_test, random_state=24, hole= hole)

    if dataset == "S Curve":
        points_train, color_train = datasets.make_s_curve(n_samples=n_samples_train, random_state=randint(0,1000))
        points_test, color_test = datasets.make_s_curve(n_samples=n_samples_test, random_state=randint(0,1000))

    if dimensions > points_train.shape[1]:

        arr0 = np.zeros((points_train.shape[0], dimensions-points_train.shape[1]))
        points_train = np.concatenate((points_train,arr0),axis=1)

        arr1 = np.zeros((points_test.shape[0], dimensions-points_test.shape[1]))
        points_test = np.concatenate((points_test,arr1),axis=1)

    return points_train, color_train, points_test, color_test


def create_embedding(points_train, color_train, neighbours=12, components=2, method="PCA", plot=False):
    params = {
        "n_neighbors": neighbours,
        "n_components": components,
        "eigen_solver": "auto",
        "random_state": 0,
    }
    if method == "LLE":
        lle_standard = manifold.LocallyLinearEmbedding(method="hessian", **params)
        embedding = lle_standard.fit_transform(points_train)

    if method == "PCA":
        pca = decomposition.PCA(n_components = components)
        embedding = pca.fit_transform(points_train)

    if method == "Isomap":
        isomap = manifold.Isomap(n_neighbors=neighbours, n_components=components, p=1)
        embedding = isomap.fit_transform(points_train)

    if method == "Spectral Embedding":
        spectral = manifold.SpectralEmbedding(n_components=components, n_neighbors=neighbours, random_state=42)
        embedding = spectral.fit_transform(points_train)

    if method == "TSNE":
        t_sne = manifold.TSNE(n_components=components,perplexity=30,init="random",n_iter=250,random_state=0)
        embedding = t_sne.fit_transform(points_train)

    if method == "perfect embedding":
        embedding = np.vstack((color_train, points_train[:,1])).T

    if plot:
        plt.scatter(embedding[:, 0], embedding[:, 1], c=color_train)
        plt.show()

    return embedding


def plot_latent_space(encoding, color_encoding):
    plt.scatter(encoding[:, 0], encoding[:, 1], c=color_encoding)
    plt.title("Latent space embedding")
    plt.show()

def plot_reconstruction(decoding, color_decoding):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    fig.add_axes(ax)
    ax.scatter(
        decoding[:, 0], decoding[:, 1], decoding[:, 2], c=color_decoding, s=50, alpha=0.8
    )
    ax.set_title("Reconstruction in Ambient Space")
    ax.view_init(azim=-66, elev=12)

    plt.show()

def reconstrction_error(sample, prediction):

    return mean_squared_error(sample, prediction)










def Create_test_train_sets(dataset="Swiss Roll", n_samples_train=2000, n_samples_test=400, hole=False, dimensions=100):
    rand = np.random.choice(n_samples_train, n_samples_test)
    noise = np.random.normal(0,0.005, (n_samples_test, 3))

    if dataset == "Swiss Roll":
        points_train, color_train = datasets.make_swiss_roll(n_samples=n_samples_train, random_state=24, hole=hole)
        points_test, color_test = points_train[rand], color_train[rand]
        points_test += noise

        # points_test, color_test = datasets.make_swiss_roll(n_samples=n_samples_test, random_state=24, hole= hole)

    if dataset == "S Curve":
        points_train, color_train = datasets.make_s_curve(n_samples=n_samples_train, random_state=randint(0,1000))
        points_test, color_test = points_train[rand], color_train[rand]
        points_test += noise
        # points_test, color_test = datasets.make_s_curve(n_samples=n_samples_test, random_state=randint(0,1000))

    if dimensions > points_train.shape[1]:

        arr0 = np.zeros((points_train.shape[0], dimensions-points_train.shape[1]))
        points_train = np.concatenate((points_train,arr0),axis=1)

        arr1 = np.zeros((points_test.shape[0], dimensions-points_test.shape[1]))
        points_test = np.concatenate((points_test,arr1),axis=1)

    return points_train, color_train, points_test, color_test
