import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from tensorflow.keras.datasets import mnist, cifar10
from scipy.sparse.linalg import lobpcg
from sklearn.metrics import accuracy_score
import json
from time import time
import keras

from sklearn.pipeline import Pipeline
from swimnetworks import Dense, Linear

def load_data(dataset, train_size=20000, test_size=2500):
    #Load dataset
    if dataset == "cifar":
        (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
        Y_train = Y_train[:][:,0]
        Y_test = Y_test[:][:,0]
    elif dataset == "mnist":
        (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    else:
        raise Exception("Incorrest dataset specified")

    #reshape data to size N x (D1xD2xD3)
    X_train = X_train.reshape(len(X_train), np.prod(X_train.shape[1:]))
    X_test = X_test.reshape(len(X_test), np.prod(X_test.shape[1:]))

    #Choose a random subset as the train and test set (because of memory constraints)
    if train_size<60000:
        random_indices = np.random.choice(X_train.shape[0],size=train_size,replace=False)
        random_indices_test = np.random.choice(X_test.shape[0],size=test_size,replace=False)

        X_train, Y_train = X_train[random_indices, :], Y_train[random_indices]
        X_test, Y_test = X_test[random_indices_test, :], Y_test[random_indices_test]

    #Normalize data
    X_train = X_train/255
    X_test = X_test/255

    #Center Data
    meanPoint = X_train.mean(axis = 0)
    X_train -= meanPoint
    X_test -= meanPoint

    return X_train, Y_train, X_test, Y_test, meanPoint


def find_similiarity_map(N, Y, num_similar_samples=5):
    # Initialize graph
    G = np.zeros((N,N))

    #Set value 1 for random num_similar_samples points for each point
    for i in range(0, N, 1):
        list_labels = [j for j, value in enumerate(Y) if value == Y[i]]
        chosen_labels = random.sample(list_labels, num_similar_samples)
        for k in chosen_labels:
            G[i,k] = 1

    return G

def enocder(X, X_t, layer_width, activation="tanh", param_sampler="tanh"):
    #create encoding pipeline for feature map
    enc0 = Dense(layer_width=layer_width, activation=activation, parameter_sampler=param_sampler, random_seed=80)
    steps_enc = [("enc0", enc0)]
    encode_model = Pipeline(steps_enc)

    #fit and transform train and test data to the encoder
    #returns encoding of shape N x layer_width
    X_train_encoding = encode_model.fit_transform(X, X)
    X_test_encoding = encode_model.transform(X_t)

    return X_train_encoding, X_test_encoding

def find_embedding(Graph, X_train_enc, X_test_enc, encoding_dim):
    row_indices, col_indices = Graph.nonzero()
    # X=[abcd] with a and b having same labels and c and having same labels: Xa= [aabbccdd], Xb = [ababcdcd]
    Xa = X_train_enc[row_indices]
    Xb = X_train_enc[col_indices]

    Da = Xa.shape[1]
    Db = Xb.shape[1]
    gamma = 0.001

    Caa = Xa.T @ Xa + np.eye(Da) * gamma # Xa ∈ RN,Da regularized with gamma (γ)
    Cbb = Xb.T @ Xb + np.eye(Db) * gamma # Xb ∈ RN,Db regularized with gamma (γ)
    RH = np.block([[Caa, np.zeros_like(Caa)], [np.zeros_like(Cbb), Cbb]]) # preparing the generalized eigenvalue prob.

    # Assuming Xa and Xb are numpy arrays
    Xa_T_Xb = Xa.T @ Xb + np.eye(Da) * gamma
    Xb_T_Xa = Xb.T @ Xa + np.eye(Db) * gamma

    # Construct the block diagonal matrix
    # [Xa_T_Xb 0
    #  0       Xb_T_Xa]
    LH = np.block([[Xa_T_Xb, np.zeros_like(Xa_T_Xb)], [np.zeros_like(Xb_T_Xa), Xb_T_Xa]])

    # Roll the matrix
    LH = np.roll(LH, Da, axis=1)

    # Initial guess for eigenvectors (random)
    X_initial_guess = np.random.rand(LH.shape[0], encoding_dim)

    # Compute the largest K eigenvalues and corresponding eigenvectors
    eigenvalues, eigenvectors = lobpcg(LH, X=X_initial_guess, B=RH, largest=True, maxiter=-1, tol=0.01)

    # Select K largest eigenvalues and corresponding eigenvectors
    sorted_indices = np.argsort(-eigenvalues)  # Sort eigenvalues in descending order
    sorted_indices = sorted_indices[:encoding_dim]  # Select K largest eigenvalues
    # largest_eigenvalues = eigenvalues[sorted_indices]  # Get largest eigenvalues
    largest_eigenvectors = eigenvectors[:, sorted_indices]  # Get corresponding eigenvectors

    Wa, Wb =largest_eigenvectors[:len(largest_eigenvectors)//2], largest_eigenvectors[len(largest_eigenvectors)//2:]
    Za = X_train_enc @ Wa
    Za_test = X_test_enc @ Wa

    return Za, Za_test

def decoder(X_emb, X_test_emb, X, layer_width, activation="tanh", param_sampler="tanh"):
    #create decoding pipeline for reconstruction
    dec0 = Dense(layer_width=layer_width, activation=activation, parameter_sampler=param_sampler, random_seed=76)
    linear = Linear(is_classifier=False, layer_width=X.shape, regularization_scale=1e-10)

    steps_dec = [
        ("dec0", dec0),
        ("linear", linear)
    ]
    decode_model = Pipeline(steps_dec)

    #fit and transform train and test data to the decoder
    #returns reconstructed image of same shape
    X_train_reconstruct = decode_model.fit_transform(X_emb, X)
    X_test_reconstruct = decode_model.transform(X_test_emb)

    return X_train_reconstruct, X_test_reconstruct

def find_loss(sample, prediction):

    return mean_squared_error(sample, prediction)

def plot_reconstructed_image(reconstructed_image, original_image, shape, meanPoint, num_images):
    reconstructed_image += meanPoint
    original_image += meanPoint

    points_test_decoding = reconstructed_image.reshape(shape)
    points_test = original_image.reshape(shape)

    for i in range(num_images):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2, 2))
        ax1.imshow(points_test_decoding[i], cmap='gray')
        ax2.imshow(points_test[i], cmap='gray')
        plt.show()

def save_images_to_file(filename, data):
    with open(filename, 'a') as outfile:
        np.savetxt(outfile, data, fmt='%-7.2f')


def classification(X, Y):
    # Head of the neural network (1 hidden layer + 1 output layer)
    steps = [
        ("fcn1",
         Dense(layer_width=10000, is_classifier=True,
               activation="tanh",
               parameter_sampler="tanh",
               random_seed=50)),
        ("lin", Linear(layer_width=len(Y), is_classifier=True,regularization_scale=1e-10))
    ]

    sampled_head = Pipeline(steps=steps,)
    history_snn = sampled_head.fit(X, Y)

    return sampled_head

def make_fully_connected_iterative_model(shape, encoding_dim, emb_dim, act):
    InputModel = keras.layers.Input(shape=(shape,)) #shape=(32*32*3,)
    EncodedLayer = keras.layers.Dense(encoding_dim, activation=act)(InputModel)
    EncodedLayer = keras.layers.Dense(emb_dim, activation=act)(EncodedLayer)
    DecodedLayer = keras.layers.Dense(shape, activation=act)(EncodedLayer)

    AutoencoderModel = keras.models.Model(InputModel, DecodedLayer)

    return AutoencoderModel
