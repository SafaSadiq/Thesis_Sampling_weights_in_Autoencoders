from Modules.Kernel_Autoencoder import *

dataset_index = 0 #set 0 for mnist and 1 for cifar
plot_images = False #set True to visualize results
dataset = ["mnist", "cifar"]

#read dataset
X_train, Y_train, X_test, Y_test, meanPoint = load_data(dataset[dataset_index], train_size=10000, test_size=2000)
X_train_complete, Y_train_complete, X_test_complete, Y_test_complete, meanPoint_comp = \
    load_data(dataset[dataset_index], train_size=60000, test_size=10000)
image_shape = X_test.shape

#create a classifier and train it using the train set
sampled_head = classification(X_train_complete, Y_train_complete)

'''
TESTING FOR DIFFERENT LAYER WIDTHS
'''

#define hyperparameters
encoding_dim = [200, 650]
similar_samples_size = 2
layer_width = [[500, 1000, 1500, 2000, 3000,3500, 4000, 4500],
               [2000, 2500, 3000, 3500, 4000]]

for width in layer_width[dataset_index]:

    time_start = time()

    #compute reconstruction
    similiarity_graph = find_similiarity_map(N=len(X_train), Y=Y_train, num_similar_samples=similar_samples_size)
    X_train_encoding, X_test_encoding = enocder(X_train, X_test, layer_width=width, activation="tanh", param_sampler="tanh")
    Za, Za_test = find_embedding(similiarity_graph, X_train_encoding, X_test_encoding, encoding_dim=encoding_dim[dataset_index])
    X_train_reconstruct, X_test_reconstruct = decoder(Za, Za_test, X_train, layer_width=width, activation="tanh", param_sampler="tanh")

    time_end = time()
    reconstruction_time = time_end - time_start #find reconstruction time

    #find train and test reconstruction error
    train_error = find_loss(X_train, X_train_reconstruct)
    test_error = find_loss(X_test, X_test_reconstruct)

    pred_train = sampled_head.transform(X_train) #run classifier on training set
    train_acc = accuracy_score(Y_train, pred_train) #find accuracy of classification on train set

    pred_train_reconstructed = sampled_head.transform(X_train_reconstruct) #run classifier on reconstructed training set
    train_acc_reconstructed = accuracy_score(Y_train, pred_train_reconstructed) #find accuracy of classification reconstructed on train set

    pred_test = sampled_head.transform(X_test) #run classifier on test set
    test_acc = accuracy_score(Y_test, pred_test) #find accuracy of classification on test set

    pred_test_reconstructed = sampled_head.transform(X_test_reconstruct) #run classifier on reconstructed test set
    test_acc_reconstructed = accuracy_score(Y_test, pred_test_reconstructed) #find accuracy of classification on reconstructed test set

    #save results to file
    details = {'Dataset': dataset[dataset_index],
               'Encoding Dimension': encoding_dim[dataset_index],
               'Num similiar samples' :similar_samples_size,
               'Layer_width': width,
               'Train time': reconstruction_time,
               'Train reconstruction error': train_error,
               'Test reconstruction error': test_error,
               'Train Accuracy': train_acc,
               'Train Reconstructed Accuracy': train_acc_reconstructed,
               'Test Accuracy': test_acc,
               'Test Reconstructed Accuracy': test_acc_reconstructed}
    with open('Results/Kernel_Autoencoder_Result.txt', 'a') as convert_file:
        convert_file.write(json.dumps(details))
        convert_file.write("\n")

    if plot_images:
        plot_reconstructed_image(X_test_reconstruct, X_test, image_shape, meanPoint, num_images=10)


'''
TESTING FOR DIFFERENT EMBEDDING DIMENSIONS
'''

#define hyperparameters
encoding_dim = [[50, 100, 150, 200, 250, 300],
                [400, 450, 500, 550, 600]]
similar_samples_size = 2
layer_width = [1000, 2000]

for enc in encoding_dim[dataset_index]:

    time_start = time()

    #compute reconstruction
    similiarity_graph = find_similiarity_map(N=len(X_train), Y=Y_train, num_similar_samples=similar_samples_size)
    X_train_encoding, X_test_encoding = enocder(X_train, X_test, layer_width=layer_width[dataset_index], activation="tanh", param_sampler="tanh")
    Za, Za_test = find_embedding(similiarity_graph, X_train_encoding, X_test_encoding, encoding_dim=enc)
    X_train_reconstruct, X_test_reconstruct = decoder(Za, Za_test, X_train, layer_width=layer_width[dataset_index], activation="tanh", param_sampler="tanh")

    time_end = time()
    reconstruction_time = time_end - time_start #find reconstruction time

    #find train and test reconstruction error
    train_error = find_loss(X_train, X_train_reconstruct)
    test_error = find_loss(X_test, X_test_reconstruct)

    pred_train = sampled_head.transform(X_train) #run classifier on training set
    train_acc = accuracy_score(Y_train, pred_train) #find accuracy of classification on train set

    pred_train_reconstructed = sampled_head.transform(X_train_reconstruct) #run classifier on reconstructed training set
    train_acc_reconstructed = accuracy_score(Y_train, pred_train_reconstructed) #find accuracy of classification reconstructed on train set

    pred_test = sampled_head.transform(X_test) #run classifier on test set
    test_acc = accuracy_score(Y_test, pred_test) #find accuracy of classification on test set

    pred_test_reconstructed = sampled_head.transform(X_test_reconstruct) #run classifier on reconstructed test set
    test_acc_reconstructed = accuracy_score(Y_test, pred_test_reconstructed) #find accuracy of classification on reconstructed test set

    #save results to file
    details = {'Dataset': dataset[dataset_index],
               'Encoding Dimension': enc,
               'Num similiar samples' :similar_samples_size,
               'Layer_width': layer_width[dataset_index],
               'Train time': reconstruction_time,
               'Train reconstruction error': train_error,
               'Test reconstruction error': test_error,
               'Train Accuracy': train_acc,
               'Train Reconstructed Accuracy': train_acc_reconstructed,
               'Test Accuracy': test_acc,
               'Test Reconstructed Accuracy': test_acc_reconstructed}
    with open('Results/Kernel_Autoencoder_Result.txt', 'a') as convert_file:
        convert_file.write(json.dumps(details))
        convert_file.write("\n")

    if plot_images:
        plot_reconstructed_image(X_test_reconstruct, X_test, image_shape, meanPoint, num_images=10)


'''
TESTING FOR DIFFERENT NUMBER OF SIMILAR SAMPLES
'''

#define hyperparameters
encoding_dim = [200, 650]
similar_samples_size = [2,6,10,14,18,22]
layer_width = [1000, 2000]

for pp in similar_samples_size:

    time_start = time()

    #compute reconstruction
    similiarity_graph = find_similiarity_map(N=len(X_train), Y=Y_train, num_similar_samples=pp)
    X_train_encoding, X_test_encoding = enocder(X_train, X_test, layer_width=layer_width[dataset_index], activation="tanh", param_sampler="tanh")
    Za, Za_test = find_embedding(similiarity_graph, X_train_encoding, X_test_encoding, encoding_dim=encoding_dim[dataset_index])
    X_train_reconstruct, X_test_reconstruct = decoder(Za, Za_test, X_train, layer_width=layer_width[dataset_index], activation="tanh", param_sampler="tanh")

    time_end = time()
    reconstruction_time = time_end - time_start #find reconstruction time

    #find train and test reconstruction error
    train_error = find_loss(X_train, X_train_reconstruct)
    test_error = find_loss(X_test, X_test_reconstruct)

    pred_train = sampled_head.transform(X_train) #run classifier on training set
    train_acc = accuracy_score(Y_train, pred_train) #find accuracy of classification on train set

    pred_train_reconstructed = sampled_head.transform(X_train_reconstruct) #run classifier on reconstructed training set
    train_acc_reconstructed = accuracy_score(Y_train, pred_train_reconstructed) #find accuracy of classification reconstructed on train set

    pred_test = sampled_head.transform(X_test) #run classifier on test set
    test_acc = accuracy_score(Y_test, pred_test) #find accuracy of classification on test set

    pred_test_reconstructed = sampled_head.transform(X_test_reconstruct) #run classifier on reconstructed test set
    test_acc_reconstructed = accuracy_score(Y_test, pred_test_reconstructed) #find accuracy of classification on reconstructed test set

    #save results to file
    details = {'Dataset': dataset[dataset_index],
               'Encoding Dimension': encoding_dim[dataset_index],
               'Num similiar samples' :pp,
               'Layer_width': layer_width[dataset_index],
               'Train time': reconstruction_time,
               'Train reconstruction error': train_error,
               'Test reconstruction error': test_error,
               'Train Accuracy': train_acc,
               'Train Reconstructed Accuracy': train_acc_reconstructed,
               'Test Accuracy': test_acc,
               'Test Reconstructed Accuracy': test_acc_reconstructed}
    with open('Results/Kernel_Autoencoder_Result.txt', 'a') as convert_file:
        convert_file.write(json.dumps(details))
        convert_file.write("\n")

    if plot_images:
        plot_reconstructed_image(X_test_reconstruct, X_test, image_shape, meanPoint, num_images=10)
