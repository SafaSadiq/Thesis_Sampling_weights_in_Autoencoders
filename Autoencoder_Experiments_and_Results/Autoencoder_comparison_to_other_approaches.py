from Modules.Kernel_Autoencoder import *

dataset_index = 1 #set 0 for mnist and 1 for cifar
plot_images = False #set True to visualize results

#define hyperparameters
dataset = ["mnist", "cifar"]
encoding_dim = [100,650]
similar_samples_size = 2
layer_width = [1000,2000]

name_of_appraoch = ["Our approach", "random sampling Tanh", "random sampling Relu", "iterative training"]
activation_func = ["tanh", "tanh", "relu", "tanh"]
param_sampler = ["tanh", "random", "random", "none"]
decoder_width = [[1000, 1000, 1000, 784], [2000, 2000, 2000, 3072]]
opt = keras.optimizers.Adam(learning_rate=0.01)
callback = keras.callbacks.EarlyStopping(monitor='loss',patience=2)

#read dataset
X_train, Y_train, X_test, Y_test, meanPoint = load_data(dataset[dataset_index], train_size=10000, test_size=2500)
X_train_complete, Y_train_complete, X_test_complete, Y_test_complete, meanPoint_comp = \
    load_data(dataset[dataset_index], train_size=60000, test_size=10000)
image_shape = X_test.shape

#create a classifier and train it using the train set
sampled_head = classification(X_train_complete, Y_train_complete)

for j in range(len(name_of_appraoch)):
    time_start = time()

    if name_of_appraoch[j] == "iterative training":

        #iteratively trained approach
        Model = make_fully_connected_iterative_model(decoder_width[dataset_index][j], layer_width[dataset_index],
                                                     encoding_dim[dataset_index], activation_func[j])

        Model.compile(optimizer='adam', loss='mse')

        history = Model.fit(X_train, X_train,
                                       batch_size=512,
                                       epochs=100,
                                       shuffle=True,
                                       callbacks=[callback])

        # Make prediction to decode the digits
        X_train_reconstruct = Model.predict(X_train)
        X_test_reconstruct = Model.predict(X_test)

    else:

        #sampled approach
        similiarity_graph = find_similiarity_map(N=len(X_train), Y=Y_train, num_similar_samples=similar_samples_size)
        X_train_encoding, X_test_encoding = enocder(X_train, X_test, layer_width=layer_width[dataset_index], activation=activation_func[j], param_sampler=param_sampler[j])
        Za, Za_test = find_embedding(similiarity_graph, X_train_encoding, X_test_encoding, encoding_dim=encoding_dim[dataset_index])
        X_train_reconstruct, X_test_reconstruct = decoder(Za, Za_test, X_train, layer_width=decoder_width[dataset_index][j], activation=activation_func[j], param_sampler=param_sampler[j])

    time_end = time()
    reconstruction_time = time_end - time_start #find reconstruction time

    #find train and test reconstruction error
    train_error = find_loss(X_train, X_train_reconstruct)
    test_error = find_loss(X_test, X_test_reconstruct)

    pred_train_reconstructed = sampled_head.transform(X_train_reconstruct) #run classifier on reconstructed training set
    train_acc_reconstructed = accuracy_score(Y_train, pred_train_reconstructed) #find accuracy of classification reconstructed on train set

    pred_test_reconstructed = sampled_head.transform(X_test_reconstruct) #run classifier on reconstructed test set
    test_acc_reconstructed = accuracy_score(Y_test, pred_test_reconstructed) #find accuracy of classification on reconstructed test set

    #save results to file
    details = {'Dataset': dataset[dataset_index],
               'Approach': name_of_appraoch[j],
               'Train time': reconstruction_time,
               'Train reconstruction error': train_error,
               'Test reconstruction error': test_error,
               'Train Reconstructed Accuracy': train_acc_reconstructed,
               'Test Reconstructed Accuracy': test_acc_reconstructed}

    with open('Autoencoder_Comparisons.txt', 'a') as convert_file:
        convert_file.write(json.dumps(details))
        convert_file.write("\n")

    if plot_images:
        print(name_of_appraoch[j])
        plot_reconstructed_image(X_test_reconstruct, X_test, image_shape, meanPoint, num_images=10)