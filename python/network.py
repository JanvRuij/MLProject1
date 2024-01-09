import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD
from keras.initializers import Constant, GlorotNormal, GlorotUniform
from keras.regularizers import L1, L2
from sklearn.model_selection import train_test_split

# read the data
X = np.genfromtxt("WisconsinBreastCancerData/X.csv", delimiter=",")
Y = np.genfromtxt("WisconsinBreastCancerData/Y.csv", delimiter=",")

# generate training, testing and validation sets
X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train, test_size=0.25, random_state=42)

# hyperparemeters
nr_nodes = [5, 10, 15, 20, 25, 35, 50]
batch_sizes = [30, 25, 20, 15, 10, 5]
regularizations = [L2, L1]
regularization_rates = [0.1, 0.05, 0.01, 0.005, 0.001]
opt_algorithms = [Adam, SGD]
learning_rates = [0.001, 0.002, 0.005, 0.01, 0.05, 0.1]
initializations = [Constant, GlorotNormal, GlorotUniform]
nr_tested = 0

initialization = initializations[1]
opt_algorithm = opt_algorithms[0]
regularization = regularizations[0]
while nr_tested < 5:
    nodes = nr_nodes[nr_tested]
    batch_size = batch_sizes[nr_tested]
    regularization_rate = regularization_rates[nr_tested]
    learning_rate = learning_rates[nr_tested]

    # create the model
    model = Sequential()
    model.add(Dense(nodes, activation='relu',
                    kernel_initializer=initialization,
                    kernel_regularizer=regularization(regularization_rate)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=opt_algorithm(learning_rate=learning_rate),
                  loss="BinaryCrossentropy")

    # train the model
    history = model.fit(X_train, Y_train, epochs=25, batch_size=batch_size)
    train_loss = float(history.history["loss"][-1])
    # validate the model
    val_loss = model.evaluate(X_val, Y_val)

    print(f"Training Loss: {train_loss}")
    print(f"Validation Loss: {val_loss}")

    nr_tested += 1
