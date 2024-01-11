import numpy as np
import random
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
nr_nodes = [40, 80, 160, 240, 350]
batch_sizes = [60, 30, 15, 7, 3]
learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]
regularization_rates = [0.1, 0.05, 0.01, 0.005, 0.03, 0.01]
everything = [nr_nodes, batch_sizes, learning_rates, regularization_rates]
regularizations = [L2, L1]
opt_algorithms = [Adam, SGD]
initializations = [Constant, GlorotNormal, GlorotUniform]

# to keep track of what is used
nr_tested = 0
testing_index = 0
index_list = [0 for _ in range(4)]
# start wtih some initialization
initialization = initializations[1]
opt_algorithm = opt_algorithms[0]
regularization = regularizations[0]


# while tested 70 or less updating the hyperparemeters
while nr_tested < 20:
    nodes = nr_nodes[index_list[0]]
    batch_size = batch_sizes[index_list[1]]
    regularization_rate = regularization_rates[index_list[2]]
    learning_rate = learning_rates[index_list[3]]

    print(f"Nr nodes: {nodes}")
    print(f"Batch size: {batch_size}")
    print(f"Regularization rate: {regularization_rate}")
    print(f"Learning rate: {learning_rate}")
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

    # if underfitting, add overfitting
    if float(train_loss) >= float(val_loss) * 0.95:
        if index_list[testing_index] < len(everything[testing_index]) - 1:
            index_list[testing_index] += 1
        else:
            if testing_index < 3:
                testing_index += 1
            else:
                testing_index = 0
    # else take a step back
    else:
        index_list[testing_index] -= 1
        if testing_index < 3:
            testing_index += 1
        else:
            testing_index = 0
    nr_tested += 1
