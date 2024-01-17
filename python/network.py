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

# hyperparemeters starting values
nr_nodes = 50
nodes_tested = 1
last_nr_nodes = 30
batch_size = 30
batch_size_tested = 1
last_batch_size = 30
learning_rate = 0.01
learning_rates_tested = 1
last_learning_rate = 0.01
regularization_rate = 0.1
regularization_rates_tested = 1
last_regularization_rate = 0.1
# Optimization algorithm, regularization and initializations
regularizations = [L2, L1]
opt_algorithms = [Adam, SGD]
initializations = [GlorotNormal, GlorotUniform]

# to keep track of what is used
nr_tested = 0
i = 0
# start wtih some initialization
initialization = initializations[1]
opt_algorithm = opt_algorithms[1]
regularization = regularizations[0]

# while tested 70 or less updating the hyperparemeters
val_loss = float('inf')
while nr_tested < 23:

    print(f"\nNr nodes: {nr_nodes}")
    print(f"Batch size: {batch_size}")
    print(f"Regularization rate: {regularization_rate}")
    print(f"Learning rate: {learning_rate}\n")
    last_loss = val_loss

    # create the model
    model = Sequential()
    model.add(Dense(nr_nodes, activation='relu',
                    kernel_initializer=initialization,
                    kernel_regularizer=regularization(regularization_rate)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=opt_algorithm(learning_rate=learning_rate),
                  loss="BinaryCrossentropy")

    # train the model
    history = model.fit(X_train, Y_train, epochs=25,
                        batch_size=batch_size, verbose=0)

    train_loss = float(history.history["loss"][-1])
    # validate the model
    val_loss = model.evaluate(X_val, Y_val, verbose=0)

    print(f"Training Loss: {train_loss}")
    print(f"Validation Loss: {val_loss}")
    # if underfitting, add overfitting
    if float(train_loss) >= float(val_loss) * 0.75:
        # switch hyperparamter
        i = random.randint(0, 3)
        if i == 0:
            last_nr_nodes = nr_nodes
            nr_nodes += int(round(20 / batch_size_tested, 0))
            nodes_tested += 1
        elif i == 1:
            last_batch_size = batch_size
            batch_size -= int(round(10 / batch_size_tested, 0))
            batch_size_tested += 1
        elif i == 2:
            last_learning_rate = learning_rate
            learning_rate += 0.005 / learning_rates_tested
            learning_rates_tested += 1
        else:
            last_regularization_rate = regularization_rate
            regularization_rate -= 0.005 / regularization_rates_tested
            regularization_rates_tested += 1

    # else return the hyperparamter back to what it was
    else:
        if i == 0:
            nr_nodes = last_nr_nodes
        elif i == 1:
            batch_size = last_batch_size
        elif i == 2:
            learning_rate = last_learning_rate
        else:
            regularization_rate = last_regularization_rate
    # count
    nr_tested += 1
