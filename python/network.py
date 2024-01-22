import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD
from keras.initializers import GlorotNormal, GlorotUniform
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
nr_nodes = 15
best_nr_nodes = nr_nodes
batch_size = 15
best_batch_size = batch_size
learning_rate = 0.01
best_learning_rate = learning_rate
regularization_rate = 0.01
best_regularization_rate = regularization_rate
# Optimization algorithm, regularization and initializations
regularizations = [L2, L1]
opt_algorithms = [Adam, SGD]
initializations = [GlorotNormal, GlorotUniform]

# to keep track of what is used
nr_tested = 0
i = 0
j = 0
multiplier = 0

# while tested 70 or less updating the hyperparemeters
val_loss = float('inf')
best = float('inf')
for initialization in initializations:
    for opt_algorithm in opt_algorithms:
        for regularization in regularizations:
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
            val_loss = float(model.evaluate(X_val, Y_val, verbose=0))
            if val_loss + train_loss < best and train_loss > val_loss * 0.85:
                best_initialization = initialization
                best_opt_algorithm = opt_algorithm
                best_regularization = regularization
                best = val_loss + train_loss

best = float('inf')
while nr_tested < 62:

    print(f"\nNr nodes: {nr_nodes}")
    print(f"Batch size: {batch_size}")
    print(f"Regularization rate: {regularization_rate}")
    print(f"Learning rate: {learning_rate}\n")
    last_loss = val_loss

    # create the model
    model = Sequential()
    model.add(Dense(nr_nodes, activation='relu',
                    kernel_initializer=best_initialization,
                    kernel_regularizer=best_regularization(regularization_rate)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=best_opt_algorithm(learning_rate=learning_rate),
                  loss="BinaryCrossentropy")

    # train the model
    history = model.fit(X_train, Y_train, epochs=25,
                        batch_size=batch_size, verbose=0)

    train_loss = float(history.history["loss"][-1])
    # validate the model
    val_loss = float(model.evaluate(X_val, Y_val, verbose=0))

    print(f"Training Loss: {train_loss}")
    print(f"Validation Loss: {val_loss}")
    # if we are overfitting or we are not improving: reset
    if val_loss + train_loss > best or train_loss <= val_loss * 0.5:
        nr_nodes = best_nr_nodes
        batch_size = best_batch_size
        learning_rate = best_learning_rate
        regularization_rate = best_regularization_rate
    elif val_loss + train_loss < best:
        best = val_loss + train_loss
        best_nr_nodes = nr_nodes
        best_batch_size = batch_size
        best_regularization_rate = regularization_rate
        best_learning_rate = learning_rate

    # switch hyperparamter and take step
    i = nr_tested % 4
    if nr_tested // 4 % 2 == 0:
        j = 1
    else:
        j = 0
    muliplier = 1
    if nr_tested < 10:
        multiplier = 10
    elif nr_tested < 20:
        multiplier = 7
    elif nr_tested < 30:
        multiplier = 4
    elif nr_tested < 40:
        multiplier = 2

    if i == 0:
        if j == 0:
            nr_nodes += multiplier
        else:
            nr_nodes = max(nr_nodes - multiplier, 1)
    elif i == 1:
        if j == 0:
            batch_size = max(batch_size - multiplier, 1)
        else:
            batch_size = batch_size + multiplier
    elif i == 2:
        if j == 0:
            learning_rate += 0.001 * multiplier
        else:
            learning_rate = max(learning_rate - 0.001 * multiplier, 0.00001)
    else:
        if j == 0:
            regularization_rate = max(regularization_rate - 0.001 * multiplier,
                                      0)
        else:
            regularization_rate = regularization_rate + 0.001 * multiplier
    # count
    nr_tested += 1


print(f"\nBest nr nodes: {best_nr_nodes}")
print(f"Best batch size: {best_batch_size}")
print(f"Best regularization rate: {best_regularization_rate}")
print(f"Best learning rate: {best_learning_rate}\n")
# create the model
model = Sequential()
model.add(Dense(best_nr_nodes, activation='relu',
                kernel_initializer=best_initialization,
                kernel_regularizer=best_regularization(best_regularization_rate)))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=best_opt_algorithm(learning_rate=best_learning_rate),
              loss="BinaryCrossentropy")
# train the model
history = model.fit(X_train, Y_train, epochs=25,
                    batch_size=batch_size, verbose=0)
test_score = float(model.evaluate(X_test, Y_test, verbose=0))
print(f"Final testing score: {test_score}")
