import gurobipy as gp
from gurobipy import GRB
import numpy as np
from sklearn.model_selection import train_test_split

nr_nodes = 8
lam = 0.01
# read the data
X = np.genfromtxt("WisconsinBreastCancerData/X.csv", delimiter=",")
Y = np.genfromtxt("WisconsinBreastCancerData/Y.csv", delimiter=",")

# generate training, testing and validation sets
X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42)

# number of training samples, features and large M
N = len(Y_train)
print(N)
P = len(X[0])
M = 10000000

# create milp model
model = gp.Model("MILP")

# add variables
Y_pred = model.addVars(N, vtype=GRB.BINARY, name="Y_pred")
W_2 = model.addVars(nr_nodes, vtype=GRB.BINARY, name="W_2")
Wcorrect_2 = model.addVars(nr_nodes, vtype=GRB.CONTINUOUS, lb=-1.1, ub=1.1, name="Wcorrect_2")
W_1 = model.addVars(nr_nodes, P, vtype=GRB.BINARY, name="W_1")
Wcorrect_1 = model.addVars(nr_nodes, P, vtype=GRB.CONTINUOUS, lb=-1.1, ub=1.1, name="Wcorrect_1")
A = model.addVars(N, nr_nodes, vtype=GRB.CONTINUOUS, lb=0, name="A")
S = model.addVars(N, nr_nodes, vtype=GRB.CONTINUOUS, lb=0, name="S")
Z = model.addVars(N, nr_nodes, vtype=GRB.BINARY, name="Z")
B1 = model.addVars(nr_nodes, vtype=GRB.CONTINUOUS, name="B1")
B2 = model.addVar(vtype=GRB.CONTINUOUS, name="B2")

# add Relu constrains
model.addConstrs((A[i, j] <= (1 - Z[i, j]) * M
                  for i in range(N)
                  for j in range(nr_nodes)))

model.addConstrs(S[i, j] <= M * Z[i, j]
                 for i in range(N) for j in range(nr_nodes))

# A's output equals the weights * input + bias for each node j and input i
# here S is to make sure its always a positive output (RELU)
model.addConstrs(gp.quicksum(
                Wcorrect_1[j, p] * X_train[i][p] for p in range(P))
                 + B1[j] - A[i, j] + S[i, j] == 0
                 for j in range(nr_nodes) for i in range(N))

# W from binary to -1 , 1
model.addConstrs(Wcorrect_1[j, p] == W_1[j, p] * 2 - 1
                 for j in range(nr_nodes) for p in range(P))
model.addConstrs(Wcorrect_2[j] == W_2[j] * 2 - 1 for j in range(nr_nodes))

# if Y predict is 0 the sum of weights * activation less than or equal 0 (with bias)
for i in range(N):
    model.addConstr(gp.quicksum(
        Wcorrect_2[j] * A[i, j] for j in range(nr_nodes))
                    + B2 - Y_pred[i] * M <= 0)

# add auxiliary variables to linearize the absolute value
aux = model.addVars(N, vtype=GRB.CONTINUOUS, name="aux")
# Add linearization constraints
model.addConstrs(Y_train[i] - Y_pred[i] <= aux[i] for i in range(N))
model.addConstrs(-Y_train[i] + Y_pred[i] <= aux[i] for i in range(N))
# Set objective function
model.setObjective(gp.quicksum(aux[i] for i in range(N))
                   + lam * (gp.quicksum(A[i, j] * A[i, j]
                   for i in range(N) for j in range(nr_nodes))
                   - B2), GRB.MINIMIZE)


model.optimize()
if model.status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT:
    Y_pred_values = np.array([Y_pred[i].X for i in range(N)])

    # Retrieve the values of decision variables
    Wcorrect_1_values = np.array([Wcorrect_1[j, p].X for j in range(nr_nodes) for p in range(P)])
    Wcorrect_1_values = Wcorrect_1_values.reshape(8, 30)

    A = np.array([A[i, j].X for i in range(N) for j in range(nr_nodes)])
    A = A.reshape(N, nr_nodes)

    Wcorrect_2_values = np.array([Wcorrect_2[i].X for i in range(nr_nodes)])

    B1_values = np.array([B1[i].X for i in range(nr_nodes)])
    B2_value = B2.X

elif model.status == GRB.INFEASIBLE:
    print("The model is infeasible.")
elif model.status == GRB.UNBOUNDED:
    print("The model is unbounded.")
else:
    print(f"Optimization terminated with status {model.status}.")

first_node_values = np.dot(Wcorrect_1_values, X_test.T) + B1_values[:, np.newaxis]
Y_test_pred = np.dot(first_node_values.T, Wcorrect_2_values) + B2_value
Y_test_pred = np.where(Y_test_pred > 0, 1, 0)
print(Y_test_pred)
print(Y_test)
# Avoid log(0) by adding a small epsilon
epsilon = 1e-15

# Clip predicted probabilities to avoid log(0) or log(1)
Y_test_pred = np.clip(Y_test_pred, epsilon, 1 - epsilon)

# Calculate binary cross entropy
binary_cross_entropy = -np.mean(Y_test * np.log(Y_test_pred) + (1 - Y_test) * np.log(1 - Y_test_pred))
print(binary_cross_entropy)

