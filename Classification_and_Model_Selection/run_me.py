from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import numpy as np
import csv
import matplotlib.pyplot as plt


""" Handler function to write CSV to a file """


def write_csv(y_pred, filename):
    """Write a 1d numpy array to a Kaggle-compatible .csv file"""
    with open(filename, "w") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Id", "Category"])
        for idx, y in enumerate(y_pred):
            csv_writer.writerow([idx, y])


""" Handler function to print the loss table """


def loss_table_print(tables):
    for table in tables:
        table = pd.DataFrame(table)
        print(table)


stuff = np.load("data.npz")
X_trn = stuff["X_trn"]
y_trn = stuff["y_trn"]
X_tst = stuff["X_tst"]

""" Train 6 different classification trees on the image data, with each of the following maximum depths: {1,3,6,9,12,14}. 
(Do not apply any other restriction when growing the tree.) Using 5-fold cross validation, estimate mean the out of sample 
(generalization) classification error, and report this as a table. You should have one row for each possible depth and one number, 
which is the mean estimated error."""

depths = [1, 3, 6, 9, 12, 14]
table = []
for depth in depths:
    cv = KFold(n_splits=5, shuffle=True, random_state=2)
    clf = DecisionTreeClassifier(max_depth=depth)
    scores = cross_val_score(clf, X_trn, y_trn, scoring="accuracy", cv=cv, n_jobs=-1)
    table.append({"depth": depth, "mean_classifcation_error": 1 - np.mean(scores)})


table = pd.DataFrame(table)
print(table)

""" What depth performs best in the previous question? Using that depth, make predictions on the test data, and upload 
your predictions to Kaggle. """

clf_best_depth = DecisionTreeClassifier(max_depth=6)
clf_best_depth = clf_best_depth.fit(X_trn, y_trn)
y_pred = clf_best_depth.predict(X_tst)
write_csv(y_pred, "to_kaggle.csv")

""" Do nearest-neighbor prediction for each of the following possible values of K: {1, 3, 5, 7, 9, 11}. Using 5-fold 
cross-validation, estimate the out of sample classification error, and report this as a table. 
Note : Out of sample classification error is the mean of the validation errors over the 5-folds"""
# shuffle the data
def k_nearest_neighbors(X_trn, y_trn, k, folds=5):
    temp = list(zip(X_trn, y_trn))
    random.shuffle(temp)
    X_trn, y_trn = zip(*temp)
    fold_size = int(len(X_trn) / folds)
    # k-fold splits
    X_trn_folds = [X_trn[i * fold_size : (i + 1) * fold_size] for i in range(5)]
    y_trn_folds = [y_trn[i * fold_size : (i + 1) * fold_size] for i in range(5)]

    # print(X_trn_folds)
    # 5 fold splits
    shuffle_means = []
    for i in range(5):
        X_trn_val = X_trn_folds[i]
        y_trn_val = y_trn_folds[i]

        X_trn_trn = X_trn_folds[0:i] + X_trn_folds[i + 1 :]
        y_trn_trn = y_trn_folds[0:i] + y_trn_folds[i + 1 :]

        X_trn_trn = [item for sublist in X_trn_trn for item in sublist]
        y_trn_trn = [item for sublist in y_trn_trn for item in sublist]

        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=4)

        # Train the model using the training sets
        knn.fit(X_trn_trn, y_trn_trn)

        # Predict the response for test dataset
        y_pred = knn.predict(X_trn_val)

        shuffle_means.append(np.mean(y_pred == y_trn_val))
    out_of_sample_error = 1 - (sum(shuffle_means) / len(shuffle_means))
    return out_of_sample_error


for i in [1, 3, 5, 7, 9, 11]:
    print(
        "Out of Sample Error - Nearest Neighbors = ",
        i,
        " : ",
        k_nearest_neighbors(X_trn, y_trn, i, 5),
    )


""" What K performs best in the previous question? Using that K, make predictions on the test data, and upload your 
predictions to Kaggle. """

# Question 8
def k_nearest_neighbors_test(X_trn, y_trn, k, X_tst):
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=4)

    # Train the model using the training sets
    knn.fit(X_trn, y_trn)

    # Predict the response for test dataset
    y_pred = knn.predict(X_tst)

    write_csv(y_pred, "Q8-sample-predictions.csv")


k_nearest_neighbors_test(X_trn, y_trn, 3, X_tst)

# ## Question 9
#

# In[ ]:


# cross validation

""" For both hinge loss and logistic loss, train linear models with ridge regularization.
 where $L$ is the loss. For each loss and each of the regularization constants $\lambda \in \{10^{-4},10^{-2},1,10,100\}$, 
 train a model and estimate the mean out of sample loss/error using 5-fold cross-validation. Organize your errors as a
  5x2 table, with one row for each value of $\lambda$ and one column for each training loss."""


def cross_fold_fit(X_trn, y_trn, lam, clf, folds=5):
    from sklearn.metrics import zero_one_loss
    from sklearn.metrics import log_loss
    from sklearn.metrics import hinge_loss

    temp = list(zip(X_trn, y_trn))
    random.shuffle(temp)
    X_trn, y_trn = zip(*temp)
    fold_size = int(len(X_trn) / folds)

    # k-fold splits
    X_trn_folds = [X_trn[i * fold_size : (i + 1) * fold_size] for i in range(5)]
    y_trn_folds = [y_trn[i * fold_size : (i + 1) * fold_size] for i in range(5)]

    # 5 fold splits
    shuffle_means = []
    h_loss = 0
    zo_loss = 0
    l_loss = 0

    for i in range(5):
        X_trn_val = X_trn_folds[i]
        y_trn_val = y_trn_folds[i]

        X_trn_trn = X_trn_folds[0:i] + X_trn_folds[i + 1 :]
        y_trn_trn = y_trn_folds[0:i] + y_trn_folds[i + 1 :]

        X_trn_trn = [item for sublist in X_trn_trn for item in sublist]
        y_trn_trn = [item for sublist in y_trn_trn for item in sublist]

        # Train the model using the training sets
        clf = clf.fit(X_trn_trn, y_trn_trn)

        # Predict the response for test dataset
        pred_decision = clf.decision_function(X_trn_val)
        h_loss += hinge_loss(y_trn_val, pred_decision)
        zo_loss += zero_one_loss(y_trn_val, np.argmax(pred_decision, axis=1))
        l_loss += log_loss(y_trn_val, pred_decision)

    h_loss = h_loss / 5
    zo_loss = zo_loss / 5
    l_loss = l_loss / 5
    loss = [h_loss, zo_loss, l_loss]
    return loss


# In[ ]:


lambda_ = [0.0001, 0.01, 1, 10, 100]
import pandas as pd
import numpy as np
import random
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

stuff = np.load("data.npz")
X_trn = stuff["X_trn"]
y_trn = stuff["y_trn"]
X_tst = stuff["X_tst"]

table_log = []
table_hinge = []

for lam in lambda_:
    clf = LogisticRegression(C=lam)
    loss = cross_fold_fit(X_trn, y_trn, lam, clf, folds=5)
    table_log.append(
        {
            "Lambda": lam,
            "Hinge_loss": loss[0],
            "Zero_one_error": loss[1],
            "Logistic loss": loss[2],
        }
    )


for lam in lambda_:
    clf = LinearSVC(loss="hinge", C=lam)
    loss = cross_fold_fit(X_trn, y_trn, lam, clf, folds=5)
    table_hinge.append(
        {
            "Lambda": lam,
            "Hinge_loss": loss[0],
            "Zero_one_error": loss[1],
            "Logistic loss": loss[2],
        }
    )

""" Choose the training loss and \lambda that you think will perform best on the public leaderboard. Make predictions 
for the test data and upload your predictions to Kaggle. """


clf = LogisticRegression(C=0.01)
clf = clf.fit(X_trn, y_trn)

# Predict the response for test dataset
pred_decision = clf.decision_function(X_tst)
y_pred = np.argmax(pred_decision, axis=1)
write_csv(y_pred, "Q10-sample-predictions.csv")


""" You will train several neural networks, each with a single hidden layer. These neural networks can be written as
- x is the input, a vector of length D
- W is a matrix of size MxD that maps input features to a hidden space
- b is the bias term for the hidden layer, a vector of length M
- tanh is the activation function. 
- V is a matrix of size OxM that maps the hidden space to the output space
- c is the bias term for the output space, a vector of length O 
where y is in {0, 1, 2, 3} is the label for the input x, and f is is the output vector. Note that f_y is therefore
 the yTH component of the output vector f. Also be careful to note that here, we are indexing f from 0 instead of 1."""

""" Write a function to evaluate the neural network and loss """


def prediction_loss(x, y, W, V, b, c):
    # do stuff here
    a = np.add(np.matmul(x, np.transpose(W)), b)
    sig_a = np.tanh(a)
    f_x = np.add(c, np.matmul(sig_a, np.transpose(V)))
    softmax_eq = np.sum(np.exp(f_x))
    L = f_x[y] + np.log(softmax_eq)
    return L


x = np.array([1, 2])
y = 1
W = np.array([[0.5, -1], [-0.5, 1], [1, 0.5]])
V = np.array([[-1, -1, 1], [1, 1, 1]])
b = np.array([0, 0, 0])
c = np.array([0, 0])


""" Write a function to evaluate the gradient of the neural network. Do not use packages outside numpy """


def prediction_output(x, y, W, V, b, c):
    a = np.add(np.matmul(x, np.transpose(W)), b)
    sig_a = np.tanh(a)

    sig_a_deriv = np.copy(sig_a)
    with np.nditer(sig_a_deriv, op_flags=["readwrite"]) as it:
        for x in it:
            x[...] = 1 - x ** 2
    f_x = np.add(c, np.matmul(sig_a, np.transpose(V)))
    return f_x, sig_a, sig_a_deriv


def prediction_grad(x, y, W, V, b, c):
    # do stuff here
    fX, sig_a, sig_a_deriv = prediction_output(x, y, W, V, b, c)
    # calculat dL/df
    eY = np.zeros(len(c))
    eY[y] = -1
    g_fX_den = np.sum(np.exp(fX))
    g_fX = np.copy(fX)
    g_fX = np.exp(g_fX)
    with np.nditer(g_fX, op_flags=["readwrite"]) as it:
        for item in it:
            item[...] = item / g_fX_den
    print("eY", eY)
    print("g_fX", g_fX)
    dLdF = np.add(eY, g_fX)
    # calculate dL/dc
    dLdC = np.copy(dLdF)

    # calculate dL/dV
    dLdV = np.outer(dLdF, np.transpose(sig_a))

    # calculate dL/db
    dLdB = sig_a_deriv * (np.matmul(np.transpose(V), dLdF))
    # return dLdW, dLdV, dLdb, dLdc

    # calculate dL/dW
    dLdW = np.outer(dLdB, np.transpose(x))
    return dLdW, dLdV, dLdB, dLdC


x = np.array([1, 2])
y = 1
W = np.array([[0.5, -1], [-0.5, 1], [1, 0.5]])
V = np.array([[-1, -1, 1], [1, 1, 1]])
b = np.array([0, 0, 0])
c = np.array([0, 0])

prediction_grad(x, y, W, V, b, c)


""" Write a function to evaluate the same gradient as in Question 12 using the autograd toolbox (Hint: You will need to
 import the NumPy wrapper, import autograd.numpy as np, and the grad high-order function, from autograd import grad) """

## Question 14
import autograd.numpy as np
from autograd import grad


def prediction_loss(x, y, W, V, b, c):
    sigma_term = np.tanh(b + np.matmul(W, x))
    f = c + np.matmul(V, sigma_term)
    Loss = -f[y] + np.log(np.sum(np.exp(f)))
    return Loss


def prediction_grad_autograd(x, y, W, V, b, c):
    dLdW = grad(prediction_loss, 2)(x, y, W, V, b, c)
    dLdV = grad(prediction_loss, 3)(x, y, W, V, b, c)
    dLdb = grad(prediction_loss, 4)(x, y, W, V, b, c)
    dLdc = grad(prediction_loss, 5)(x, y, W, V, b, c)
    return dLdW, dLdV, dLdb, dLdc


x = np.array([1, 2])
y = 1
W = np.array([[0.5, -1], [-0.5, 1], [1, 0.5]])
V = np.array([[-1, -1, 1], [1, 1, 1]])
b = np.array([0, 0, 0])
c = np.array([0, 0])

print(prediction_loss(x, y, W, V, b, c))

X = np.array([[1, 2], [2, 4], [5, 2], [6, 4]])
Y = np.array([0, 1, 1, 0])
W = np.array([[0.5, -1], [-0.5, 1], [1, 0.5]])
V = np.array([[-1, -1.0, 1], [1, 1, 1]])
b = np.array([[1], [1], [1]])
c = np.array([[0], [0]])
λ = 1


""" Update your function from question 11. Instead of taking a single input x and a single output y, take an 2D of inputs 
X (where the first dimension indexes the different examples) and a 1D array of outputs Y. Also, take a regularization 
constant λ and apply squared regularization to W and V. Do not regularize b or c. 
Your function should be the sum of the logistic losses for each example in the dataset, plus the regularizer loss
 applied to W and V."""


def prediction_loss_full(X, Y, W, V, b, c, λ):
    regularization = λ * (np.sum(np.sum(np.square(W))) + np.sum(np.sum(np.square(V))))
    h = np.tanh(b + np.matmul(W, X.T))
    f_terms = c + np.dot(V, h)
    f_terms = f_terms.T
    Loss_terms = np.array(
        [-f[y] + np.log(np.sum(np.exp(f))) for f, y in zip(f_terms, Y)]
    )
    return np.sum(Loss_terms) + regularization


prediction_loss_full(X, Y, W, V, b, c, λ)


""" Update your gradient function to work on a full dataset and include regularization, as in the previous question."""


def prediction_grad_full(X, Y, W, V, b, c, λ):
    dLdW, dLdV, dLdb, dLdc = grad(prediction_loss_full, [2, 3, 4, 5])(
        X, Y, W, V, b, c, λ
    )
    return dLdW, dLdV, dLdb, dLdc


X = np.array([[1.0, 2.0]])
Y = [1]
W = np.array([[0.5, -1], [-0.5, 1], [1, 0.5]])
V = np.array([[-1.0, -1.0, 1.0], [1.0, 1.0, 1.0]])
b = np.array([[0.0], [0.0], [0.0]])
c = np.array([[0.0], [0.0]])
λ = 0.0
prediction_grad_full(X, Y, W, V, b, c, λ)


""" For each size of the hidden layer, M in {5, 40, 70}, train your neural network on the main data for this homework. 
Weights for layers W and V should be initialized by sampling from  the standard normal distribution and D is the number 
of input dimensions for that layer. Weight for b and c should be initialized as zeros. 

Use gradient descent with momentum, with 1000 iterations, a step size of .000025, a momentum of 0.1, and lambda = 1.

Report the following:

1. For each value of $M$, what is the total training time (in ms) for all iterations. (Give a table with 3 entries.)
2. Make a plot of the training objective (regularized loss) as a function of iterations. This should be a single plot 
with 3 curves, one for each value of $M$. Include the plot in your report."""


def optimize_grad_descent(
    X, Y, W, V, b, c, M, λ=1, O=4, max_iterations=1000, stepsize=0.000025, momentum=0.1
):
    ave_grad_dW = 0
    ave_grad_dV = 0
    ave_grad_db = 0
    ave_grad_dc = 0
    for iter_ in tqdm(range(max_iterations)):
        dLdW, dLdV, dLdb, dLdc = prediction_grad_full(X, Y, W, V, b, c, λ)
        loss = prediction_loss_full(X, Y, W, V, b, c, λ)
        ave_grad_dW = ((1 - momentum) * ave_grad_dW) + (momentum * dLdW)
        ave_grad_dV = ((1 - momentum) * ave_grad_dV) + (momentum * dLdV)
        ave_grad_db = ((1 - momentum) * ave_grad_db) + (momentum * dLdb)
        ave_grad_dc = ((1 - momentum) * ave_grad_dc) + (momentum * dLdc)
        W = W - stepsize * ave_grad_dW
        V = V - stepsize * ave_grad_dV
        b = b - stepsize * ave_grad_db
        c = c - stepsize * ave_grad_dc
        Data_to_plot.append({"M": M, "iter": iter_, "loss": loss})
    return W, V, b, c


O = 4
D = len(X_trn[0])

for M in [70]:
    W = np.random.normal(0, 1, size=(M, D)) / np.sqrt(D)
    V = np.random.normal(0, 1, size=(O, M)) / np.sqrt(M)
    b = np.full((M, 1), [0.0])
    c = np.full((O, 1), [0.0])
    optimize_grad_descent(X_trn, y_trn, W, V, b, c, M)


""" Make a single train-validation split of the data with 50% used for training and 50% for testing. Train your neural
 network using the parameters above for each value of M and give the estimated generalization error. Again, using the 
 same initial weights generated using the scheme above. Then, retrain your network on all the data,  make predictions 
 for the Kaggle data, and upload to Kaggle. Report your accuracy on the public leaderboard. """


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_trn, y_trn, test_size=0.5, random_state=10
)
O = 4
D = len(X_train[0])
Data_to_plot = []
Metrics_table = []


def predict(W, V, b, c, X_test, y_test, M):
    count_true = 0
    h = np.tanh(b + np.matmul(W, X_test.T))
    f_terms = c + np.dot(V, h)
    y_pred = np.argmax(f_terms, 0)
    for pred, true in zip(y_pred, y_test):
        if pred == true:
            count_true += 1
    accuracy = count_true / len(y_pred)
    Metrics_table.append({"M": M, "accuracy": accuracy, "error": 1 - accuracy})


for M in [70]:
    W = np.random.normal(0, 1, size=(M, D)) / np.sqrt(D)
    V = np.random.normal(0, 1, size=(O, M)) / np.sqrt(M)
    b = np.full((M, 1), [0.0])
    c = np.full((O, 1), [0.0])
    W, V, b, c = optimize_grad_descent(X_train, y_train, W, V, b, c, M)
    predict(W, V, b, c, X_test, y_test, M)


M = 70
W = np.random.normal(0, 1, size=(M, D)) / np.sqrt(D)
V = np.random.normal(0, 1, size=(O, M)) / np.sqrt(M)
b = np.full((M, 1), [0.0])
c = np.full((O, 1), [0.0])
W, V, b, c = optimize_grad_descent(X_trn, y_trn, W, V, b, c, M)

h = np.tanh(b + np.matmul(W, X_tst.T))
f_terms = c + np.dot(V, h)
y_pred = np.argmax(f_terms, 0)


# In[ ]:


write_csv(y_pred, "to_kaggle_nn_70.csv")
