# Report prepared by Team members - Naveen Jafer Nizar,
# Haritha AnanthKrishnan and Madhu Samhita Vangara.

# Question 1 -


P = {
    0: 0.2,
    1: 0.2,
    2: 0.2,
    3: 0.1,
    4: 0.1,
    5: 0.05,
    6: 0.05,
    7: 0.05,
    8: 0.025,
    9: 0.025,
}


def prior(m):
    p = P[m]
    return p


# Question 2 -


for m in range(10):
    print("m", m, "prior(m)", prior(m))

import matplotlib.pyplot as plt


def plot_graph(x, y, xlabel, ylabel, title):
    fig = plt.figure(figsize=(10, 5))

    # creating the bar plot
    plt.bar(x, y, color="maroon", width=0.4)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


m = list(P.keys())
p_m = list(P.values())
plot_graph(m, p_m, "m", "p(m)", "prior")

# Question 3 -

"""
$p(y|x,m) = \mathcal{N}(y \vert \mu=f_m(x), \sigma^2)$

$
f_m(x) = \begin{cases}0 & m=0 \\ x^m & m>0\end{cases}.$

"""


def f(m, x):
    if m == 0:
        return 0
    else:
        return x ** m


def likelihood_single(x, y, m):
    mu, sigma = f(m, x), 0.1
    PI = 3.14159265359
    PI = 22 / 7
    p = (1 / sigma / np.sqrt(2 * PI)) * np.exp(-0.5 * ((y - f(m, x)) / sigma) ** 2)
    return p


"""# Question 4 - """


def likelihood(X, Y, m):
    p = 1
    for i in range(10):
        p = p * likelihood_single(X[i], Y[i], m)
    return p


import numpy as np
from numpy import genfromtxt

X = genfromtxt("x.csv", delimiter=",")
Y = genfromtxt("y.csv", delimiter=",")

"""# Question 5
Make a bar chart of the likelihood, with m on the x-axis and the likelihood on the y-axis.
"""

likelihood_m = {m: likelihood(X, Y, m) for m in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}
plot_graph(likelihood_m.keys(), likelihood_m.values(), "m", "p(y|x,m)", "Likelihood")

likelihood_m

"""# Question 6 - 

Posterior $= Prior * Likelihood $ normalized

 $= \frac {P(m) * \prod_{n=1}^{10} p(y^{(n)} \vert x^{(n)}, m)}{\sum_{m=1}^{m=10} P(m) * \prod_{n=1}^{10} p(y^{(n)} \vert x^{(n)}, m)}$

# Question 7 -
"""


def posterior(X, Y, m):
    constant = 0
    for m_ in P.keys():
        constant += P[m_] * likelihood(X, Y, m_)
    return P[m] * likelihood(X, Y, m) / constant


"""# Question 8 - 

"""

posterior_m = {m: posterior(X, Y, m) for m in P.keys()}
plot_graph(
    posterior_m.keys(), posterior_m.values(), "m", "posterior(X,Y,m)", "Posterior"
)
print(posterior_m)

"""# Question 9 -"""


def MAP(X, Y):
    posterior_m = {m: posterior(X, Y, m) for m in P.keys()}
    m = max(posterior_m, key=posterior_m.get)
    return m


m = MAP(X, Y)

posterior_m

"""# Question 10 -
m = 2


posterior = 0.7610021202297498

# Question 11 -
"""


def predict_MAP(x, X, Y):
    # do stuff
    f_n = f(m, x)
    return f_n


"""# Question 12 -"""

X_test = genfromtxt("x_test.csv", delimiter=",")
Y_test = genfromtxt("y_test.csv", delimiter=",")

pred = []

for x in X_test:
    pred.append(f(m, x))


plt.figure()
from matplotlib.pyplot import figure

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X_test, Y_test, color="red", marker="x")
ax.scatter(X_test, pred)
ax.legend(["Actual", "Predicted"])
ax.title.set_text("chart")

"""# Question 13

"""

map_total_error = 0

predictions_map = []
for x_test in X_test:
    pred = predict_MAP(x_test, X, Y)
    predictions_map.append(pred)


total_error_map = 0
for i in range(len(X_test)):
    total_error_map += (Y_test[i] - predictions_map[i]) ** 2

total_error_map = total_error_map / len(X_test)
print("mean-squared test error of MAP: ", total_error_map)

"""# Question 14"""


def predict_Bayes(x, X, Y):
    # do stuff
    ans = 0
    for m in range(10):
        f_x_m = 0
        if m == 0:
            f_x_m = 0
        else:
            f_x_m = x ** m
        post = posterior(X, Y, m)
        ans += post * f_x_m
    return ans


predict_Bayes(0.7385, X, Y)

"""# Question15"""

from matplotlib.pyplot import figure

predictions = []
for x_test in X_test:
    pred = predict_Bayes(x_test, X, Y)
    predictions.append(pred)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X_test, Y_test, color="red", marker="x")
ax.scatter(X_test, predictions, color="green", marker="o")
ax.legend(["Actual", "Predicted"])
ax.title.set_text("chart")

"""# Question 16

"""

total_error = 0
for i in range(len(X_test)):
    total_error += (Y_test[i] - predictions[i]) ** 2

total_error = total_error / len(X_test)
print("mean-squared test error of Bayes estimate: ", total_error)

"""# Question 17
The mean-squared test error of Bayes estimate is lower than MAP. In MAP, we first find the m and rely only on that for calculating all the new values. This is always sub-optimal (and equivalent only in the case when all of the posterior is concentrated around a single model) than Bayes by construction.
"""
