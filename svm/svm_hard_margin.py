import numpy as np
import matplotlib.pyplot as plt
import cvxopt
import seaborn as sns


# Generate two points to construct the line
x1_targetf = np.random.uniform(-1, 1, 2)
x2_targetf = np.random.uniform(-1, 1, 2)
print("x1_targetf", x1_targetf)
print("x2_targetf", x2_targetf)

# Computing the slop m and the y-intercept b
# 这里m求的是法向量，k * k' = -1
# k' = (x2 - x1)/(y2 - y1)
# 求得k' 以后，x2_targetf 是在直线上，求得b_target
m = (x2_targetf[1] - x2_targetf[0]) / (x1_targetf[1] - x1_targetf[0])
b_target = x2_targetf[0] - m * x1_targetf[0]
print("m = ", m, "b_target = ", b_target)

# Computing the x2 value of the extremities of the line knowing that the x1
# values range is [-1, 1]
x2_lefttargetf = m * (-1) + b_target
x2_righttargetf = m * (1) + b_target

# Plot of target function
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
plt.plot([-1, 1], [x2_lefttargetf, x2_righttargetf])
plt.axis((-1, 1, -1, 1))
plt.scatter(x1_targetf[0], x2_targetf[0], marker='o')
plt.scatter(x1_targetf[1], x2_targetf[1], marker='o')

# Generating N training data points
N = 20
x1_input = np.random.uniform(-1, 1, N)
x2_input = np.random.uniform(-1, 1, N)

y_output = []

# Computing the y_output (class or category) from the target function
for index, each in enumerate(x1_input):
    # Computing the value of x2 if it was on the line
    x2_line = m*(each) + b_target

    # checking the value of x2 corresponding to the x1 with the x2 from the line
    # Assigning a -1 or +1 to class to it
    if x2_input[index] > x2_line:
        y_output.append(-1.0)
    else:
        y_output.append(1.0)

# plotting the training point
plt.scatter(x1_input, x2_input, c= y_output, cmap='bwr', alpha=1, s=100, edgecolors='k')

# Combining the data vectors into an X matrix of N vectors of features
X_svm = np.column_stack((
    x1_input,
    x2_input
))
# Changing the class list into a Numpy array
y_output = np.array(y_output)
# Initializing the N x N Gram matrix
K = np.zeros(shape=(
    len(x1_input),
    len(x1_input)
))

# Computing the inner products for each pair of vectors
for i in range(len(x1_input)):
    for j in range(len(x1_input)):
        K[i, j] = np.dot(X_svm[i], X_svm[j])

# Generating all the matrices and vectors
P = cvxopt.matrix(np.outer(y_output, y_output) * K)
q = cvxopt.matrix(np.ones(len(x1_input)) * -1)
G = cvxopt.matrix(np.vstack([
    np.eye(len(x1_input)) * -1,
    np.eye(len(x1_input))
]))
h = cvxopt.matrix(np.hstack([
    np.zeros(len(x1_input)),
    np.ones(len(x1_input)) * 999999999.0
]))
A = cvxopt.matrix(y_output, (1, len(x1_input)))
b = cvxopt.matrix(0.0)

# Solving the QP problem
solution = cvxopt.solvers.qp(P, q, G, h, A, b)

# Display the Lagrange multipliers
a = np.ravel(solution['x'])
print(a)

# Create a boolean list of non-zero alphas
ssv = a > 1e-5
# Select the index of these alphas. They are the support vectors.
ind = np.arange(len(a))[ssv]
# Select the corresponding alphas a, support vectors sv and class labels sv_y
a = a[ssv]
sv = X_svm[ssv]
sv_y = y_output[ssv]

# Plotting the support vectors
plt.scatter(
    sv[:, 0],
    sv[:, 1],
    facecolors='none',
    s=400,
    edgecolors='k'
)

# Computing the weights w_svm
w_svm = np.zeros(X_svm.shape[1])

for each in range(len(a)):
    w_svm += a[each] * sv_y[each] * sv[each]
print('w_svm:', w_svm)

# Computing the intercept b_svm
b_svm = sv_y[0] - np.dot(w_svm, sv[0])
print('b_svm:', b_svm)

# Plot of SVM function
x2_lefttargeth = -(w_svm[0] * (-1) + b_svm) / w_svm[1]
x2_righttargeth = -(w_svm[0] * (1) + b_svm) / w_svm[1]

ax1.plot([-1, 1], [x2_lefttargeth, x2_righttargeth])
plt.show()



