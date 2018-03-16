from numpy import *
import matplotlib.pyplot as plt


def sgn(x):
    y = (x > 0) ^ (x < 0)
    return y


def MahNMF_SGD(R1, R2, P, Q, epoch, alpha_0, beta):
    Q = Q.T
    counter = 0
    while True:
        alpha_t = alpha_0 / (counter + 1)
        for i in range(m):
            for j in range(n):
                if R1[i][j] > 0:
                    eij = sgn(R1[i][j] - dot(P[i, :], Q[:, j]))
                    temp_vector1 = P[i, :] + alpha_t * (eij * Q[:, j] - beta * P[i, :])
                    temp_vector2 = Q[:, j] + alpha_t * (eij * P[i, :] - beta * Q[:, j])
                    P[i, :] = maximum(temp_vector1, 0)
                    Q[:, j] = maximum(temp_vector2, 0)
        error0 = 0
        error1 = 0
        for i in range(m):
            for j in range(n):
                if R1[i][j] > 0:
                    error0 += abs(R1[i][j] - dot(P[i, :], Q[:, j]))\
                              + (beta/2) * dot(P[i, :], P[i, :])\
                              + (beta/2) * dot(Q[:, j], Q[:, j])
        error0 /= 80000
        for i in range(m):
            for j in range(n):
                if R2[i][j] > 0:
                    error1 += abs(R2[i][j] - dot(P[i, :], Q[:, j]))\
                              + (beta/2) * dot(P[i, :], P[i, :])\
                              + (beta/2) * dot(Q[:, j], Q[:, j])
        error1 /= 20000
        print(counter, ' ', error0, ' ', error1)
        reg1.append(counter)
        reg2.append(error0)
        reg3.append(error1)
        counter += 1
        if counter > epoch:
            plt.plot(reg1, reg2, label='training loss')
            plt.plot(reg1, reg3, label='validation loss')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()
            break
    return P, Q.T


def GetMatrix(filename):
    M = [[0 for col in range(1682)] for row in range(943)]
    for line in open(filename, 'r'):
        (userid, movieid, rating, timestamp) = line.split('\t')
        M[int(userid)-1][int(movieid)-1] = float(rating)
    return M


if __name__ == "__main__":

    A_train = GetMatrix('./ml-100k/u2.base')
    A_val = GetMatrix('./ml-100k/u2.test')

    A_train = array(A_train)
    A_val = array(A_val)

    m = 943
    n = 1682
    r = 3
    epoch = 100
    alpha = 0.001
    beta = 0.001

    reg1 = []
    reg2 = []
    reg3 = []

    P = random.rand(m, r)
    Q = random.rand(n, r)

    nP, nQ = MahNMF_SGD(A_train, A_val, P, Q, epoch, alpha, beta)
    nR = dot(nP, nQ.T)
