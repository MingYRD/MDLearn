import numpy as np
import pandas as pd

def pm(u, e, x, k, j, i):
    pmx = 0.
    pmx += np.exp((-0.5)*(x[j, :] - u[i]).T*e[i].I*(x[j, :] - u[i])) / np.power(2*np.pi, x.shape[0] / 2)*np.sqrt(np.linalg.det(e[i]))

    return pmx


def r_ji(u, e, x, a, i, j, k):
    p = 0.
    for l in range(k):
        p += a[l]*pm(u, e, x, k, j, l)

    return a[i]*pm(u, e, x, k, j) / p

def GMM(x, k):

    n_samples, n_features = x.shape
    a = np.zeros(k)
    a = a + 1 / k

    u = []
    e = np.array([0.5*np.eye(n_features) for i in range(k)])

    rji = np.zeros((n_samples, k))
    for i in range(k):
        n = np.random.randint(0, n_samples)
        ui = x[n, :]
        u.append(ui)
    iters = 3
    while iters:
        iters -= 1
        for j in range(n_samples):
            for i in range(k):
                rji[j, i] = r_ji(u, e, x, a, i, j, k)


        for i in range(k):
            u_1 = np.zeros(n_features)
            u_2 = 0
            for j in range(n_samples):
                u_1 += rji[j, i]*x[j, :]
                u_2 += rji[j, i]
            u[i] = u_1 / u_2

            e_1 = np.zeros((n_features, n_features))
            # e_2 = 0
            for j in range(n_samples):
                e_1 += rji[j, i]*(x[j, :] - u[i])*(x[j, :] - u[i]).T
                # e_2 +=
            e[i] = e_1 / u_2
            a[i] = u_2 / n_samples


    c = np.zeros(n_samples)
    for j in range(n_samples):
        the_max = 0
        idx = 0
        for i in range(k):
            if r_ji(u, e, x, a, i, j, k) > the_max:
                the_max = r_ji(u, e, x, a, i, j, k)
                idx = i
        c[j] = idx


    return c


data = pd.read_csv("xigua.txt", header=None, names=['A', 'B'])

x = np.asarray(data.values)
k = 3
c = GMM(x, k)
print(c)


