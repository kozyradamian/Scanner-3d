import numpy as np
from numpy.linalg import inv

x_observations = np.array([])
y_observations = np.array([])
z_observations = np.array([])
vx_observations = np.array([])
vy_observations = np.array([])
vz_observations = np.array([])


def kalman_filter(wsp, position, velocity):
    global x_observations
    global y_observations
    global z_observations
    global vx_observations
    global vy_observations
    global vz_observations

    if len(x_observations) > 10:
        x_observations = x_observations[10:]
        vx_observations = vx_observations[10:]
    if len(y_observations) > 10:
        y_observations = y_observations[10:]
        vy_observations = vy_observations[10:]
    if len(z_observations) > 10:
        z_observations = z_observations[10:]
        vz_observations = vz_observations[10:]

    if wsp == 'x':

        x_observations = np.concatenate((x_observations, np.array([position])))
        vx_observations = np.concatenate((vx_observations, np.array([velocity])))
        z = np.c_[x_observations, vx_observations]
    elif wsp == 'y':
        y_observations = np.concatenate((y_observations, np.array([position])))
        vy_observations = np.concatenate((vy_observations, np.array([velocity])))
        z = np.c_[y_observations, vy_observations]
    elif wsp == 'z':
        z_observations = np.concatenate((z_observations, np.array([position])))
        vz_observations = np.concatenate((vz_observations, np.array([velocity])))
        z = np.c_[z_observations, vz_observations]

    a = 2
    t = 1

    error_est_x = 20
    error_est_v = 5

    error_obs_x = 25
    error_obs_v = 6

    def prediction2d(x, v, t, a):
        A = np.array([[1, t],
                      [0, 1]])
        X = np.array([[x],
                      [v]])
        B = np.array([[0.5 * t ** 2],
                      [t]])
        X_prime = A.dot(X) + B.dot(a)
        return X_prime

    def covariance2d(sigma1, sigma2):
        cov1_2 = sigma1 * sigma2
        cov2_1 = sigma2 * sigma1
        cov_matrix = np.array([[sigma1 ** 2, cov1_2],
                               [cov2_1, sigma2 ** 2]])
        return np.diag(np.diag(cov_matrix))

    # Initial Estimation Covariance Matrix
    P = covariance2d(error_est_x, error_est_v)
    A = np.array([[1, t],
                  [0, 1]])

    # Initial State Matrix
    X = np.array([[position],
                  [velocity]])

    n = len(z[0])

    for data in z[1:]:
        X = prediction2d(X[0][0], X[1][0], t, a)
        P = np.diag(np.diag(A.dot(P).dot(A.T)))

        H = np.identity(n)
        R = covariance2d(error_obs_x, error_obs_v)
        S = H.dot(P).dot(H.T) + R
        K = P.dot(H).dot(inv(S))
        Y = H.dot(data).reshape(n, -1)

        X = X + K.dot(Y - H.dot(X))
        P = (np.identity(len(K)) - K.dot(H)).dot(P)

        return X[0][0]
