from scipy.linalg import cholesky
import matplotlib.pyplot as plt
import numpy as np
from core.models.kalman import Kalman
from core.models.expectation_maximisation import SignalProcessingEM


def generating_params():
    """ The params used to generate the data. """
    # Make params
    a = (2 * np.pi) / 180
    b = (2 * np.pi) / 90
    A = 0.99 * np.array(
        [[np.cos(a), - np.sin(a), 0, 0],
         [np.sin(a), np.cos(a), 0, 0],
         [0, 0, np.cos(b), - np.sin(b)],
         [0, 0, np.sin(b), np.cos(b)]]
    )
    C = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [0.5, 0.5, 0.5, 0.5]
    ])
    Q = np.eye(A.shape[0]) - (A @ A.T)
    R = np.eye(d_train)

    # make init params
    k = C.shape[1]
    y_init = np.random.randn(k)
    Q_init = np.eye(k)

    return A, C, Q, R, y_init, Q_init


def random_init():
    """ Generate random initial conditions for EM """
    A = np.random.rand(4, 4)
    C = np.random.rand(5, 4)
    Q = np.identity(A.shape[0]) - np.matmul(A, A.T)
    R = np.random.rand() * np.identity(d_train)

    y_init = np.random.randn(4)
    Q_init = np.identity(4)

    return y_init, Q_init, A, Q, C, R


def plot_means(y_hat, V_hat):

    logdet = lambda A: 2 * np.sum(np.log(np.diag(cholesky(A))))
    ld = [logdet(V) for V in V_hat]

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(y_hat.T)
    ax[0].set_title("Kalman Filter Latent Prediction")
    ax[1].plot(ld)
    ax[1].set_title("Kalman Filter Log Determinant of Covariance")

    plt.show()


if __name__ == "__main__":

    # load data
    train = np.loadtxt("../data/ssm_spins.txt").T
    test = np.loadtxt("../data/ssm_spins_test.txt").T

    d_train, t_train = train.shape
    d_test, t_test = test.shape

    # filter
    y_init, Q_init, A, Q, C, R = random_init()
    kf = Kalman(train, y_init, Q_init, A, Q, C, R, smooth=False)
    y_hat, V_hat, V_joint, likelihood = kf.run()
    plot_means(y_hat, V_hat)

    # smoothed
    y_init, Q_init, A, Q, C, R = random_init()
    kf = Kalman(train, y_init, Q_init, A, Q, C, R, smooth=True)
    y_hat, V_hat, V_joint, likelihood = kf.run()
    plot_means(y_hat, V_hat)

    # with EM
    em = SignalProcessingEM(train, y_init, Q_init, A, Q, C, R)
    _, y_hat, V_hat, _, _, _, _, _ = em.run()
    plot_means(y_hat, V_hat)
