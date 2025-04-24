from core.models.utils import check_param_dimensions
from tqdm import tqdm
import numpy as np
from core.models.kalman import Kalman


class SignalProcessingEM:

    def __init__(self, X, y_init, Q_init, A, Q, C, R):

        check_param_dimensions(X, y_init, Q_init, A, Q, C, R)
        self.X, self.y_init, self.Q_init = X, y_init, Q_init
        self.A, self.Q, self.C, self.R = A, Q, C, R

    def run(self, max_iter: int = 50):
        assert max_iter > 0, "Max Iter must be greater than 0"

        A, Q, C, R = self.A, self.Q, self.C, self.R
        log_likelihoods = []

        # run EM
        for i in tqdm(range(max_iter), desc="Running EM"):

            y_hat, V_hat, V_joint, likelihood = self._e_step(A, Q, C, R)  # E
            A, Q, C, R = self._m_step(y_hat, V_hat, V_joint)  # M

            # store and check log likelihood
            log_likelihoods.append(np.sum(likelihood))
            if i >= 1:
                if self.check_convergence(log_likelihoods[i - 1], log_likelihoods[i]):
                    print(f"EM Converged on iteration {i}")
                    break

        return log_likelihoods, y_hat, V_hat, V_joint, A, Q, C, R

    def _m_step(self, y_hat, V_hat, V_joint):
        """
            Run the m step of the EM algorithm after Kalman Smoothing E step

            :param X:       train data
            :param y_hat:   state estimates
            :param V_hat:   y_hat[t] to y_hat[t] covariances
            :param V_joint  y_hat[t+1] to y_hat[t] covariances

            :return: Updated A, Q, C, R
            """

        X = self.X

        T = X.shape[1]  # Number of time steps
        k = y_hat.shape[0]  # Dimension of the latent state
        d = X.shape[0]  # Dimension of the observations

        # calculate A
        zz_t_a = np.zeros((k, k))
        zz_t_next = np.zeros((k, k))

        for t in range(T - 1):
            zz_t_next += np.outer(y_hat[:, t + 1], y_hat[:, t]) + V_joint[t]
            zz_t_a += np.outer(y_hat[:, t], y_hat[:, t]) + V_hat[t]

        zz_t_a += np.outer(y_hat[:, T - 1], y_hat[:, T - 1]) + V_hat[T - 1]  # add last T
        A_new = zz_t_next.dot(np.linalg.pinv(zz_t_a))

        # calculate Q
        zz_t_q = np.zeros((k, k))
        zz_t_next = np.zeros((k, k))
        for t in range(1, T):
            zz_t_next += np.outer(y_hat[:, t], y_hat[:, t - 1]) + V_joint[t - 1]
            zz_t_q += np.outer(y_hat[:, t], y_hat[:, t]) + V_hat[t]

        Q_new = (zz_t_q - zz_t_next.dot(A_new.T)) / (T - 1)

        # calculate C
        xz_t = np.zeros((d, k))
        for t in range(T):
            xz_t += np.outer(X[:, t], y_hat[:, t])
        C_new = xz_t.dot(np.linalg.pinv(zz_t_a))

        # calculate R
        xx_t = np.zeros((d, d))
        for t in range(T):
            xx_t += np.outer(X[:, t], X[:, t])
        R_new = (xx_t - xz_t.dot(C_new.T)) / T

        # symmetrise R and Q
        R_new = (R_new + R_new.T) / 2
        Q_new = (Q_new + Q_new.T) / 2

        return A_new, Q_new, C_new, R_new

    def _e_step(self, A, Q, C, R):

        stabiliser = 1e-12  # add to matrices to ensure invertible

        X, y_init, Q_init = self.X, self.y_init, self.Q_init
        A += stabiliser * np.eye(y_init.shape[0])
        Q += stabiliser * np.eye(y_init.shape[0])
        R += stabiliser * np.eye(X.shape[0])

        # initialise Kalman Filter
        kf = Kalman(X, y_init, Q_init, A, Q, C, R, smooth=True)
        y_hat, V_hat, V_joint, likelihood = kf.run()
        return y_hat, V_hat, V_joint, likelihood

    def check_convergence(self, prev_log_likelihood: float, curr_log_likelihood: float):
        """ If log-likehoods similar enough on each time-step, say converged """
        return abs(curr_log_likelihood - prev_log_likelihood) < 1e-6


