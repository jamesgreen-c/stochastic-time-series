# stochastic-time-series

# Kalman Filtering and Smoothing with Expectation Maximisation

This repository provides a modular implementation of Kalman filtering and smoothing for time-series signal processing, along with an Expectation-Maximisation (EM) algorithm to learn model parameters.

The setup is particularly suited for latent-state space models where observations are noisy projections of hidden dynamics.

---

## Features

- **Kalman Filtering** (forward inference of hidden states)
- **Kalman Smoothing** (forward-backward inference for improved estimates)
- **EM Algorithm** for unsupervised parameter learning
- **Synthetic Data Generation** utilities for testing
- **Visualisation** of latent means and uncertainty

---

## Installation

Clone this repository and install required packages:

```bash
git clone https://github.com/jamesgreen-c/stochastic-time-series.git
cd stochastic-time-series
pip install -r requirements.txt
```

Dependencies include:

- `numpy`
- `scipy`
- `matplotlib`
- `tqdm`

---

## File Structure

```
core/
├── models/
│   ├── kalman.py                   # Kalman filter and smoother
│   ├── expectation_maximisation.py # EM algorithm for parameter estimation
│   └── utils.py                    # Helper functions (e.g., shape checking)
tests/
├── kalman_test.py              # Run examples and plot results
data/
├── ssm_spins.txt
├── ssm_spins_test.txt
```

---

## Usage

### 1. Kalman Filtering/Smoothing

```python
from core.models.kalman import Kalman
from core.tests.kalman_test import random_init
import numpy as np

X = np.loadtxt("../data/ssm_spins.txt").T
y_init, Q_init, A, Q, C, R = random_init()
kf = Kalman(X, y_init, Q_init, A, Q, C, R, smooth=True)
y_hat, V_hat, V_joint, likelihood = kf.run()
```

### 2. Expectation Maximisation

```python
from core.models.expectation_maximisation import SignalProcessingEM
from core.tests.kalman_test import random_init
import numpy as np

X = np.loadtxt("../data/ssm_spins.txt").T
y_init, Q_init, A, Q, C, R = random_init()

em = SignalProcessingEM(X, y_init, Q_init, A, Q, C, R)
log_likelihoods, y_hat, V_hat, V_joint, A, Q, C, R = em.run()
```

---

## Parameters

- `X`: Observation matrix of shape `(d, T)`
- `y_init`: Initial latent state `(k,)`
- `Q_init`: Initial latent covariance `(k, k)`
- `A`: Latent transition matrix `(k, k)`
- `Q`: Latent innovation covariance `(k, k)`
- `C`: Observation matrix `(d, k)`
- `R`: Observation noise covariance `(d, d)`

---

## Visualisation

Latent means and log-determinants of posterior covariances can be visualised via:

```python
from core.models.kalman import Kalman
from core.tests.kalman_test import random_init
from core.tests.kalman_test import plot_means
import numpy as np

X = np.loadtxt("../data/ssm_spins.txt").T
y_init, Q_init, A, Q, C, R = random_init()

kf = Kalman(X, y_init, Q_init, A, Q, C, R, smooth=True)
y_hat, V_hat, V_joint, likelihood = kf.run()

plot_means(y_hat, V_hat)
```

---

## Notes

- All matrix shapes are validated with assertions before execution.
- The EM algorithm terminates early when log-likelihood convergence is detected.
- Small stabilisation noise is added to matrices during inversion to ensure numerical stability.

---

## License

MIT License

---

## Acknowledgements

Inspired by classical state-space models and implementations from time-series analysis literature.
