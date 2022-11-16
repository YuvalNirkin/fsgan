import math
import numpy as np


def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0, t_e=33.333):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.t_e = t_e
        self.a_d = smoothing_factor(self.t_e, self.d_cutoff)
        self.x_prev = self.dx_prev = None

    def reset(self):
        self.x_prev = self.dx_prev = None

    def __call__(self, x):
        """Compute the filtered signal."""
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            return x, 0.

        # The filtered derivative of the signal.
        # dx = (x - self.x_prev) / self.t_e
        dx = np.linalg.norm(x - self.x_prev) / self.t_e
        dx_hat = exponential_smoothing(self.a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = smoothing_factor(self.t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat

        return x_hat, a
