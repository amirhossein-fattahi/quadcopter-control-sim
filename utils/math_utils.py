import numpy as np


def clamp(value, low, high):
    return np.minimum(np.maximum(value, low), high)


def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.sqrt(np.mean((a - b) ** 2))


def vector_norm_rows(x):
    x = np.asarray(x, dtype=float)
    return np.linalg.norm(x, axis=1)


def unit_vector(v, eps=1e-12):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def skew(v):
    v = np.asarray(v, dtype=float)
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


def finite_difference(signal, dt):
    signal = np.asarray(signal, dtype=float)
    dt = float(dt)

    if signal.ndim == 1:
        out = np.zeros_like(signal)
        out[1:-1] = (signal[2:] - signal[:-2]) / (2.0 * dt)
        out[0] = (signal[1] - signal[0]) / dt
        out[-1] = (signal[-1] - signal[-2]) / dt
        return out

    out = np.zeros_like(signal)
    out[1:-1, :] = (signal[2:, :] - signal[:-2, :]) / (2.0 * dt)
    out[0, :] = (signal[1, :] - signal[0, :]) / dt
    out[-1, :] = (signal[-1, :] - signal[-2, :]) / dt
    return out