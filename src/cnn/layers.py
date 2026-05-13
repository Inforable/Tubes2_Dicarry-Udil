import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared.activations import relu, softmax, sigmoid, tanh

_ACTIVATION_MAP = {
    'relu': relu,
    'softmax': softmax,
    'sigmoid': sigmoid,
    'tanh': tanh,
    'linear': lambda x: x,
    None: lambda x: x,
}

def _get_activation(name):
    if callable(name):
        return name
    return _ACTIVATION_MAP.get(name, lambda x: x)


class Conv2DLayer:
    """
    2D Convolution with shared parameters (standard Conv2D).
    Weights loaded from a trained Keras Conv2D layer.
    """

    def __init__(self, kernel, bias, strides=(1, 1), padding='valid', activation=None):
        # kernel: [kH, kW, C_in, C_out]
        self.kernel = kernel
        self.bias = bias
        self.stride_h, self.stride_w = strides
        self.padding = padding.lower()
        self.activation = _get_activation(activation)

    def _pad(self, x):
        if self.padding == 'valid':
            return x
        kH, kW = self.kernel.shape[:2]
        pad_h = (kH - 1) // 2
        pad_w = (kW - 1) // 2
        return np.pad(x, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')

    def forward(self, x):
        # Ensure batch dim
        squeeze = x.ndim == 3
        if squeeze:
            x = x[np.newaxis]

        x = self._pad(x)
        N, H, W, C_in = x.shape
        kH, kW, _, C_out = self.kernel.shape

        out_H = (H - kH) // self.stride_h + 1
        out_W = (W - kW) // self.stride_w + 1
        out = np.zeros((N, out_H, out_W, C_out), dtype=np.float32)

        for i in range(out_H):
            for j in range(out_W):
                h_start = i * self.stride_h
                w_start = j * self.stride_w
                patch = x[:, h_start:h_start+kH, w_start:w_start+kW, :]
                # patch: (N, kH, kW, C_in), kernel: (kH, kW, C_in, C_out)
                out[:, i, j, :] = np.einsum('nhwc,hwck->nk', patch, self.kernel) + self.bias

        out = self.activation(out)
        return out[0] if squeeze else out


class LocallyConnected2DLayer:
    """
    Locally connected 2D layer (no parameter sharing).
    Each spatial output position has its own set of weights.
    Weights loaded from a trained Keras LocallyConnected2D layer.
    """

    def __init__(self, kernel, bias, kH, kW, strides=(1, 1), activation=None):
        # kernel: [out_rows*out_cols, kH*kW*C_in, C_out]
        # bias:   [out_rows*out_cols, C_out]
        self.kernel = kernel
        self.bias = bias
        self.kH = kH
        self.kW = kW
        self.stride_h, self.stride_w = strides
        self.activation = _get_activation(activation)

    def forward(self, x):
        squeeze = x.ndim == 3
        if squeeze:
            x = x[np.newaxis]

        N, H, W, C_in = x.shape
        kH, kW = self.kH, self.kW
        _, flat_in, C_out = self.kernel.shape

        out_H = (H - kH) // self.stride_h + 1
        out_W = (W - kW) // self.stride_w + 1
        out = np.zeros((N, out_H, out_W, C_out), dtype=np.float32)

        for i in range(out_H):
            for j in range(out_W):
                p = i * out_W + j
                h_start = i * self.stride_h
                w_start = j * self.stride_w
                patch = x[:, h_start:h_start+kH, w_start:w_start+kW, :]
                # patch: (N, kH, kW, C_in) → flatten: (N, kH*kW*C_in)
                patch_flat = patch.reshape(N, -1)
                # kernel[p]: (kH*kW*C_in, C_out)
                out[:, i, j, :] = patch_flat @ self.kernel[p] + self.bias[p]

        out = self.activation(out)
        return out[0] if squeeze else out


class MaxPooling2DLayer:
    """Max pooling with sliding window."""

    def __init__(self, pool_size=(2, 2), strides=None, padding='valid'):
        self.pool_h, self.pool_w = pool_size
        if strides is None:
            self.stride_h, self.stride_w = pool_size
        else:
            self.stride_h, self.stride_w = strides
        self.padding = padding.lower()

    def _pad(self, x):
        if self.padding == 'valid':
            return x
        pad_h = (self.pool_h - 1) // 2
        pad_w = (self.pool_w - 1) // 2
        return np.pad(x, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                      mode='constant', constant_values=-np.inf)

    def forward(self, x):
        squeeze = x.ndim == 3
        if squeeze:
            x = x[np.newaxis]

        x = self._pad(x)
        N, H, W, C = x.shape
        out_H = (H - self.pool_h) // self.stride_h + 1
        out_W = (W - self.pool_w) // self.stride_w + 1
        out = np.zeros((N, out_H, out_W, C), dtype=np.float32)

        for i in range(out_H):
            for j in range(out_W):
                h_start = i * self.stride_h
                w_start = j * self.stride_w
                patch = x[:, h_start:h_start+self.pool_h, w_start:w_start+self.pool_w, :]
                out[:, i, j, :] = np.max(patch, axis=(1, 2))

        return out[0] if squeeze else out


class AveragePooling2DLayer:
    """Average pooling with sliding window."""

    def __init__(self, pool_size=(2, 2), strides=None, padding='valid'):
        self.pool_h, self.pool_w = pool_size
        if strides is None:
            self.stride_h, self.stride_w = pool_size
        else:
            self.stride_h, self.stride_w = strides
        self.padding = padding.lower()

    def _pad(self, x):
        if self.padding == 'valid':
            return x
        pad_h = (self.pool_h - 1) // 2
        pad_w = (self.pool_w - 1) // 2
        return np.pad(x, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')

    def forward(self, x):
        squeeze = x.ndim == 3
        if squeeze:
            x = x[np.newaxis]

        x = self._pad(x)
        N, H, W, C = x.shape
        out_H = (H - self.pool_h) // self.stride_h + 1
        out_W = (W - self.pool_w) // self.stride_w + 1
        out = np.zeros((N, out_H, out_W, C), dtype=np.float32)

        for i in range(out_H):
            for j in range(out_W):
                h_start = i * self.stride_h
                w_start = j * self.stride_w
                patch = x[:, h_start:h_start+self.pool_h, w_start:w_start+self.pool_w, :]
                out[:, i, j, :] = np.mean(patch, axis=(1, 2))

        return out[0] if squeeze else out


class GlobalAveragePooling2DLayer:
    """Reduce spatial dims by computing mean over H and W."""

    def forward(self, x):
        squeeze = x.ndim == 3
        if squeeze:
            x = x[np.newaxis]
        out = np.mean(x, axis=(1, 2))
        return out[0] if squeeze else out


class GlobalMaxPooling2DLayer:
    """Reduce spatial dims by computing max over H and W."""

    def forward(self, x):
        squeeze = x.ndim == 3
        if squeeze:
            x = x[np.newaxis]
        out = np.max(x, axis=(1, 2))
        return out[0] if squeeze else out


class FlattenLayer:
    """Flatten (H, W, C) → 1D vector, row-major (C order), consistent with Keras."""

    def forward(self, x):
        squeeze = x.ndim == 3
        if squeeze:
            x = x[np.newaxis]
        out = x.reshape(x.shape[0], -1)
        return out[0] if squeeze else out


class DenseLayer:
    """Fully-connected (dense) layer."""

    def __init__(self, weights, bias, activation=None):
        # weights: [input_dim, output_dim]
        self.weights = weights
        self.bias = bias
        self.activation = _get_activation(activation)

    def forward(self, x):
        out = x @ self.weights + self.bias
        return self.activation(out)


class BatchNormLayer:
    """
    Batch normalization in inference mode.
    Applies: (x - moving_mean) / sqrt(moving_var + eps) * gamma + beta
    """

    def __init__(self, gamma, beta, moving_mean, moving_var, eps=1e-3):
        self.gamma = gamma
        self.beta = beta
        self.moving_mean = moving_mean
        self.moving_var = moving_var
        self.eps = eps

    def forward(self, x):
        return (x - self.moving_mean) / np.sqrt(self.moving_var + self.eps) * self.gamma + self.beta
