import numpy as np
from .layers import (
    Conv2DLayer, LocallyConnected2DLayer,
    MaxPooling2DLayer, AveragePooling2DLayer,
    GlobalAveragePooling2DLayer, GlobalMaxPooling2DLayer,
    FlattenLayer, DenseLayer, BatchNormLayer,
)


class CNNFromScratch:
    """
    Wraps a trained Keras CNN model and reproduces its forward pass using
    only NumPy-based layer implementations.

    Usage:
        scratch_model = CNNFromScratch(keras_model)
        preds = scratch_model.predict(images_np)  # shape (N, num_classes)
    """

    def __init__(self, keras_model):
        self.layers = []
        self._build(keras_model)

    def _build(self, keras_model):
        for layer in keras_model.layers:
            cls_name = type(layer).__name__
            cfg = layer.get_config()
            weights = layer.get_weights()

            if cls_name == 'InputLayer':
                continue

            elif cls_name == 'Conv2D':
                kernel, bias = weights
                strides = tuple(cfg.get('strides', (1, 1)))
                padding = cfg.get('padding', 'valid')
                activation = cfg.get('activation', None)
                self.layers.append(
                    Conv2DLayer(kernel, bias, strides=strides, padding=padding,
                                activation=activation)
                )

            elif cls_name in ('LocallyConnected2D'):
                kernel, bias = weights
                strides = tuple(cfg.get('strides', (1, 1)))
                activation = cfg.get('activation', None)
                kH, kW = cfg['kernel_size']
                self.layers.append(
                    LocallyConnected2DLayer(kernel, bias, kH=kH, kW=kW,
                                            strides=strides, activation=activation)
                )

            elif cls_name == 'MaxPooling2D':
                pool_size = tuple(cfg.get('pool_size', (2, 2)))
                strides = cfg.get('strides')
                strides = tuple(strides) if strides else None
                padding = cfg.get('padding', 'valid')
                self.layers.append(
                    MaxPooling2DLayer(pool_size=pool_size, strides=strides, padding=padding)
                )

            elif cls_name == 'AveragePooling2D':
                pool_size = tuple(cfg.get('pool_size', (2, 2)))
                strides = cfg.get('strides')
                strides = tuple(strides) if strides else None
                padding = cfg.get('padding', 'valid')
                self.layers.append(
                    AveragePooling2DLayer(pool_size=pool_size, strides=strides, padding=padding)
                )

            elif cls_name == 'GlobalAveragePooling2D':
                self.layers.append(GlobalAveragePooling2DLayer())

            elif cls_name == 'GlobalMaxPooling2D':
                self.layers.append(GlobalMaxPooling2DLayer())

            elif cls_name == 'Flatten':
                self.layers.append(FlattenLayer())

            elif cls_name == 'Dense':
                w, b = weights
                activation = cfg.get('activation', None)
                self.layers.append(DenseLayer(w, b, activation=activation))

            elif cls_name == 'BatchNormalization':
                # Keras returns [gamma, beta, moving_mean, moving_var]
                gamma, beta, moving_mean, moving_var = weights
                eps = cfg.get('epsilon', 1e-3)
                self.layers.append(
                    BatchNormLayer(gamma, beta, moving_mean, moving_var, eps=eps)
                )

            elif cls_name == 'Dropout':
                # No-op during inference
                continue

            else:
                print(f"[CNNFromScratch] Skipping unsupported layer: {cls_name}")

    def predict(self, x, batch_size=32):
        """
        Run forward propagation on input images.

        Args:
            x (np.ndarray): Input array, shape (N, H, W, C) or (H, W, C).
            batch_size (int): Number of images to process at once.

        Returns:
            np.ndarray: Predictions, shape (N, num_classes).
        """
        squeeze = x.ndim == 3
        if squeeze:
            x = x[np.newaxis]

        n = x.shape[0]
        results = []

        for start in range(0, n, batch_size):
            batch = x[start:start + batch_size]
            out = batch.astype(np.float32)
            for layer in self.layers:
                out = layer.forward(out)
            results.append(out)

        preds = np.concatenate(results, axis=0)
        return preds[0] if squeeze else preds
