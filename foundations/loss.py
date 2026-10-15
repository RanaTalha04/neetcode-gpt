import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(
        self,
        y_true: NDArray[np.float64],
        y_pred: NDArray[np.float64]
    ) -> float:
        # Prevent log(0)
        eps = 1e-7
        y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)

        loss = -np.mean(
            y_true * np.log(y_pred_clipped)
            + (1 - y_true) * np.log(1 - y_pred_clipped)
        )

        return round(float(loss), 4)

    def categorical_cross_entropy(
        self,
        y_true: NDArray[np.float64],
        y_pred: NDArray[np.float64]
    ) -> float:
        # Prevent log(0)
        eps = 1e-7
        y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)

        # Sum over classes for each sample
        per_sample_loss = -np.sum(
            y_true * np.log(y_pred_clipped),
            axis=1
        )

        # Average over samples
        loss = np.mean(per_sample_loss)

        return round(float(loss), 4)