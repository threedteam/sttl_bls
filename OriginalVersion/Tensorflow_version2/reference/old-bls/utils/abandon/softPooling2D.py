import numpy as np
from scipy.special import softmax


# Attention : this trick is not effective in small datasets such as cifar10.

# Reference:
# https://towardsdatascience.com/building-convolutional-neural-network-using-numpy-from-scratch-b30aac50e50a
def soft_pooling(feature_map, size=2, stride=2):
    pool_out = np.zeros(
        (
            int(np.ceil((feature_map.shape[1] - size + 1) / stride)),
            int(np.ceil((feature_map.shape[2] - size + 1) / stride)),
            feature_map.shape[-1],
        ),
        dtype=np.float32,
    )
    for map_num in range(feature_map.shape[-1]):
        r2 = 0
        for r in np.arange(0, feature_map.shape[1] - size - 1, stride):
            c2 = 0
            for c in np.arange(0, feature_map.shape[2] - size - 1, stride):
                activation_weights = softmax(
                    feature_map[r : r + size, c : c + size, map_num]
                )
                pool_out[r2, c2, map_num] = np.sum(
                    activation_weights
                    * feature_map[r : r + size, c : c + size, map_num]
                )
                c2 = c2 + 1
            r2 = r2 + 1
    return pool_out
