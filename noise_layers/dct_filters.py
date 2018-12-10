import math
import numpy as np


def delta(a: int, b: int) -> int:
    if a == b:
        return 1
    else:
        return 0


def dct_coefficient(n, k, N):
    """Discrete cosine coefficient
    Coefficient for dct
    :param n:
    :param k:
    :param N:
    :return:
    """
    return math.cos(np.pi / N * (n + 1. / 2.) * k)


def idct_coefficient(n, k, N):
    return (delta(0, n) * (- 1 / 2) + math.cos(
        np.pi / N * (k + 1. / 2.) * n)) * math.sqrt(1 / (2. * N))


class DctFilterGenerator:
    """
    Generate dct filters
    """
    def __init__(self, tile_size_x: int = 8, tile_size_y: int = 8, channels: int = 3):
        self.tile_size_x = tile_size_x
        self.tile_size_y = tile_size_y
        self.channels = channels


    def generate_per_channel_filter(self, size_x: int, size_y: int, function: callable) -> np.ndarray:
        filters = np.zeros((size_x, size_y, size_x * size_y))
        for k_y in range(size_y):
            for m_x in range(size_x):
                for n_y in range(size_y):
                    for l_x in range(size_x):
                        filters[n_y, l_x, k_y * self.tile_size_x + m_x] = function(n_y, k_y, size_y) * function(l_x,
                                                                                                                m_x,
                                                                                                                size_x)
        return filters


    def get_dct_filters(self) -> np.ndarray:

        filters = self.generate_per_channel_filter(self.tile_size_x, self.tile_size_y, dct_coefficient)
        result = np.zeros(
            (self.channels, self.tile_size_x, self.tile_size_y, self.channels, self.tile_size_x * self.tile_size_y),
            dtype=np.float32)
        for channel in range(self.channels):
            result[channel, :, :, channel, :] = filters[:, :, :]
        return result

    def get_idct_filters(self) -> np.ndarray:

        filters = self.generate_per_channel_filter(self.tile_size_x, self.tile_size_y, idct_coefficient)
        result = np.zeros(
            (self.channels, self.tile_size_x, self.tile_size_y, self.channels, self.tile_size_x * self.tile_size_x),
            dtype=np.float32)
        for channel in range(self.channels):
            result[channel, :, :, channel, :] = filters[:, :, :]
        return result


    def get_jpeg_yuv_filter_mask(self, shape: tuple, N: int, count: int):
        mask = np.zeros((N, N), dtype=np.uint8)

        index_order = sorted(((x, y) for x in range(N) for y in range(N)),
                             key=lambda p: (p[0] + p[1], -p[1] if (p[0] + p[1]) % 2 else p[1]))

        for i, j in index_order[0:count]:
            mask[i, j] = 1

        return np.tile(mask, (int(np.ceil(shape[0] / N)), int(np.ceil(shape[1] / N))))[0: shape[0], 0: shape[1]]