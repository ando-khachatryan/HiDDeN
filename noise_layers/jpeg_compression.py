import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def gen_filters(size_x: int, size_y: int, dct_or_idct_fun: callable) -> np.ndarray:
    tile_size_x = 8
    filters = np.zeros((size_x * size_y, size_x, size_y))
    for k_y in range(size_y):
        for k_x in range(size_x):
            for n_y in range(size_y):
                for n_x in range(size_x):
                    filters[k_y * tile_size_x + k_x, n_y, n_x] = dct_or_idct_fun(n_y, k_y, size_y) * dct_or_idct_fun(n_x,
                                                                                                            k_x,
                                                                                                            size_x)
    return filters

# def zigzag_indices(shape: (int, int), count):
#     x_range, y_range = shape
#     index_order = sorted(((x, y) for x in range(x_range) for y in range(y_range)),
#                          key=lambda p: (p[0] + p[1], -p[1] if (p[0] + p[1]) % 2 else p[1]))
#
#     mask = np.zeros(shape)
#     for r, c in index_order[:count]:
#         mask[r,c] = 1
#
#     return mask

def get_jpeg_yuv_filter_mask(image_shape: tuple, window_size: int, keep_count: int):
    mask = np.zeros((window_size, window_size), dtype=np.uint8)

    index_order = sorted(((x, y) for x in range(window_size) for y in range(window_size)),
                         key=lambda p: (p[0] + p[1], -p[1] if (p[0] + p[1]) % 2 else p[1]))

    for i, j in index_order[0:keep_count]:
        mask[i, j] = 1

    return np.tile(mask, (int(np.ceil(image_shape[0] / window_size)),
                          int(np.ceil(image_shape[1] / window_size))))[0: image_shape[0], 0: image_shape[1]]


def dct_coeff(n, k, N):
    return np.cos(np.pi / N * (n + 1. / 2.) * k)


def idct_coeff(n, k, N):
    return (int(0 == n) * (- 1 / 2) + np.cos(
        np.pi / N * (k + 1. / 2.) * n)) * np.sqrt(1 / (2. * N))


def rgb2yuv(image_rgb):
    r = image_rgb[:, 0:1, :, :]
    g = image_rgb[:, 1:2, :, :]
    b = image_rgb[:, 2:3, :, :]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.14713 * r - 0.28886 * g + 0.436 * b
    v = 0.615 * r - 0.51499 * g - 0.10001 * b

    return torch.cat([y, u, v], dim=1)


def yuv2rgb(image_yuv):
    y = image_yuv[:, 0:1, :, :]
    u = image_yuv[:, 1:2, :, :]
    v = image_yuv[:, 2:3, :, :]

    r = y + 1.13983 * v
    g = y - 0.39465 * u - 0.58060 * v
    b = y + 2.03211 * u

    return torch.cat([r, g, b], dim=1)


class JpegCompression(nn.Module):
    def __init__(self, device, yuv_keep_weights = (25, 9, 9)):
        super(JpegCompression, self).__init__()
        self.device = device

        self.dct_conv_weights = torch.tensor(gen_filters(8, 8, dct_coeff), dtype=torch.float32).to(self.device)
        self.dct_conv_weights.unsqueeze_(1)
        self.idct_conv_weights = torch.tensor(gen_filters(8, 8, idct_coeff), dtype=torch.float32).to(self.device)
        self.idct_conv_weights.unsqueeze_(1)

        self.yuv_keep_weighs = yuv_keep_weights
        self.keep_coeff_masks = []

        self.jpeg_mask = None

        # create a new large mask which we can use by slicing for images which are smaller
        self.create_mask((1024, 1024))


    def create_mask(self, requested_shape):
        if self.jpeg_mask is None or requested_shape > self.jpeg_mask.shape[1:]:
            self.jpeg_mask = torch.empty((3,) + requested_shape, device=self.device)
            for channel, weights_to_keep in enumerate(self.yuv_keep_weighs):
                mask = torch.from_numpy(get_jpeg_yuv_filter_mask(requested_shape, 8, weights_to_keep))
                self.jpeg_mask[channel] = mask

    def get_mask(self, image_shape):
        if any(
            requested > existing
            for requested, existing
            in zip(image_shape, self.jpeg_mask.shape[1:])
        ):
            self.create_mask(image_shape)
        # return the correct slice of it
        return self.jpeg_mask[:, :image_shape[1], :image_shape[2]].clone()


    def apply_conv(self, image, filter_type: str):

        if filter_type == 'dct':
            filters = self.dct_conv_weights
        elif filter_type == 'idct':
            filters = self.idct_conv_weights
        else:
            raise('Unknown filter_type value.')

        image_conv_channels = []
        for channel in range(image.shape[1]):
            image_yuv_ch = image[:, channel, :, :].unsqueeze(1)
            image_conv = F.conv2d(image_yuv_ch, filters, stride=8)
            image_conv = image_conv.permute(0, 2, 3, 1)
            image_conv = image_conv.view(image_conv.shape[0], image_conv.shape[1], image_conv.shape[2], 8, 8)
            image_conv = image_conv.permute(0, 1, 3, 2, 4)
            image_conv = image_conv.contiguous().view(image_conv.shape[0],
                                                  image_conv.shape[1]*image_conv.shape[2],
                                                  image_conv.shape[3]*image_conv.shape[4])

            image_conv = image_conv.unsqueeze(1)

            # image_conv = F.conv2d()
            image_conv_channels.append(image_conv)

        image_conv_stacked = torch.cat(image_conv_channels, dim=1)

        return image_conv_stacked


    def forward(self, noised_and_cover):

        noised_image = noised_and_cover[0]

        pad_height = (8 - noised_image.shape[2] % 8) % 8
        pad_width = (8 - noised_image.shape[3] % 8) % 8

        noised_image = nn.ZeroPad2d(
            (0, pad_width, 0, pad_height)
        )(noised_image)

        image_yuv = rgb2yuv(noised_image)

        image_dct = self.apply_conv(image_yuv, 'dct')

        mask = self.get_mask(image_dct.shape[1:])

        image_dct_mask = image_dct * mask

        image_idct = self.apply_conv(image_dct_mask, 'idct')

        image_ret_padded = yuv2rgb(image_idct)

        end_h = (
            image_ret_padded.shape[2] - pad_height
            if pad_height > 0
            else image_ret_padded.shape[2]
        )

        end_w = (
            image_ret_padded.shape[3] - pad_width
            if pad_width > 0
            else image_ret_padded.shape[3]
        )

        noised_image = image_ret_padded[:, :, :end_h, :end_w]

        return [noised_image, noised_and_cover[1]]
