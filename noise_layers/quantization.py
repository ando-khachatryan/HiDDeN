import torch.nn as nn

def tensor_float_to_uint(tensor, values_range):
    """
    Quantize a float tensor of range [0, 1] or [-1, 1] into [0, 255]
    :param tensor:
    :param values_range:
    :return:
    """
    factor = 255.0/(values_range[1] - values_range[0])
    tensor_uint = ((tensor - values_range[0]) * factor).round().clamp(0, 255)
    return tensor_uint


def tensor_uint_to_float(tensor, target_range_min):
    """
    Normalize a tensor of range [0, 255] into either [0, 1] or [-1, 1]
    :param tensor:
    :param values_range:
    :return:
    """
    factor = (1-target_range_min)/255
    return tensor.float()*factor + target_range_min


def tensor_save_load(tensor, min_value):
     tensor_uint = tensor_float_to_uint(tensor, [min_value, 1])
     tensor_float = tensor_uint_to_float(tensor_uint, min_value)
     return tensor_float


class Quantization(nn.Module):
    def __init__(self):
        super(Quantization, self).__init__()
        self.min_value = -1

    def forward(self, batch):
        batch[0] = tensor_save_load(batch[0], self.min_value)
        return batch
