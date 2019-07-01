import numpy as np
import tensorboardX


class TensorBoardLogger:
    """
    Wrapper class for easy TensorboardX logging
    """
    def __init__(self, log_dir):
        self.grads = {}
        self.tensors = {}
        self.writer = tensorboardX.SummaryWriter(log_dir)

    def grad_hook_by_name(self, grad_name):
        def backprop_hook(grad):
            self.grads[grad_name] = grad
        return backprop_hook

    def save_losses(self, losses_accu: dict, epoch: int):
        for loss_name, loss_value in losses_accu.items():
            self.writer.add_scalar('losses/{}'.format(loss_name.strip()), loss_value.avg, global_step=epoch)

    def save_grads(self, epoch: int):
        for grad_name, grad_values in self.grads.items():
            self.writer.add_histogram(grad_name, grad_values, global_step=epoch)

    def add_tensor(self, name: str, tensor):
        self.tensors[name] = tensor

    def save_tensors(self, epoch: int):
        for tensor_name, tensor_value in self.tensors.items():
            self.writer.add_histogram('tensor/{}'.format(tensor_name), tensor_value, global_step=epoch)
