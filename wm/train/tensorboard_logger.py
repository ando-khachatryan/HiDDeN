import numpy as np
import torch
import torch.utils.tensorboard as tb


class TensorBoardLogger:
    """
    Wrapper class for easy Tensorboard logging
    """
    def __init__(self, log_dir):
        self.grads = {}
        self.tensors = {}
        self.writer = tb.SummaryWriter(log_dir)

    def grad_hook_by_name(self, grad_name):
        def backprop_hook(grad):
            self.grads[grad_name] = grad
        return backprop_hook

    def _save_losses(self, losses_accu: dict, epoch: int):
        for loss_name, loss_value in losses_accu.items():
            self.writer.add_scalar(f'losses/{loss_name.strip()}', loss_value, global_step=epoch)

    def _save_grads(self, epoch: int):
        for grad_name, grad_values in self.grads.items():
            self.writer.add_histogram(grad_name, grad_values, global_step=epoch)

    def add_tensor(self, name: str, tensor):
        self.tensors[name] = tensor

    def _save_tensors(self, epoch: int):
        for tensor_name, tensor_value in self.tensors.items():
            self.writer.add_histogram(f'tensor/{tensor_name}', tensor_value, global_step=epoch)

    def _save_images(self, epoch: int, cover_images: torch.Tensor, encoded_images: torch.Tensor, noised_images: torch.Tensor):
        if cover_images:
            self.writer.add_images(tag=f'cover/{epoch}', img_tensor=cover_images, global_step=epoch)
        if encoded_images: 
            self.writer.add_images(tag=f'encoded/{epoch}', img_tensor=encoded_images, global_step=epoch)
        if noised_images:
            self.writer.add_images(tag=f'noised/{epoch}', img_tensor=noised_images, global_step=epoch)
        if cover_images and encoded_images:
            image_diff = cover_images - encoded_images
            self.writer.add_images(tag=f'diff x 8/{epoch}', img_tensor=image_diff * 8)
            self.writer.add_images(tag=f'diff x 4/{epoch}', img_tensor=image_diff * 4)


    def save(self, epoch: int, losses: dict, cover_images: torch.Tensor=None, encoded_images: torch.Tensor=None, noised_images: torch.Tensor=None):
        self._save_losses(losses_accu=losses, epoch=epoch)
        self._save_tensors(epoch=epoch)
        self._save_grads(epoch=epoch)
        self._save_images(epoch=epoch, cover_images=cover_images, encoded_images=encoded_images, noised_images=noised_images)
