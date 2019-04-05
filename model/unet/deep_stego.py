import numpy as np
import torch.nn as nn
import torch
import torch.nn
import torch.optim

from settings import DeepStegoSettings
from noise_layers.noiser import Noiser
from model.unet.encoder_decoder import EncoderDecoder

def extracted_watermark_accuracy(watermark, extracted_watermark):
    extracted_wm_np = extracted_watermark.detach().cpu().numpy().round()
    extracted_wm_np = np.clip(extracted_wm_np, 0, 1)
    accuracy = 1 - np.sum(np.abs(watermark.detach().cpu().numpy() - extracted_wm_np)) / extracted_wm_np.size
    return accuracy


class DeepStego:
    def __init__(self, device: torch.device, settings: DeepStegoSettings, noiser: Noiser):
        self.encoder_decoder = EncoderDecoder(noiser, settings).to(device)
        self.optimizer = torch.optim.Adam(self.encoder_decoder.parameters())
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

        self.encoder_lambda = 0.5

    def train_on_batch(self, batch):

        self.optimizer.zero_grad()
        image, watermark = batch
        self.encoder_decoder.train()
        with torch.enable_grad():
            # im_and_wm = torch.cat([image, watermark], dim=1)
            encoded_image, noised_image, extracted_watermark = self.encoder_decoder([image, watermark])

            image_loss = self.mse_loss(encoded_image, image)
            wm_loss = self.bce_loss(extracted_watermark, watermark)

            combined_loss = self.encoder_lambda * image_loss + wm_loss
            combined_loss.backward()
            self.optimizer.step()

        accuracy = extracted_watermark_accuracy(watermark, extracted_watermark)
        losses = {
            'loss       ': combined_loss.item(),
            'enc_mse    ': image_loss.item(),
            'dec_bce    ': wm_loss.item(),
            'accuracy   ': accuracy
        }

        return losses, (encoded_image, noised_image, extracted_watermark)


    def validate_on_batch(self, batch):

        image, watermark = batch
        with torch.no_grad():
            im_and_wm = torch.cat([image, watermark], dim=1)
            encoded_image, noised_image, extracted_watermark = self.encoder_decoder(im_and_wm)

            image_loss = self.mse_loss(encoded_image, image)
            wm_loss = self.bce_loss(extracted_watermark, watermark)

            combined_loss = self.encoder_lambda * image_loss + wm_loss
            accuracy  = extracted_watermark_accuracy(watermark, extracted_watermark)

        losses = {
            'loss       ': combined_loss.item(),
            'enc_mse    ': image_loss.item(),
            'dec_bce    ': wm_loss.item(),
            'accuracy   ': accuracy
        }

        return losses, (encoded_image, noised_image, extracted_watermark)


    def to_string(self):
        return str(self.encoder_decoder)

