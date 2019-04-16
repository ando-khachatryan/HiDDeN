import torch
import torch.nn
import argparse
import os
import numpy as np
from options import HiDDenConfiguration

import utils
from model.hidden import *
from noise_layers.noiser import Noiser
from PIL import Image
import torchvision.transforms.functional as TF


def randomCrop(img, height, width):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    x = np.random.randint(0, img.shape[1] - width)
    y = np.random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    return img


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    parser = argparse.ArgumentParser(description='Test trained models')
    parser.add_argument('--options-file', '-o', default='options-and-config.pickle', type=str,
                        help='The file where the simulation options are stored.')
    parser.add_argument('--checkpoint-file', '-c', required=True, type=str, help='Model checkpoint file')
    parser.add_argument('--batch-size', '-b', default=12, type=int, help='The batch size.')
    parser.add_argument('--source-image', '-s', required=True, type=str,
                        help='The image to watermark')
    # parser.add_argument('--times', '-t', default=10, type=int,
    #                     help='Number iterations (insert watermark->extract).')

    args = parser.parse_args()

    train_options, hidden_config, noise_config = utils.load_options(args.options_file)
    noiser = Noiser(noise_config)

    checkpoint = torch.load(args.checkpoint_file)
    hidden_net = Hidden(hidden_config, device, noiser, None)
    utils.model_from_checkpoint(hidden_net, checkpoint)


    image_pil = Image.open(args.source_image)
    image = randomCrop(np.array(image_pil), hidden_config.H, hidden_config.W)
    image_tensor = TF.to_tensor(image).to(device)
    image_tensor = image_tensor * 2 - 1  # transform from [0, 1] to [-1, 1]
    image_tensor.unsqueeze_(0)

    # for t in range(args.times):
    message = torch.Tensor(np.random.choice([0, 1], (image_tensor.shape[0],
                                                    hidden_config.message_length))).to(device)
    losses, (encoded_images, noised_images, decoded_messages) = hidden_net.validate_on_batch([image_tensor, message])
    decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
    message_detached = message.detach().cpu().numpy()
    print('original: {}'.format(message_detached))
    print('decoded : {}'.format(decoded_rounded))
    print('error : {:.3f}'.format(np.mean(np.abs(decoded_rounded - message_detached))))
    utils.save_images(image_tensor.cpu(), encoded_images.cpu(), 'test', '.', resize_to=(256, 256))

    # bitwise_avg_err = np.sum(np.abs(decoded_rounded - message.detach().cpu().numpy()))/(image_tensor.shape[0] * messages.shape[1])



if __name__ == '__main__':
    main()
