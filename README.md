# HiDDeN
Pytorch implementation of paper "HiDDeN: Hiding Data With Deep Networks" by Jiren Zhu, Russell Kaplan, Justin Johnson, and Li Fei-Fei: https://arxiv.org/abs/1807.09937

The authors have Lua+Torch implementation here: https://github.com/jirenz/HiDDeN

I have tested this on Pytorch 1.0. Note that this is a work in progress, and I was not yet able to fully reproduce the results of the original paper.

## Requirements

You need Pytorch 1.0 with TorchVision to run this, and matplotlib.
If you want to use Tensorboard, you need to install TensorboardX and Tensorboard. This allows to use a subset of Tensorboard functionality to visualize the training. However, this is optional.
The code has been tested and runs on Ubuntu 16.04 and Windows 10. 

## Data

We use 10,000 images for training and 1,000 images for validation. Following the original paper, we chose 
those 10,000 + 1,000 images randomly from one of the coco datasets.  http://cocodataset.org/#download

The data directory has the following structure:
```
<data_root>/
  train/
    train_class/
      train_image1.jpg
      train_image2.jpg
      ...
  val/
    val_class/
      val_image1.jpg
      val_image2.jpg
      ...
```

```train_class``` and ```val_class``` folders are so that we can use the standard torchvision data loaders without change.

## Running

You will need to install the dependencies, then run 
```
python main.py --data-dir <data_root> --batch-size <b>
```
By default, tensorboard logging is enabled. To use this, you need to install Tensorboard and TensorboardX. 
However, if you don't want to use Tensorboard, you don't need to install Tensorboard/TensorboardX. Simply run with the 
```--no-tensorboard``` switch.

There are additional parameters for main.py, see the code. I will add their description here over time.

Each run creates a folder in ./runs/<date-and-time> and stores all the info about the run in there.

## Experiments

I will post experimental runs in here.
