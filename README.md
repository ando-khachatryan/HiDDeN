# HiDDeN
Pytorch implementation of paper "HiDDeN: Hiding Data With Deep Networks" by Jiren Zhu*, Russell Kaplan*, Justin Johnson, and Li Fei-Fei: https://arxiv.org/abs/1807.09937  
*: These authors contributed equally

The authors have Lua+Torch implementation here: https://github.com/jirenz/HiDDeN

Note that this is a work in progress, and I was not yet able to fully reproduce the results of the original paper.

## Requirements

You need [Pytorch](https://pytorch.org/) 1.0 with TorchVision to run this.
If you want to use Tensorboard, you need to install TensorboardX and Tensorboard. This allows to use a subset of Tensorboard functionality to visualize the training. However, this is optional.
The code has been tested with Python 3.6+ and runs both on Ubuntu 16.04, 18.04 and Windows 10.

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

You will need to install the requirements, then run 
```
python main.py new --name <experiment_name> --data-dir <data_root> --batch-size <b> 
```
By default, tensorboard logging is disabled. To enable it, use the ```--tensorboard``` switch. 
If you want to continue from a training run, use 
```
python main.py continue --folder <incomplete_run_folder>
```
There are additional parameters for main.py. Use
```
python main.py --help
```
to see the description of all of the parameters.
Each run creates a folder in ./runs/<experiment_name date-and-time> and stores all the information about the run in there.


### Running with Noise Layers
You can specify noise layers configuration. To do so, use the ```--noise``` switch, following by configuration of noise layer or layers.
For instance, the command 
```
python main.py new --name 'combined-noise' --data-dir /data/ --batch-size 12 --noise  'crop((0.2,0.3),(0.4,0.5))+cropout((0
.11,0.22),(0.33,0.44))+dropout(0.2,0.3)+jpeg()'
```
runs the training with the following noise layers applied to each watermarked image: crop, then cropout, then dropout, then jpeg compression. The parameters of the layers are explained below. **It is important to use the quotes around the noise configuration. Also, avoid redundant spaces** If you want to stack several noise layers, specify them using + in the noise configuration, as shown in the example. 

**Important**
In the paper, when using combined noise (several noise layers), for each batch, a single layer is picked randomly and applied. Instead, we apply ALL the layers in exactly the order as specified in the --noise argument. I plan to change enable support for the original approach as well.

### Noise Layer paremeters
* _Crop((height_min,height_max),(width_min,width_max))_, where **_(height_min,height_max)_** is a range from which we draw a random number and keep that fraction of the height of the original image. **_(width_min,width_max)_** controls the same for the width of the image. 
Put it another way, given an image with dimensions **_H x W,_** the Crop() will randomly crop this into dimensions **_H' x W'_**, where **_H'/H_** is in the range **_(height_min,height_max)_**, and **_W'/W_** is in the range **_(width_min,width_max)_**. In the paper, the authors use a single parameter **_p_** which shows the ratio **_(H' * W')/ (H * W)_**, i.e., which fraction of the are to keep. In our setting, you can obtain the appropriate **_p_** by picking **_height_min_**, **_height_max_**  **_width_min_**, **_width_max_** to be all equal to **_sqrt(p)_**
*  _Cropout((height_min,height_max), (width_min,width_max))_, the parameters have the same meaning as in case of _Crop_. 
* _Dropout(keep_min, keep_max)_ : where the ratio of the pixels to keep from the watermarked image, **_keep_ratio_**, is drawn uniformly from the range **_(keep_min,keep_max)_**.
* _Resize(keep_min, keep_max)_, where the resize ratio is drawn uniformly from the range **_(keep_min, keep_max)_**. This ratio applies to both dimensions. For instance, of we have Resize(0.7, 0.9), and we randomly draw the number 0.8 for a particular image, then the resulting image will have the dimensions (H * 0.8, W * 0.8).
* _Jpeg_ does not have any parameters. 


## Experiments
The data for some of the experiments are stored in './experiments/<name of the experiment> folder. This includes: figures, detailed training and validation losses, all the settings in *pickle* format, and the checkpoint file of the trained model. Here, we provide summary of the experiments.

### Setup
We try to follow the experimental setup of the original paper as closely as possibly.
We train the network on 10,000 random images from [COCO dataset](http://cocodataset.org/#home). We use 200-400 epochs for training and validation.
The validation is on 1,000 images. During training, we take randomly positioned center crops of the images. This makes sure that there is very low chance the network will see the exact same cropped image during training. For validation, we take center crops which are not random, therefore we can exactly compare metrics from one epoch to another. 

Due to random cropping, we observed no overfitting, and our train and validation metrics (mean square error, binary cross-entropy) were extremely close. For this reason, we only show the validation metrics here. 

When measuring the decoder accuracy, we do not use error-correcting codes like in the paper. We take the decoder output, clip it to range [0, 1], then round it up. We call this "Decoder bitwise error". We also report mean squate error of of the decoder for consistency with the paper.


### Experimental runs 

This table summarizes experimental runs. Detailed information about the runs can be found in ./experiments folder.

|experiment_name | loss | encoder_mse | bitwise-error | dec_mse | epoch |
|----------------------------|----------------|----------------|------------------------|-------------|----------------|
|crop(0.2-0.25) | 0.046 | 0.0019 | 0.0603 | 0.0435 | 300 |
|crop(0.5-0.6)+jpeg | 0.0433 | 0.0039 | 0.0325 | 0.0388 | 300 |
|crop(0.6-0.65)+jpeg | 0.06 | 0.0044 | 0.0612 | 0.0547 | 300 |
|crop(0.5-0.6)+resize(0.7-0.8)+jpeg-batch-12 | 0.0924 | 0.0053 | 0.1158 | 0.087 | 400 |
|cropout(0.55-0.6) | 0.071 | 0.0011 | 0.0647 | 0.0662 | 300 |
|dropout(0.55-0.6) | 0.033 | 0.0019 | 0.008 | 0.0298 | 300 |
|jpeg | 0.0272 | 0.0025 | 0.0096 | 0.0253 | 300 | 
|resize(0.3-0.5)+jpeg | 0.0664 | 0.0098 | 0.0503 | 0.0575 | 400 | 
|resize(0.7-0.8) | 0.0251 | 0.0016 | 0.0052 | 0.0238 | 300 | 


* **No noise** means no noise layers.
* **Crop((0.2,0.25),...)** is shorthand for Crop((0.2,0.25),(0.2,0.25)). This means that the height and the weight of the cropped image have the expected value of (0.25 + 0.2)/2 = 0.225. Therefore, the ratio of (expected) area of the  Cropped image against the original image is 0.225x0.225 ≈ 0.05. The paper used p = 0.035.
* **Cropout((0.55,0.6),...)** is a shorhand for Cropout((0.55,0.6),(0.55,0.6)). Similar to Crop(...), this translates to ratio of Cropped vs original image areas with p ≈ 0.33. The paper used p = 0.3
* **Jpeg** the same as the Jpeg layer from the paper. It is a differentiable approximation of Jpeg compression with the highest compression coefficient.


