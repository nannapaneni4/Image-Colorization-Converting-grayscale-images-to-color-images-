# Image Colorization Overview
The objective is to produce color images given grayscale input image. This is a difficult task in general as the gray scale images are of single channel and color images are of 3 channels i.e RGB. Since we don't have a straightforward mathematical formula to achieve gray2color conversion, we use Neural Network based models to achieve this.

`grayscale2color.py` is the entry point for training. The dataset consists of around 4000 landscape images. These images are first split into training and validation set during training. I used a split of 20%. Based on the `model` argument,`Trainer` objects are created and `train` method is invoked for training.

`train.py` contains 2 types of `Trainer` class. `Trainer` trains on data according to the model specified and runs training and validation. Dataloader is initialized from one of the `colorize_data.py` classes. `colorize_data.py` creates `DataSet` objects for use in the torch dataloader.

## Network Architecture

`basic_network.py` contains 2 types of models: `basic` and `U2Net` 

`basic` network is a simple autoencoder. The encoder consists of first 6 children of Resnet18 which takes in a gray input and the decoder consists of an upsampling network that gives a 3-channel output corresponding to the RGB output.

`U2Net` network which is based on U-Net architecture is a modification of the `basic` network that has skip connections from layers in encoder to layers in decoder. There are 5 layers each in encoder and decoder. In decoder, we concatenate the output from its layer with the output from encoder layer(having same no of channels). This network also takes in gray input and gives a 3-channel RGB output.

## Loss Function and Optimizer

Both the networks used here estimate the values for the rgb images. Thus, the loss function used is a regression loss function. We are trying to reduce the error between the true rgb values to the predicted rgb values. MAE, MSE and Huber loss are suitable for this case. Also wanted to try using Normalised Cross Correlation(NCC) but refrained from doing so because it works even when there is illumination change, which is exactly opposite of what I want. I want to penalise as much as possible if there is any difference while making sure to not leave any outliers. Huber loss is like a mixture of MSE and MAE and perfectly fulfills our need. Experiments showed that Huber loss gave a lower validation error in comparison to mse and mae loss. The optimizer used during training is the Adam Optimizer.

## Data Preprocessing

In the given dataset of 4282 images, 4 images were grayscale. Since we cannot use those 4 for training we exclude them from the training set. `colorize_data.py` contains classes that prepare the color images for training. 

`ColorizeData`: Prepares data for the basic network. The rgb image is transformed to grayscale, resized to 256x256 and normalized with mean=0.5 and std=0.5. The rbg image is resized to 256x256 and normalized with mean=0.5 and std=0.5 for target image.

## Training Results

|     Model    |    Epochs    |  Learning Rate |  Batch Size  |   Loss Func  | Validation Error |
|     basic    |     30       |     0.01       |     64       |     MSE      |      0.057       |
|     basic    |     30       |     0.01       |     64       |     MAE      |      0.16        |
|     basic    |     30       |     0.01       |     64       |    Huber     |      0.0273      |
|     U2Net    |     30       |     0.01       |     64       |     MSE      |      9.86        |
|     U2Net    |     30       |     0.01       |     64       |     MAE      |      0.120       |
|     U2Net    |     30       |     0.01       |     64       |    Huber     |      0.019       |

## Observation

During inference for basic network, the image was not resized to 256x256. This gave good resolution outputs. For the case of U2Net, images were resized. I attribute the lack of quality in the generated images to this.

# Usage
## General Environment Setup
Create a conda environment with pytorch, cuda. 

`$ conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia`

For systems without a dedicated gpu, you may use a CPU version of pytorch.
`$ conda install pytorch torchvision torchaudio cpuonly -c pytorch`

Install scikit-image for image processing and matplotlib for plotting the images
`pip install scikit-image matplotlib scikit-learn pandas tqdm`

## Training
Download the training dataset as `landscape_images/` in the project working directory.

To run training, run `grayscale2color.py` with the following arguments:

`model`: Specify the model to run. Available options are `basic`, `U2Net`

`bs`: Specify the batch size for training

`epochs`: Number of epochs to train

`lr`: Learning rate for training

`loss`: Specify the loss funtion to use. Available options are `mse` and `huber` and `mae`

Example usage: `python grayscale2color.py --model basic --bs 8 --lr 1e-2 --epochs 1 --loss huber`

## Inference

`inference.py` contains different functions to run inference for `basic`, `U2Net` networks. The `inference.py` script generates the color image given a grayscale or color image. If rgb images are given as input, they are converted to gray and fed into the model.

The arguments for this script are as below:

`input`: The input image file location. The generated images are also stored in this location.

`model_pt`: Path to the trained model

`model`: Type of the trained model. Available choices are the same as that of the training script- `basic` and `U2Net`

`image_type`: Type of image. Available choices are `gray` and `rgb`

Example usage: `$ python inference.py --input dir/file_loc.jpg --image_type gray --model_pt basic_colorizer.model --model basic`


## Bonus
You are tasked to control the average color/mood of the image that you are colorizing. What are some ideas that come to your mind? (Bonus: Implement)

I think we can use HSV (Hue, Saturation and Value) to control the mood of the image. Unlike RGB format where changing the pixel values just change the color, HSV color space describes colors (hue or tint) in terms of their shade (saturation or amount of gray) and their brightness value.


