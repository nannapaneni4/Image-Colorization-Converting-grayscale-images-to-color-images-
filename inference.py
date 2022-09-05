import torch
import torchvision.transforms as T
from torchvision.utils import save_image
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

"""
Functions to generate color images are defined here.
Input images can be of gray scale or rgb. In case of rgb, the image is converted to gray and then fed to the model
The model to use can be the basic network or U-Net based network
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

input_rgb2gray_transform = T.Compose([T.ToTensor(),
                                 T.Grayscale(),
                                 T.Resize(size=(1024, 1024)),
                             ])

def _parse_args():
    """
    Command-line arguments to the system.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='inference.py')

    # General system running and configuration options
    parser.add_argument('--input', type=str, help='location of the image')
    parser.add_argument('--model_pt', type=str, help='path to the trained model')
    parser.add_argument('--model', type=str, default='basic', help='model type to run')
    parser.add_argument('--image_type', type=str, default='gray', help='input image type')

    args = parser.parse_args()
    return args

def save_gen_image(img, file):
    """
    Saves the img at location file
    Parameters:
      img: Image to save
      file: Image file name
    """
    print("saving the generated image in the dir of input image")
    f = file.split('.')[0]
    f += '_out.jpg'
    save_image(img, f)

def display_images(gray, color):
    """
    Display the Model Input and Output
    Parameters:
      file: Gray scale image to convert
      model_pt: Path to the model to use
    """
    f, axarr = plt.subplots(1, 2, figsize=(20, 10), dpi=200)
    axarr[0].imshow(gray, cmap="gray")
    axarr[0].set_title("Grayscale Image (Model Input)")
    axarr[1].imshow(color)
    axarr[1].set_title("RGB Image (Model Output)")
    plt.show()

def generate_gray2color(file, model_pt):
    """
    Generates color image from gray scale images using the basic network
    The predicted color image is stored as well as displayed
    Parameters:
      file: Gray scale image to convert
      model_pt: Path to the model to use
    """
    print("loading model")
    model = torch.load(model_pt, map_location=device)
    model.eval()
    img = Image.open(file)
    try:
        input = T.ToTensor()(img).unsqueeze(0).to(device)
    except:
        raise TypeError("Image not of grayscale type")
    with torch.no_grad():
        output = model(input)
    output = output.cpu()
    input = input.cpu()
    save_gen_image(output, file)
    display_images(input.squeeze(0).permute(1, 2, 0), output.squeeze(0).permute(1, 2, 0))


def generate_color2color(color_file, model_pt):
    """
    Generates color image from color images using the basic network. The input RGB image is first converted to gray scale.
    The predicted color image is stored as well as displayed
    Parameters:
      file: Gray scale image to convert
      model_pt: Path to the model to use
    """
    print("loading model")
    model = torch.load(model_pt, map_location=device)
    model.eval()
    img = Image.open(color_file)
    try:
        input = input_rgb2gray_transform(img).unsqueeze(0).to(device)
    except:
        raise TypeError("Image not of rgb type")
    print("running model inference")
    with torch.no_grad():
        output = model(input)
    output = output.cpu()
    input = input.cpu()
    save_gen_image(output, color_file)
    display_images(input.squeeze(0).permute(1, 2, 0), output.squeeze(0).permute(1, 2, 0))

if __name__ == "__main__":

    args = _parse_args()
    if args.image_type == 'rgb':
        generate_color2color(args.input, args.model_pt)
    elif args.image_type == 'gray':
        generate_gray2color(args.input, args.model_pt)