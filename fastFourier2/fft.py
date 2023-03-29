import argparse
import numpy
import matplotlib
import cv2
import math

# Global variables
image = None

def dft_1D(row_vector : numpy.ndarray): 
    """
    Calculate the DFT of the givne input signal
    Input: a vector of pixels
    """    
    output_vector = numpy.zeros(len(row_vector))

    dft = lambda k, N, n: math.exp(-1j * 2 * math.pi * k * n / N); 

    for n in range (0, len(row_vector)):
        for k in range (0, len(row_vector)):
            output_vector[n] += row_vector[k] * dft(k, len(row_vector), n)      


# def dft_2D(): 

# def fft_1D(): 

# def fft_2D(): 

# def inverse_fft_1D(): 

# def inverse_fft_2D():

# def plot_dft(): 

# def save_dft():


def collect_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str, default="1")
    parser.add_argument('-i', type=str, default="moonlanding.png")

    return  parser.parse_args()

def get_image(image_name):
    global image
    # read image as grayscale
    image = cv2.imread(image_name, 0)

if __name__ == "__main__":
    args = collect_args()
    
    mode = args.m
    image_name = args.i

    get_image(image_name)