import argparse
import numpy
import matplotlib
import cv2
import math
import cmath
import threading

# Global variables

# default image is 474 by 630 pixels
image = None

def dft_1D(row_vector : numpy.ndarray): 
    """
    Calculate the DFT of the given row vecotr input signal
    Input: a vector of pixels
    """  
    output_vector = numpy.zeros(len(row_vector), dtype=complex)

    dft = lambda k, N, n: cmath.exp(-1j * 2 * math.pi * k * n / N); 
    N = len(row_vector)
    for k in range(N):
        for n in range(N):
            output_vector[n] += row_vector[k] * dft(k, len(row_vector), n)   

    return output_vector

 
def dft_2D(matrix : numpy.ndarray):
    global verbose
    """
    Calculate the DFT of the given 2D input signal
    Input: a 2D array of values
    """ 
    if __debug__:
        print(f"DFT 2D - Input Matrix:\n{matrix}\n\n")

    # N rows, M columns
    N, M = matrix.shape


    output_matrix = numpy.zeros(matrix.shape, dtype=complex)
    inner_matrix = numpy.zeros(matrix.shape, dtype=complex)


    dft_1d = lambda k, N, n: cmath.exp(-1j * 2 * math.pi * k * n / N)
        
    if __debug__:
        print("Row: ", flush=True, end="")

    for n in range(N):
        if __debug__:
            print(f"{n} ", flush=True, end=" ")
        inner_matrix[n, 0:M] = dft_1D(matrix[n, 0:M])

    print(f"\n\nCol: ", flush=True, end="")
    for m in range(M):
        if __debug__:
            print(f"{m}", flush=True, end=" ")
        output_matrix[0:N, m] = dft_1D(inner_matrix[0:N, m])

    if __debug__:
        print(f"DFT 2D - Output Matrix:\n{output_matrix}\n\n")
    return output_matrix

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

    vector = numpy.array(image[0, 0:630])

    dft_1D(vector)

    result = dft_2D(image)

    oracle = numpy.fft.fft2(image)

    print(f"Oracle:\n{oracle}\n\n")