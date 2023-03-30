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
        
    if __debug__:
        print("Row: ", flush=True, end="")

    for n in range(N):
        if __debug__:
            print(f"{n} ", flush=True, end=" ")
        inner_matrix[n, 0:M] = dft_1D(matrix[n, 0:M])

    if __debug__:
        print(f"\n\nCol: ", flush=True, end="")

    for m in range(M):
        if __debug__:
            print(f"{m}", flush=True, end=" ")
        output_matrix[0:N, m] = dft_1D(inner_matrix[0:N, m])

    if __debug__:
        print(f"\n\nDFT 2D - Output Matrix:\n{output_matrix}\n\n")

    return output_matrix

def fft_1D(row_vector : numpy.ndarray):
    """
    Calculate the FFT of the given row vector input signal
    Input: a vector of pixels
    """

    N = len(row_vector)

    # Make vector size to nearest power of 2
    pow2 = math.ceil(math.log2(N))
    N = 2 ** pow2

    # Pad vector with zeros
    row_vector = numpy.pad(row_vector, (0, N - len(row_vector)), 'constant')

    output_vector = numpy.zeros(N, dtype=complex)

    # TODO Chose proper base case
    # Base case
    if N == 8:
        return dft_1D(row_vector)
    else:
        # Split vector into even and odd indices
        even = row_vector[0::2]
        odd = row_vector[1::2]

        # Calculate FFT of even and odd indices
        even_fft = fft_1D(even)
        odd_fft = fft_1D(odd)

        # Calculate the output vector
        output_vector = numpy.zeros(N, dtype=complex)
        for k in range(N // 2):
            output_vector[k] = even_fft[k] + odd_fft[k] * cmath.exp(-1j * 2 * math.pi * k / N)
            output_vector[k + N//2] = even_fft[k] - odd_fft[k] * cmath.exp(-1j * 2 * math.pi * k / N)

        return output_vector

def fft_2D(matrix : numpy.ndarray): 
    """
    Calculate the FFT of the given 2D input signal
    Input: a 2D array of values
    """ 
    if __debug__:
        print(f"FFT 2D - Input Matrix:\n{matrix}\n\n")

    # N rows, M columns
    N, M = matrix.shape

    # Make matrix size to nearest power of 2
    pow2N = math.ceil(math.log2(N))
    N = 2 ** pow2N

    pow2M = math.ceil(math.log2(M))
    M = 2 ** pow2M

    # Pad matrix with zeros
    matrix = numpy.pad(matrix, ((0, N - matrix.shape[0]), (0, M - matrix.shape[1])), 'constant')

    output_matrix = numpy.zeros((N, M), dtype=complex)
    inner_matrix = numpy.zeros((N, M), dtype=complex)
        
    if __debug__:
        print("Row: ", flush=True, end="")

    for n in range(N):
        if __debug__:
            print(f"{n} ", flush=True, end=" ")
        inner_matrix[n, 0:M] = fft_1D(matrix[n, 0:M])

    if __debug__:
        print(f"\n\nCol: ", flush=True, end="")

    for m in range(M):
        if __debug__:
            print(f"{m}", flush=True, end=" ")
        output_matrix[0:N, m] = fft_1D(inner_matrix[0:N, m])

    if __debug__:
        print(f"\n\nFFT 2D - Output Matrix:\n{output_matrix}\n\n")
        
    return output_matrix

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

    # pad image with zeros
    image2 = numpy.pad(image, ((0, 512 - image.shape[0]), (0, 1024 - image.shape[1])), 'constant')
    fft_2D(image)
    dft_2D(image2)


    oracle = numpy.fft.fft2(image, (1024, 1024))

    print(f"Oracle:\n{oracle}\n\n")