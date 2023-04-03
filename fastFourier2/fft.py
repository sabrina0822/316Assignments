import argparse
import numpy
import matplotlib.pyplot as plt 
import matplotlib.colors as plc
import cv2
import math
import cmath
import threading

# Global variables

# default image is 474 by 630 pixels
image = None

def dft_1D(vector : numpy.ndarray): 
    """
    Calculate the DFT of the given row vecotr input signal
    Input: a vector of pixels
    """  
    output_vector = numpy.zeros(len(vector), dtype=complex)

    dft = lambda k, N, n: cmath.exp(-1j * 2 * math.pi * k * n / N); 
    N = len(vector)
    for k in range(N):
        for n in range(N):
            output_vector[n] += vector[k] * dft(k, len(vector), n)   

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
            if n % 10 == 0:
                print(f"{n} ", flush=True, end=" ")
        inner_matrix[n, 0:M] = dft_1D(matrix[n, 0:M])

    if __debug__:
        print(f"\n\nCol: ", flush=True, end="")

    for m in range(M):
        if __debug__:
            if m % 10 == 0:
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
            if n % 10 == 0:
                print(f"{n}", flush=True, end=" ")
        inner_matrix[n, 0:M] = fft_1D(matrix[n, 0:M])

    if __debug__:
        print(f"\n\nCol: ", flush=True, end="")

    for m in range(M):
        if __debug__:
            if m % 10 == 0:
                print(f"{m}", flush=True, end=" ")
        output_matrix[0:N, m] = fft_1D(inner_matrix[0:N, m])

    if __debug__:
        print(f"\n\nFFT 2D - Output Matrix:\n{output_matrix}\n\n")
        
    return output_matrix

def inverse_dft_1D(vector: numpy.ndarray): 
    """
    Calculate the DFT of the given row vecotr input signal
    Input: a vector of pixels
    """  
    output_vector = numpy.zeros(len(vector), dtype=complex)

    dft = lambda k, N, n: cmath.exp(1j * 2 * math.pi * k * n / N); 
    N = len(vector)
    for n in range(N):
        for k in range(N):
            output_vector[n] += vector[k] * dft(k, len(vector), n)   

    return output_vector 

def inverse_fft_1D(row_vector : numpy.ndarray): 
    """
    Calculate the FFT of the given row vector input signal
    Input: a vector of pixels
    """

    N = len(row_vector)

    # Make vector size to nearest power of 2
    pow2 = math.ceil(math.log2(N))
    N = 2 ** pow2

    # Pad vector with zeros
    #row_vector = numpy.pad(row_vector, (0, N - len(row_vector)), 'constant')

    output_vector = numpy.zeros(N, dtype=complex)
    x[n] = (1/N) * sum(k=0 to N-1) { outut_vector[k] * exp(j * 2*pi * n * k / N) }

    tmpconj = numpy.conj(row_vector)

    Xconj = fft_1D(tmpconj)

    output_vector = Xconj / N

    return output_vector 

def inverse_fft_2D(matrix : numpy.ndarray):
    """
    Calculate the FFT of the given 2D input signal
    Input: a 2D array of values
    """ 
    if __debug__:
        print(f"Inverse FFT 2D - Input Matrix:\n{matrix}\n\n")

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
            print(f"{n}", flush=True, end=" ")
        inner_matrix[n, 0:M] = inverse_fft_1D(matrix[n, 0:M])

    if __debug__:
        print(f"\n\nCol: ", flush=True, end="")

    for m in range(M):
        if __debug__:
            print(f"{m}", flush=True, end=" ")
        output_matrix[0:N, m] = inverse_fft_1D(inner_matrix[0:N, m])

    if __debug__:
        print(f"\n\nInverse FFT 2D - Output Matrix:\n{output_matrix}\n\n")
        
    return output_matrix

def plot_dft(output_matrix, title):
    """
    Plots the resulting output DFT on a log scale plot
    """
    #get the magnitude of the complex numbers
    plt.imshow(numpy.abs(output_matrix), norm=plc.LogNorm(), cmap=plt.cm.Greys, interpolation='none')
    plt.colorbar()
    plt.show()
    plt.savefig(title + '.svg')    # oracle = numpy.fft.fft2(image, (512, 1024))

def filter_dft(output): 
    keep_fraction = 0.1
    modified_output = output.copy()
    rows, columns = modified_output.shape
    # Set r and c to be the number of rows and columns of the array.

    # Set to zero all rows with indices between r*keep_fraction and
    # r*(1-keep_fraction):
    modified_output[int(rows*keep_fraction):int(rows*(1-keep_fraction))] = 0

    # Similarly with the columns:
    modified_output[:, int(columns*keep_fraction):int(columns*(1-keep_fraction))] = 0

    plt.figure()
    plot_dft(modified_output, 'filtered')
    plt.show()

    return modified_output

def save_dft(output_matrix):
    """
    Saves the resulting output DFT on a csv or txt file
    """
    numpy.savetxt('2d_dft.csv', output_matrix, delimiter=',')


def mode_one(image): 
    """
    Given an image, perform a 2D FFT on the image and then plot the resulting output DFT on a log scale plot
    """
    output = fft_2D(image)
    output = output[0:474,0:630]
    plot_dft(output, 'fft')

    return output

def mode_two(image): 
    output = fft_2D(image)
    #output = numpy.fft.fft2(image)
    #output = output[0:474,0:630]
    filtered = filter_dft(output)

    im_new = inverse_fft_2D(filtered).real
    expected = numpy.fft.ifft2(filtered).real
    plot_dft(im_new[0:474,0:630], 'reconstructed')
    plot_dft(expected[0:474,0:630], 'expected')

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

    #mode_one(image)
    mode_two(image)
    #print(numpy.real(output))
    # matrix = fft_2D(image)
    #matrix = output2[0:474,0:630]
    # print("about to plot")

    print("here")


"""
Notes for sabrina: 


for each frequency: the magnitude (abs value) of the complex value represents the amplitude 
of a constintuent complex sinusoid at that frequency integrated over the domain 

the argument of the complex value represents the phase of the same sinusoid

Mode 1: 
    Perform FFT 
    Output a one by two subplot of the original image 
    besides the subplot output its fourier transform 

    Fourier transform needs to be log scaled --> LogNorm from matplotlib.colours
    Produces a logarithmic colour map???


Mode 2: 
    Output a 1x2 subplot of original image next to its denoised version

    denoised version: 
    take the FFt of the image and set all high frequencies to zero before 
    inverting back to the filtered original 

    "high" frequencies can be determined from trial and error 

    FFT is from 0 to 2pi --> any frequency close to 0 or 2 pi is considered low 

    Program should also print the number of non-zeroes using and the fraction 
    they represent of th eoriginal Fourier coefficients 

Mode 3: 
    Take the FFT of the image to compress it 
    Compression comes from setting some fourier coefficients to 0 
        1.can threshold the coefficient's magnitude and take only the largest percentile of them?????
        2.keep all the coefficients of very low frequencies as well as a fraction of the largest coefficients 
        from higher frequencies to also filter the image at the same time 

        Experiment with various schemes and decide what works best and justify it in the report 
    
    Then 
    Display a 2x3 subplot of the image at 6 different compression levels 
        start from original image (0%) 
        to setting 95% of the coefficients to 0 

    Get the images by inverse transofrming the modified Fourier coefficients 

    Save the Fourier transform matrix of coefficients in a csv, txt or soemthing else 
    Program should print the number of nonszero Fourier coefficients in each of the 6 images 

Mode 4: 
    Produce plots that summarize the runtime complexity of your algorithms 
    Code should print the means and variances of the runtime of your algorithms vs the problem size
"""