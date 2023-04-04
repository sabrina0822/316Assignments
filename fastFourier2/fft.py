import argparse
import numpy
import matplotlib.pyplot as plt
import matplotlib.colors as plc
import cv2
import math
import cmath
import time
import multiprocessing


# Global variables

# default image is 474 by 630 pixels
image = None
height = 0
width = 0


def dft_1D(vector: numpy.ndarray):
    """
    Calculate the DFT of the given row vecotr input signal
    Input: a vector of pixels
    """
    output_vector = numpy.zeros(len(vector), dtype=complex)

    dft = lambda k, N, n: cmath.exp(-1j * 2 * math.pi * k * n / N)
    N = len(vector)
    for k in range(N):
        for n in range(N):
            output_vector[n] += vector[k] * dft(k, len(vector), n)

    return output_vector


def dft_2D(matrix: numpy.ndarray):
    """
    Calculate the DFT of the given 2D input signal
    Input: a 2D array of values
    """
    # N rows, M columns
    N, M = matrix.shape

    output_matrix = numpy.zeros(matrix.shape, dtype=complex)
    inner_matrix = numpy.zeros(matrix.shape, dtype=complex)

    for n in range(N):
        inner_matrix[n, 0:M] = dft_1D(matrix[n, 0:M])

    for m in range(M):
        output_matrix[0:N, m] = dft_1D(inner_matrix[0:N, m])

    return output_matrix


def fft_1D(row_vector: numpy.ndarray):
    """
    Calculate the FFT of the given row vector input signal
    Input: a vector of pixels
    """

    N = len(row_vector)

    # Make vector size to nearest power of 2
    pow2 = math.ceil(math.log2(N))
    N = 2**pow2

    # Pad vector with zeros
    row_vector = numpy.pad(row_vector, (0, N - len(row_vector)), "constant")

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
            output_vector[k] = even_fft[k] + odd_fft[k] * cmath.exp(
                -1j * 2 * math.pi * k / N
            )
            output_vector[k + N // 2] = even_fft[k] - odd_fft[k] * cmath.exp(
                -1j * 2 * math.pi * k / N
            )

        return output_vector


def fft_2D(matrix: numpy.ndarray):
    """
    Calculate the FFT of the given 2D input signal
    Input: a 2D array of values
    """
    # N rows, M columns
    N, M = matrix.shape

    # Make matrix size to nearest power of 2
    pow2N = math.ceil(math.log2(N))
    N = 2**pow2N

    pow2M = math.ceil(math.log2(M))
    M = 2**pow2M

    # Pad matrix with zeros
    matrix = numpy.pad(
        matrix, ((0, N - matrix.shape[0]), (0, M - matrix.shape[1])), "constant"
    )

    output_matrix = numpy.zeros((N, M), dtype=complex)
    inner_matrix = numpy.zeros((N, M), dtype=complex)

    for n in range(N):
        inner_matrix[n, 0:M] = fft_1D(matrix[n, 0:M])

    for m in range(M):
        output_matrix[0:N, m] = fft_1D(inner_matrix[0:N, m])

    return output_matrix


def inverse_dft_1D(vector: numpy.ndarray):
    """
    Calculate the DFT of the given row vecotr input signal
    Input: a vector of pixels
    """
    output_vector = numpy.zeros(len(vector), dtype=complex)

    dft = lambda k, N, n: cmath.exp(1j * 2 * math.pi * k * n / N)
    N = len(vector)
    for n in range(N):
        for k in range(N):
            output_vector[n] += vector[k] * dft(k, len(vector), n)

    return output_vector


def inverse_fft_1D(row_vector: numpy.ndarray):
    """
    Calculate the FFT of the given row vector input signal
    Input: a vector of pixels
    """

    N = len(row_vector)

    # Make vector size to nearest power of 2
    pow2 = math.ceil(math.log2(N))
    N = 2**pow2

    # Pad vector with zeros
    # row_vector = numpy.pad(row_vector, (0, N - len(row_vector)), 'constant')

    output_vector = numpy.zeros(N, dtype=complex)
    # x[n] = (1/N) * sum(k=0 to N-1) { outut_vector[k] * exp(j * 2*pi * n * k / N) }

    # ! Old version of inverse
    # tmpconj = numpy.conj(row_vector)

    # Xconj = fft_1D(tmpconj)

    # output_vector = Xconj / N

    # return output_vector

    # ! New version of inverse
    # TODO Chose proper base case
    # Base case
    if N == 8:
        return inverse_dft_1D(row_vector)
    else:
        # Split vector into even and odd indices
        even = row_vector[0::2]
        odd = row_vector[1::2]

        # Calculate FFT of even and odd indices
        even_fft = inverse_fft_1D(even)
        odd_fft = inverse_fft_1D(odd)

        # Calculate the output vector
        output_vector = numpy.zeros(N, dtype=complex)
        for k in range(N // 2):
            output_vector[k] = even_fft[k] + odd_fft[k] * cmath.exp(
                1j * 2 * math.pi * k / N
            )
            output_vector[k + N // 2] = even_fft[k] - odd_fft[k] * cmath.exp(
                1j * 2 * math.pi * k / N
            )

        return output_vector / N


def inverse_fft_2D(matrix: numpy.ndarray):
    """
    Calculate the FFT of the given 2D input signal
    Input: a 2D array of values
    """

    # N rows, M columns
    N, M = matrix.shape

    # Make matrix size to nearest power of 2
    pow2N = math.ceil(math.log2(N))
    N = 2**pow2N

    pow2M = math.ceil(math.log2(M))
    M = 2**pow2M

    # Pad matrix with zeros
    matrix = numpy.pad(
        matrix, ((0, N - matrix.shape[0]), (0, M - matrix.shape[1])), "constant"
    )

    output_matrix = numpy.zeros((N, M), dtype=complex)
    inner_matrix = numpy.zeros((N, M), dtype=complex)

    for n in range(N):
        inner_matrix[n, 0:M] = inverse_fft_1D(matrix[n, 0:M])

    for m in range(M):
        output_matrix[0:N, m] = inverse_fft_1D(inner_matrix[0:N, m])

    return output_matrix


def plot_dft(output_matrix, save_as, title):
    """
    Plots the resulting output DFT/FFT on a log scale plot
    """
    # get the magnitude of the complex numbers
    plt.figure()
    plt.imshow(
        numpy.abs(output_matrix),
        norm=plc.LogNorm(),
        cmap=plt.cm.Greys,
        interpolation="none",
    )  # Lognorm to logscale, numpy.abs to get the magnitude of the complex numbers
    plt.suptitle(title)
    plt.colorbar()
    plt.show()
    plt.savefig(save_as + ".svg")  # oracle = numpy.fft.fft2(image, (512, 1024))


def plot_dft_one(output_matrix, title, image):
    """
    Plots the resulting output DFT on a log scale plot besides its original image
    """
    # create a figure with two subplots
    fig, (plot1, plot2) = plt.subplots(1, 2, figsize=(12, 7))

    plot1.imshow(image, cmap="gray")
    plot1.set_title("Original Image")
    # get the magnitude of the complex numbers
    plot2.imshow(
        numpy.abs(output_matrix), norm=plc.LogNorm(), cmap="gray", interpolation="none"
    )
    plot2.set_title("Fast Fourier Transform")
    plt.show()
    plt.savefig(title + ".svg")


def plot_two_transforms(fft1, fft2, title1, title2, save_as):
    """
    Plots two FFT transforms besides each other
    """
    # create a figure with two subplots
    fig, (plot1, plot2) = plt.subplots(1, 2, figsize=(10, 5))

    plot1.imshow(numpy.abs(fft1), norm=plc.LogNorm(), cmap="gray", interpolation="none")
    plot1.set_title(title1)
    # get the magnitude of the complex numbers
    plot2.imshow(numpy.abs(fft2), norm=plc.LogNorm(), cmap="gray", interpolation="none")
    plot2.set_title(title2)
    plt.show()
    plt.savefig(save_as + ".svg")


def filter_dft(output):
    keep_fraction = 0.1
    modified_output = output.copy()
    rows, columns = modified_output.shape
    # Set r and c to be the number of rows and columns of the array.

    # Set to zero all rows with indices between r*keep_fraction and
    # r*(1-keep_fraction):
    modified_output[int(rows * keep_fraction) : int(rows * (1 - keep_fraction)), :] = 0

    # Similarly with the columns:
    modified_output[
        :, int(columns * keep_fraction) : int(columns * (1 - keep_fraction))
    ] = 0

    return modified_output


def save_dft(output_matrix, title):
    """
    Saves the resulting output DFT on a csv or txt file
    """
    numpy.savetxt(title + ".csv", output_matrix, delimiter=",")


def upper_threshold(sorted_fft, p, fft):
    """
    Given a sorted matrix (lowest to highest), take the top percent of values and zero all other values
    """

    threshold = sorted_fft[int(p * len(sorted_fft))]

    lower_values = numpy.abs(fft) > threshold

    return fft * lower_values


def compress_two_threshold(sorted_fft, upper, lower, fft):
    upper_threshold = sorted_fft[int(upper * len(sorted_fft))]
    lower_threshold = sorted_fft[int((1 - lower) * len(sorted_fft))]

    set_zeros = fft
    mask = (numpy.abs(fft) < lower_threshold) & (numpy.abs(fft) > upper_threshold)
    set_zeros[mask] = 0
    zeros = 0
    for s in set_zeros:
        for k in s:
            if k == 0:
                zeros = zeros + 1

    print(
        "Compression level: "
        + str(int(upper * 100))
        + " Number of zeros: "
        + str(zeros)
    )

    return fft


def compress_fft(fft, percent):
    """
    Given the fourier coefficients of an image, remove the highest coefficients
    """
    # start by sorting all of the coefficients
    # fft = fft[0:height,0:width]
    sorted_fft = numpy.sort(numpy.abs(fft.reshape(-1)))

    set_zero = upper_threshold(sorted_fft, percent, fft)

    zeros = 0
    for s in set_zero:
        for k in s:
            if k == 0:
                zeros = zeros + 1

    print(
        "Compression level: "
        + str(int(percent * 100))
        + " Number of zeros: "
        + str(zeros)
    )

    inverted = inverse_fft_2D(set_zero)
    inverted = inverted[0:height, 0:width]

    return inverted


def compress_two(fft, percent):
    """
    given fft, keep the lowest frequencies and a fraction of the largest coefficient
    """
    # start by sorting all of the coefficients
    # fft = fft[0:height,0:width]
    sorted_fft = numpy.sort(numpy.abs(fft.reshape(-1)))

    set_zero = compress_two_threshold(sorted_fft, percent, 0.1, fft)

    inverted = inverse_fft_2D(set_zero)
    inverted = inverted[0:height, 0:width]

    return inverted


def compressed_subplot(fft):
    fig, axs = plt.subplots(2, 3, figsize=(12, 7))
    percents = [0, 0.19, 0.38, 0.57, 0.76, 0.95]
    count = 0
    N, M = fft.shape

    for i in range(2):
        for j in range(3):
            percent = percents[count]
            title = str(int(percent * 100)) + "% " + "compression"
            compressed = compress_fft(fft, percent)
            axs[i, j].imshow(
                numpy.abs(compressed),
                norm=plc.LogNorm(),
                cmap="gray",
                interpolation="none",
            )
            axs[i, j].set_title(title)

            csv_name = str(int(percent * 100)) + "_compress"
            # save_dft(compressed, csv_name)
            plt.savefig("compressed_subplot" + str(count) + ".svg")

            count = count + 1
    plt.show()
    plt.savefig("compressed_subplot.svg")
    return


def test_transforms(thread_num):
    # create 2D array of size 2^5 to 2^10
    dft_res = [0] * 6
    fft_res = [0] * 6
    for i in range(5, 11):
        size = 2**i
        array = numpy.random.random((size, size))

        # perform 2D DFT
        dft_start = time.time()
        dft_2D(array)
        dft_end = time.time()
        # compute time taken
        print(f"{thread_num}: DFT - 2^{i}: {dft_end - dft_start} s")
        dft_res[i - 5] = dft_end - dft_start

        # perform 2D FFT
        fft_start = time.time()
        fft_2D(array)
        fft_end = time.time()
        # compute time taken
        print(f"{thread_num}: FFT - 2^{i}: {fft_end - fft_start} s")
        fft_res[i - 5] = fft_end - fft_start

    return dft_res, fft_res


def mode_one(image):
    """
    Given an image, perform a 2D FFT on the image and then plot the resulting output DFT on a log scale plot
    """
    output = fft_2D(image)
    output = output[0:height, 0:width]

    plot_dft_one(output, "fft", image)

    return output


def mode_two(image):
    output = fft_2D(image)

    # method 2
    filtered = filter_dft(output)

    im_new = inverse_fft_2D(filtered).real
    # expected = numpy.fft.ifft2(filtered).real
    plot_dft(im_new[0:height, 0:width], "denoised", "Denoised Image")
    # plot_dft(expected[0:height,0:width], 'expected', "Expected Inverse")


def mode_three(image):
    """
    Compress the image with percents ranging from 0-95%
    """
    fft = fft_2D(image)
    # compress_fft(fft, .76)
    compressed_subplot(fft)
    return


def mode_four():
    """
    Performance testing for 2D transforms between 2^5 and 2^10
    """
    max_matrix_size = 11

    # multiprocessing for performance testing
    with multiprocessing.Pool(10) as pool:
        results = pool.map(test_transforms, range(10))

    # create matrix for data
    dft_data = numpy.zeros((6, 10))
    fft_data = numpy.zeros((6, 10))
    # put results in 2d array
    for i in range(10):
        for j in range(5, max_matrix_size):
            dft_data[j - 5, i] = results[i][0][j - 5]
            fft_data[j - 5, i] = results[i][1][j - 5]

    avg_dft = numpy.zeros(6)
    std_dft = numpy.zeros(6)
    avg_fft = numpy.zeros(6)
    std_fft = numpy.zeros(6)

    # calculate average and std
    print("DFT:\tSize \tMean \t\t\t\tStd:")
    for i in range(5, max_matrix_size):
        avg_dft[i - 5] = numpy.average(dft_data[i - 5, :])
        std_dft[i - 5] = numpy.std(dft_data[i - 5, :])
        print(
            f"\t2^{i} \t{numpy.average(dft_data[i-5, :])} \t\t{numpy.std(dft_data[i-5, :])}"
        )

    print("\nFFT:\tSize \tMean \t\t\t\tStd")
    for i in range(5, max_matrix_size):
        avg_fft[i - 5] = numpy.average(fft_data[i - 5, :])
        std_fft[i - 5] = numpy.std(fft_data[i - 5, :])
        print(
            f"\t2^{i} \t{numpy.average(fft_data[i-5, :])} \t\t{numpy.std(fft_data[i-5, :])}"
        )

    # compute error bars
    std_dft = std_dft * 2
    std_fft = std_fft * 2

    # plot data with error bars
    plt.figure()
    plt.errorbar(list(range(5, 11)), avg_dft, yerr=std_dft, label="DFT")
    plt.errorbar(list(range(5, 11)), avg_fft, yerr=std_fft, label="FFT")
    plt.title("Performance of 2D DFT and FFT")
    plt.xlabel("Size of the 2D array (2^x)")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.savefig("performance.png")
    plt.show()


def collect_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=int, default=1)
    parser.add_argument("-i", type=str, default="moonlanding.png")

    return parser.parse_args()


def get_image(image_name):
    global image, height, width
    # read image as grayscale
    image = cv2.imread(image_name, 0)
    height, width = image.shape


if __name__ == "__main__":
    args = collect_args()

    mode = args.m
    image_name = args.i

    get_image(image_name)

    vector = numpy.array(image[0, 0:630])

    # pad image with zeros
    image2 = numpy.pad(
        image, ((0, 512 - image.shape[0]), (0, 1024 - image.shape[1])), "constant"
    )

    match mode:
        case 1:
            print("Mode 1")
            mode_one(image)
        case 2:
            print("Mode 2")
            mode_two(image)
        case 3:
            print("Mode 3")
            mode_three(image)
        case 4:
            print("Mode 4")
            mode_four()
        case _:
            print("Invalid mode")

