# Assignment 1 - Group 5
Sabrina Mansour: 260945807  
Mathieu Geoffroy: 260986559  


## Information
Python version: 3.10.2  

Run the file with the following command and arguments below: 
```
python fft [-m mode] [-i image] 
```

-m : optional mode 
    - [1] (Default) for fast mode where the image is converted into its FFT form and displayed 
    - [2] for denoising where the image is denoised by applying an FFT, truncating high frequencies and then displayed
    - [3] for compressing and saving the image 
    - [4] for plotting the runtime 
-i: image (optional) - filename of the image we wish to take the DFT of.

Libaries needed to run the code: 
- numpy
- matplotlib 
- cv2 (open-cv)
- cmath
- time 
- multiprocessing