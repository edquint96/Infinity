import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('RX.png')


# New dummy image that will contain the adjustments
img2= np.zeros(img.shape, img.dtype)

# Defining alpha and beta:
alpha = 1  # Contrast Control [1.0-3.0]
beta = 10   # Brightness Control [0-100]

# Scaling and converting the image contrast and brightness
img2= cv.convertScaleAbs(img, alpha=alpha, beta=beta)

# Displaying the adusted image
cv.imshow('Resulting Image', img2)
img = cv.imread('RX.png')
cv.imshow('Original', img)

blank = np.zeros(img2.shape[:2], dtype='uint8')

center_coordinates = (155, 155)

radius = 110

mask = cv.circle(blank, center_coordinates, radius, 255, -1)

masked = cv.bitwise_and(img, img, mask=mask)
cv.imshow('Output', masked)


imgprocesed= cv.fastNlMeansDenoisingColored(masked,None,10,10,9,21)
plt.subplot(121),plt.imshow(img)
plt.subplot(122),plt.imshow(imgprocesed)
plt.show()
