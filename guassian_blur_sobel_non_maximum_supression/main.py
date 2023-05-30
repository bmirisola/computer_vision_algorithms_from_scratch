import cv2
import sys

import numpy
import numpy as np

def generate_gaussian_kernel(sigma):

    kernel = np.linspace(-3*sigma, 3 * sigma, 6*sigma+1)
    kernel = np.exp(-0.5 * np.square(kernel) / (sigma**2))
    kernel = np.outer(kernel,kernel)
    return kernel / np.sum(kernel)

def guassian_filter(image: np.ndarray, sigma):
    kernel = generate_gaussian_kernel(sigma)
    output_image = np.pad(image, (3*sigma, 3*sigma), 'edge')
    output_image = convolution2d(image,output_image,kernel,sigma)
    return output_image


def convolution2d(original_image, image: np.ndarray, kernel,sigma):
    image_row, image_col = image.shape
    output_image = numpy.matrix.copy(original_image)
    for r in range(3*sigma,image_row-3*sigma):
        for c in range(3*sigma, image_col-3*sigma):
           output_image[r-3*sigma,c-3*sigma] = np.sum(np.multiply(kernel,image[r-3*sigma:r + 3 * sigma + 1, c-3*sigma:c + 3 * sigma + 1]))
    return output_image

def sobel_filter(image: np.ndarray, threshold):
    padded_image = np.pad(image, (1, 1), 'edge')

    vertical_mask = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    horizontal_mask = np.flip(vertical_mask.T, axis=0)

    output_image_row, output_image_col = padded_image.shape
    input_image_row, input_image_col = image.shape
    gx = np.ndarray((input_image_row,input_image_col))
    gy = np.ndarray((input_image_row,input_image_col))

    for r in range(1, output_image_row-1):
        for c in range(1, output_image_col-1):
           gx[r-1,c-1] = np.sum(np.multiply(vertical_mask, padded_image[r-1:r+2, c-1:c + 2]))
           gy[r-1,c-1] = np.sum(np.multiply(horizontal_mask, padded_image[r-1:r+2, c-1:c + 2]))

    gradient_magnitude = np.sqrt(np.square(gx) + np.square(gy))
    normalization = 255.0 / gradient_magnitude.max()
    gradient_magnitude = np.multiply(gradient_magnitude,normalization)
    gradient_magnitude = gradient_magnitude.astype(np.uint8)

    for r in range(0, input_image_row):
        for c in range(0, input_image_col):
            if gradient_magnitude[r, c] <= threshold:
                gradient_magnitude[r, c] = 0


    gradient_direction = np.arctan2(gy,gx)
    gradient_direction = np.multiply(180/np.pi, gradient_direction)

    return gradient_magnitude,gradient_direction

def non_maximum_supression(filtered_image: np.ndarray, gradients: np.ndarray):

    input_image_row, input_image_col = filtered_image.shape
    suppressed_image = np.zeros((input_image_row,input_image_col))

    for r in range(1, input_image_row-1):
        for c in range(1, input_image_col-1):
            angle = gradients[r,c]

            if (angle>=-22.5 and angle<=22.5) or (angle<-157.5 and angle >=-180):
                if (filtered_image[r,c] >= filtered_image[r,c+1]) and (filtered_image[r,c] >= filtered_image[r,c-1]):
                    suppressed_image[r,c] = filtered_image[r,c]
                else:
                    suppressed_image[r,c] = 0
            elif (angle>=22.5 and angle<=67.5) or (angle<-112.5 and angle >=-157.5):
                if (filtered_image[r,c] >= filtered_image[r+1,c+1]) and (filtered_image[r,c] >= filtered_image[r-1,c-1]):
                    suppressed_image[r,c] = filtered_image[r,c]
                else:
                    suppressed_image[r,c] = 0
            elif (angle>=67.5 and angle<=112.5) or (angle<-67.5 and angle >=-112.5):
                if (filtered_image[r,c] >= filtered_image[r+1,c]) and (filtered_image[r,c] >= filtered_image[r-1,c]):
                    suppressed_image[r,c] = filtered_image[r,c]
                else:
                    suppressed_image[r,c] = 0
            elif (angle>=112.5 and angle<=157.5) or (angle<-22.5 and angle >=-67.5):
                if (filtered_image[r,c] >= filtered_image[r+1,c-1]) and (filtered_image[r,c] >= filtered_image[r-1,c+1]):
                    suppressed_image[r,c] = filtered_image[r,c]
                else:
                    suppressed_image[r,c] = 0
    return suppressed_image


if __name__ == "__main__":

    if(len(sys.argv) == 1):
        input_image = "pics/plane.pgm"
        sigma = 1
        threshold = 50
    else:
        input_image = sys.argv[1]
        sigma = int(sys.argv[2])
        threshold = int(sys.argv[3])

    original_image = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
    kernel = generate_gaussian_kernel(sigma)

    blurred_image = guassian_filter(original_image,sigma)
    sobel_filtered_image,sobel_filtered_directions = sobel_filter(blurred_image, threshold)
    suppressed_im = non_maximum_supression(sobel_filtered_image,sobel_filtered_directions)

    non_maximum_supression(sobel_filtered_image,sobel_filtered_directions)
    photo = "red"
    cv2.imshow("Original", original_image)
    cv2.imshow("Blurred", blurred_image)
    cv2.imshow("sobel", sobel_filtered_image)
    cv2.imshow("suppression", suppressed_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
