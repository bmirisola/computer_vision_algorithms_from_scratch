import numpy as np

def generate_gaussian_kernel(sigma):
    kernel = np.linspace(-3 * sigma, 3 * sigma, 6 * sigma + 1)
    kernel = np.exp(-0.5 * np.square(kernel) / (sigma ** 2))
    kernel = np.outer(kernel, kernel)
    return kernel / np.sum(kernel)


def guassian_filter(image: np.ndarray, sigma):
    kernel = generate_gaussian_kernel(sigma)
    output_image = np.pad(image, (3 * sigma, 3 * sigma), 'edge')
    output_image = convolution2d(image, output_image, kernel, sigma)
    return output_image


def convolution2d(original_image, image: np.ndarray, kernel, sigma):
    image_row, image_col = image.shape
    output_image = np.matrix.copy(original_image)
    for r in range(3 * sigma, image_row - 3 * sigma):
        for c in range(3 * sigma, image_col - 3 * sigma):
            output_image[r - 3 * sigma, c - 3 * sigma] = np.sum(
                np.multiply(kernel, image[r - 3 * sigma:r + 3 * sigma + 1, c - 3 * sigma:c + 3 * sigma + 1]))
    return output_image

def sobel_filter(image:np.ndarray):
    padded_image = np.pad(image, (1, 1), 'edge')

    vertical_mask = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    horizontal_mask = np.flip(vertical_mask.T, axis=0)

    output_image_row, output_image_col = padded_image.shape
    input_image_row, input_image_col = image.shape
    gx = np.ndarray((input_image_row, input_image_col))
    gy = np.ndarray((input_image_row, input_image_col))

    for r in range(1, output_image_row - 1):
        for c in range(1, output_image_col - 1):
            gx[r - 1, c - 1] = np.sum(np.multiply(vertical_mask, padded_image[r - 1:r + 2, c - 1:c + 2]))
            gy[r - 1, c - 1] = np.sum(np.multiply(horizontal_mask, padded_image[r - 1:r + 2, c - 1:c + 2]))

    return gx, gy

def non_maximum_supression(filtered_image: np.ndarray):
    input_image_row, input_image_col = filtered_image.shape
    suppressed_image = np.zeros((input_image_row, input_image_col))
    for r in range(1, input_image_row - 1):
        for c in range(1, input_image_col - 1):
            pixel_value = filtered_image[r][c]
            kernel = filtered_image[r - 1:r + 2, c - 1:c + 2]
            max = kernel.max()
            if max > pixel_value:
                suppressed_image[r, c] = 0
            else:
                suppressed_image[r, c] = pixel_value

    return suppressed_image