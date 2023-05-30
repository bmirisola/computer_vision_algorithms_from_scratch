import cv2
import random
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


def hessian_detector(image: np.ndarray, threshold):
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

    gx_padded_image = np.pad(gx, (1, 1), 'edge')
    gy_padded_image = np.pad(gy, (1, 1), 'edge')

    gxx = np.ndarray((input_image_row, input_image_col))
    gyy = np.ndarray((input_image_row, input_image_col))

    for r in range(1, output_image_row - 1):
        for c in range(1, output_image_col - 1):
            gxx[r - 1, c - 1] = np.sum(np.multiply(vertical_mask, gx_padded_image[r - 1:r + 2, c - 1:c + 2]))
            gyy[r - 1, c - 1] = np.sum(np.multiply(horizontal_mask, gy_padded_image[r - 1:r + 2, c - 1:c + 2]))

    gxy = np.multiply(gx, gy)
    gxy_2 = np.square(gxy)

    determinant = np.subtract(np.multiply(gxx, gyy), gxy_2)

    for r in range(0, input_image_row):
        for c in range(0, input_image_col):
            if determinant[r, c] <= threshold:
                determinant[r, c] = 0

    determinant = non_maximum_supression(determinant)

    count = 0
    for r in range(0, input_image_row):
        for c in range(0, input_image_col):
            if determinant[r, c] != 0:
                count += 1

    print("count is " + str(count))

    return determinant.astype(np.uint8)


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


def hough_transform(image: np.ndarray, original_image_3_channels: np.ndarray):
    input_image_row, input_image_col = image.shape
    theta_max = 180
    thetas = np.deg2rad(np.arange(0, theta_max, step=1))
    diag_len = int(np.ceil(np.sqrt(np.square(input_image_row) + np.square(input_image_col))))

    num_thetas = len(thetas)
    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint8)

    cosines = np.cos(thetas)
    sines = np.sin(thetas)
    for r in range(0, input_image_row):
        for c in range(0, input_image_col):
            if image[r, c] != 0:
                for t in range(0, theta_max):
                    rho = round(c * cosines[t] + r * sines[t])

                    accumulator[rho, t] = accumulator[rho, t] + 1

    accumulator = non_maximum_supression(accumulator)
    idx = np.argpartition(accumulator, accumulator.size - 4, axis=None)[-4:]
    result = np.column_stack(np.unravel_index(idx, accumulator.shape))
    print(result)
    for r in range(0, result.shape[0]):
        rho = result[r][0]
        theta = result[r][1]
        # Set y = 0 and solve for x
        theta = np.deg2rad(theta)
        x0 = round(rho * (1 / np.sin(theta)))

        # Set y = thing and solve for x
        x1 = round(rho * (1 / np.sin(theta)) - input_image_col / np.tan(theta))
        pt1 = (0, x0)
        pt2 = (input_image_col, x1)
        cv2.line(original_image_3_channels, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

        #print("First Point: " + str(pt1) + " Second Point" + str(pt2))

    return original_image_3_channels.astype(np.uint8)

def ransac(hessian_image: np.ndarray, distance_threshold, inlier_threshold, original_image_3_channels):
    input_image_rows, input_image_cols = hessian_image.shape

    # These points are in (x,y) orientation to make line math easier for me :)
    list_of_points = []
    for r in range(0,input_image_rows):
        for c in range(0,input_image_cols):
            if hessian_image[r][c] != 0:
                list_of_points.append((c,r))
    num_lines = 0
    while num_lines < 4:
        pt1 = random.choice(list_of_points)
        list_of_points.remove(pt1)
        pt2 = random.choice(list_of_points)
        list_of_points.remove(pt2)

        if pt2[0]-pt1[0] == 0:
            continue

        slope = (pt2[1]-pt1[1]) / (pt2[0]-pt1[0])
        intercept = pt2[1] - slope*pt2[0]
        #print("y = " + str(slope) + "x + " + str(intercept))
        A = slope
        B = -1
        C = intercept

        inliers = []

        for point in list_of_points:

            distance = abs(A*point[0] + B*point[1] + C)/ np.sqrt(A**2 + 1)

            if distance <= distance_threshold:
                #print("Point: " + str(point) + " is an inlier with a distance of: " + str(distance))
                inliers.append(point)

        if len(inliers) >= inlier_threshold:
            num_lines += 1
            pt1_y = intercept
            pt2_y = input_image_cols*slope + intercept
            p1 = (0,int(pt1_y))
            p2 = (int(input_image_cols), int(pt2_y))
            cv2.line(original_image_3_channels, p1,p2, (0, 0, 255), 3, cv2.LINE_AA)
            #cv2.line(original_image_3_channels, pt1,pt2, (0, 255, 0), 3, cv2.LINE_AA)

            for point in inliers:
                cv2.rectangle(original_image_3_channels,(point[0]-1,point[1]-1),(point[0]+1,point[1]+1),(255,0,0),1)
                list_of_points.remove(point)
        else:
            list_of_points.append(pt1)
            list_of_points.append(pt2)

    return original_image_3_channels

if __name__ == "__main__":
    sigma = 1
    hessian_threshold = 75000
    ransac_distance_threshold = 1.95
    ransac_inlier_threshold = 8
    image_location = "/home/bmirisola/PycharmProjects/558/hw2/road.png"
    original_image = cv2.imread(image_location, cv2.IMREAD_GRAYSCALE)
    line_image = cv2.imread(image_location)

    kernel = generate_gaussian_kernel(sigma)

    blurred_image = guassian_filter(original_image, sigma)
    hessian_image = hessian_detector(blurred_image, hessian_threshold)
    #lines_detected_hough = hough_transform(hessian_image, line_image)
    ransac_image = ransac(hessian_image,ransac_distance_threshold, ransac_inlier_threshold, line_image)

    cv2.imshow("Original", original_image)
    cv2.imshow('Hessian', hessian_image)
    #cv2.imshow('Hough', lines_detected_hough)
    cv2.imshow("RANSAC", ransac_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
