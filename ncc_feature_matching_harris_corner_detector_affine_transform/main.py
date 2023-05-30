import math
import random
import cv2
import numpy as np
import old

def harris_corner_detector(image: np.ndarray, sigma, alpha, original_image: np.ndarray):
    rows, columns = image.shape

    gx, gy = old.sobel_filter(image)

    gx_padded = np.pad(gx, (3 * sigma, 3 * sigma), 'edge')
    gy_padded = np.pad(gy, (3 * sigma, 3 * sigma), 'edge')

    all_corners_image = np.zeros((rows,columns))

    '''
    Each Pixel will be a 2x2 array
    The first element is the sum of the squares of the x graident in a window the size of the gaussian kernel
    The fourth element is the sum of the elements in the The first element is the sum of the squares of the x graident in a window the size of the gaussian kernel
    The diagonals are the sum of the multiplications of x and y
     
    If trace > 1 it is a corner
    '''

    gx_squared = np.square(gx_padded)
    gy_squared = np.square(gy_padded)
    gxy = np.multiply(gy_padded,gx_padded)

    output_img = cv2.cvtColor(original_image.copy(), cv2.COLOR_GRAY2RGB)

    for r in range(3 * sigma, rows - 3 * sigma):
        for c in range(3 * sigma, columns - 3 * sigma):
            Ixx = np.sum(gx_squared[r-3 * sigma:r+3*sigma+1,c - 3 * sigma:c + 3 * sigma + 1])
            Iyy = np.sum(gy_squared[r - 3 * sigma:r + 3 * sigma + 1, c - 3 * sigma:c + 3 * sigma + 1])
            Ixy = np.sum(gxy[r - 3 * sigma:r + 3 * sigma + 1, c - 3 * sigma:c + 3 * sigma + 1])

            determinant = (Ixx * Iyy) - Ixy**2
            trace = Ixx + Iyy
            #R is threshold that must be overcome

            R = determinant - alpha*(trace**2)
            if R > 1:
                all_corners_image[r - 3 * sigma,c - 3 * sigma] = R

    highest_corner_values_image = np.zeros((rows, columns))

    for x in range(0,1000):
        highest_index = np.unravel_index(np.argmax(all_corners_image),all_corners_image.shape)
        highest_corner_values_image[highest_index] = all_corners_image[highest_index]
        all_corners_image[highest_index] = 0

    nms_image = old.non_maximum_supression(highest_corner_values_image)

    feature_list = []
    for r in range(0,rows):
        for c in range(0,columns):
            if nms_image[r][c] != 0:
                cv2.circle(output_img, (c,r), 3, (0,0,255),3)
                feature_list.append([r,c])

    #feature_list is the coordinates of the features
    return nms_image,output_img, feature_list

def feature_matching(left_image: np.ndarray, right_image: np.ndarray, left_features, right_features, N, kernel_size):
    combined_images = np.concatenate((left_image, right_image), axis=1)
    rows, columns = left_image.shape

    correlated_left_feature_coordinates = []
    correlated_right_feature_coordinates = []
    correlation_scores = np.empty(0, dtype=float)
    for x in left_features:
        g_window = left_image[x[0] - int(kernel_size/2):x[0] + int(kernel_size/2) + 1, x[1] - int(kernel_size/2):x[1] + int(kernel_size/2) + 1]
        best_correlation = 0
        best_correlation_left_image_coordinate = []
        best_correlation_right_image_coordinate = []

        g_window_mean = np.mean(g_window)
        g_window = np.subtract(g_window,g_window_mean)

        for y in right_features:
            f_window = right_image[y[0] - int(kernel_size/2):y[0] + int(kernel_size/2) + 1, y[1] - int(kernel_size/2):y[1] + int(kernel_size/2) + 1]
            f_window_mean = np.mean(f_window)
            f_window = np.subtract(f_window, f_window_mean)
            ncc_numerator = np.sum(g_window*f_window)

            ncc_denominator = math.sqrt(np.multiply(np.sum(np.square(g_window)), np.sum(np.square(f_window))))

            correlation_score = ncc_numerator / ncc_denominator

            if correlation_score > best_correlation:
                best_correlation = correlation_score
                best_correlation_left_image_coordinate = [x[1],x[0]]
                best_correlation_right_image_coordinate = [y[1],y[0]]

        correlated_left_feature_coordinates.append(best_correlation_left_image_coordinate)
        correlated_right_feature_coordinates.append(best_correlation_right_image_coordinate)
        correlation_scores = np.append(correlation_scores, best_correlation)
        #print("The correlation between points " + str(best_correlation_left_image_coordinate) + " and "  + str(best_correlation_right_image_coordinate) + " is " + str(best_correlation))

    combined_images = cv2.cvtColor(combined_images.copy(), cv2.COLOR_GRAY2RGB)

    correlation_scores,correlated_left_feature_coordinates, \
    correlated_right_feature_coordinates, = \
    (list(t) for t in zip(*sorted(zip(correlation_scores,correlated_left_feature_coordinates, \
    correlated_right_feature_coordinates))))

    highest_correlations = []
    index = len(correlation_scores) - 1

    highest_correlations_left = []
    highest_correlations_right = []

    while len(highest_correlations) < N and index >=0:
        left_feature = correlated_left_feature_coordinates[index]
        right_feature = correlated_right_feature_coordinates[index]

        if not [a for a in highest_correlations if left_feature in a or right_feature in a]:
            highest_correlations += ([(left_feature, right_feature)])
            pt1 = (left_feature[0], left_feature[1])
            pt2 = (right_feature[0] + columns, right_feature[1])
            highest_correlations_left.append([left_feature[0], left_feature[1]])
            highest_correlations_right.append([right_feature[0], right_feature[1]])
            cv2.line(combined_images, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)

        index-=1

    return combined_images, highest_correlations_left, highest_correlations_right

def ransac(left_features_original: list, right_features_original: list, error_threshold, inlier_threshold):
    N = float('inf')
    left_features = left_features_original.copy()
    right_features = right_features_original.copy()
    lowest_error = 0
    ransac_iterations = 0
    highest_num_of_inliers = 0
    s = 3
    best_matrix = []

    while N > ransac_iterations:
        error = 0
        pt1 = random.choice(left_features)
        index = left_features.index(pt1)
        pt1_prime = right_features[index]
        left_features.remove(pt1)
        right_features.remove(pt1_prime)

        pt2 = random.choice(left_features)
        index = left_features.index(pt2)
        pt2_prime = right_features[index]
        left_features.remove(pt2)
        right_features.remove(pt2_prime)

        pt3 = random.choice(left_features)
        index = left_features.index(pt3)
        pt3_prime = right_features[index]
        left_features.remove(pt3)
        right_features.remove(pt3_prime)

        points = [pt1,pt2,pt3]

        X = np.ndarray((6, 6), np.uint64)

        for points_index in range(0, len(points)):
            X[2 * points_index] = [points[points_index][0], points[points_index][1], 1, 0, 0, 0]
            X[2 * points_index + 1] = [0, 0, 0, points[points_index][0], points[points_index][1], 1]

        primes = np.array([pt1_prime,pt2_prime,pt3_prime]).reshape(6,1)

        if np.linalg.det(X) == 0:
            continue
        affine_transformation_matrix = np.linalg.solve(X,primes).reshape(2,3)

        inliers = []

        for original_point,prime_point in zip(left_features, right_features):
            og = [original_point[0], original_point[1], 1]
            projected_point = np.dot(affine_transformation_matrix, og)
            distance = math.sqrt(pow(projected_point[0] - prime_point[0],2)+pow(projected_point[1] - prime_point[1],2))
            #print("Projected Point: " + str(projected_point) + " has a distance of: " + str(distance) + " from prime point: " + str(prime_point))

            if distance <= error_threshold:
                #print("Projected Point: " + str(projected_point) + " is an inlier with a distance of: " + str(distance))
                inliers.append(projected_point)
                error += distance

        if len(inliers) >= inlier_threshold:
            error/= len(inliers)
            return affine_transformation_matrix, error,ransac_iterations

        else:
            if len(inliers) > highest_num_of_inliers:
                best_matrix = affine_transformation_matrix
                highest_num_of_inliers = len(inliers)
                lowest_error = error/len(inliers)

            left_features.append(pt1)
            left_features.append(pt2)
            left_features.append(pt3)

            right_features.append(pt1_prime)
            right_features.append(pt2_prime)
            right_features.append(pt3_prime)

            if not len(inliers) == 0:
                e = 1 - (len(inliers) / len(left_features))
                #print("len of inliers = " + str(len(inliers)))
                #print("e is " + str(e))
                N = math.log(e)/math.log(1-(1-e)**s)
                #print("N is " + str(N))
            ransac_iterations += 1

    return best_matrix,lowest_error,ransac_iterations

if __name__ == '__main__':
    left_image_source = "/home/bmirisola/PycharmProjects/558/hw3/uttower_left.jpg"
    right_image_source = "/home/bmirisola/PycharmProjects/558/hw3/uttower_right.jpg"
    left_image = cv2.imread(left_image_source, cv2.IMREAD_GRAYSCALE)
    right_image = cv2.imread(right_image_source, cv2.IMREAD_GRAYSCALE)

    sigma = 2
    alpha = .06

    left_image_blurred = old.guassian_filter(left_image, sigma)
    right_image_blurred = old.guassian_filter(right_image, sigma)

    left_corner_features, left_harris, left_feature_list = harris_corner_detector(left_image_blurred, sigma,alpha, left_image)
    right_corner_features, right_harris, right_feature_list = harris_corner_detector(right_image_blurred,sigma, alpha, right_image)

    N = 20
    kernel_size = 11

    both, highest_correlated_features_left, highest_correlated_features_right = feature_matching(left_image, right_image, left_feature_list, right_feature_list, N, kernel_size)

    inlier_threshold = 5
    distance_threshold = 25

    print(highest_correlated_features_left)
    print(highest_correlated_features_right)

    a1_affine, a1_error, a1_iterations = ransac(highest_correlated_features_left, highest_correlated_features_right, distance_threshold, inlier_threshold)
    print("The a_1 affine is \n" + str(a1_affine))
    print("The average a_1 error is : " + str(a1_error))
    print("Iterations is equal to: " + str(a1_iterations))

    a1_affine[0][2] = np.negative(a1_affine[0][2]) + 200

    warp = cv2.warpAffine(left_image, a1_affine, (left_image.shape[1] * 2, left_image.shape[0]))
    warp[0:right_image.shape[0], right_image.shape[1]:right_image.shape[1] * 2] = right_image

    a2_left = highest_correlated_features_left.copy()
    a2_right = highest_correlated_features_right.copy()

    for i in range(0, 30):
        left_row_coordinate = random.randint(0, left_image.shape[0]-1)
        left_col_coordinate = random.randint(0, left_image.shape[1]-1)
        left_pt = [left_row_coordinate, left_col_coordinate]

        right_row_coordinate = random.randint(0, left_image.shape[0] - 1)
        right_col_coordinate = random.randint(0, left_image.shape[1] - 1)

        right_pt = [right_row_coordinate, right_col_coordinate]

        a2_left.append(left_pt)
        a2_right.append(right_pt)

    a2_affine, a2_error, a2_iterations = ransac(a2_left, a2_right, distance_threshold, inlier_threshold)
    print("The a_2 affine is \n" + str(a2_affine))
    print("The average a_2 error is : " + str(a2_error))
    print("a2 Iterations is equal to: " + str(a2_iterations))

    cv2.imshow("left_harris", left_harris)
    cv2.imshow("right_harris", right_harris)
    cv2.imshow("both", both)
    cv2.imshow("warp", warp)

    cv2.waitKey(0)
    cv2.destroyAllWindows()