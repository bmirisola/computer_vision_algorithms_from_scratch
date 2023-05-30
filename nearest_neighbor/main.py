import cv2
import numpy as np
import math
import copy

def calculate_color_distance(bgr1, bgr2, pt1, pt2):
    dist = math.sqrt((float(bgr1[0]) - float(bgr2[0])) ** 2 + (float(bgr1[1]) - float(bgr2[1])) ** 2 + (float(bgr1[2]) - float(bgr2[2])) ** 2 + (float(pt1[0]) - float(pt2[0])) ** 2 + (float(pt1[1]) - float(pt2[1])) ** 2)
    return dist

# def calculate_color_distance(bgr1, bgr2, pt1, pt2):
#     dist = math.sqrt((int(bgr1[0]) - int(bgr2[0])) ** 2 + (int(bgr1[1]) - int(bgr2[1])) ** 2 + (int(bgr1[2]) - int(bgr2[2])) ** 2 + (int(pt1[0]) - int(pt2[0])) ** 2 + (int(pt1[1]) - int(pt2[1])) ** 2)
#     #print("The distance bwteen pt1: " + str(pt1) + " and point 2: " + str(pt2) + " is " + str(dist))
#     return dist

def k_means_segmentation(image: np.ndarray):
    # K is the number of clusters you want to identify
    k = 10
    threshold = 3
    yx_centroids = []
    rows, columns, _ = image.shape
    for i in range(k):
        yx_centroids.append([np.random.randint(0, rows), np.random.randint(0, columns)])

    bgr_centroids = []
    for i in yx_centroids:
        r,c = i
        a_list = list(image[r][c])
        bgr_centroids.append(a_list)

    print(bgr_centroids)

    bgr_cluster_rows = []
    yx_cluster_rows = []

    for i in range(k):
        bgr_cluster_rows.append([])
        yx_cluster_rows.append([])

    converged = False

    iterations = 0
    while not converged:
        for cluster in bgr_cluster_rows:
            index = bgr_cluster_rows.index(cluster)
            bgr_cluster_rows[index] = []
            yx_cluster_rows[index] = []

        for r in range(0, rows):
            for c in range(0, columns):
                closest_rgb_centroid = []
                best_coordinate = []
                smallest_distance = 1000000
                coords = [r,c]
                p1 = list(image[r][c])
                for p in bgr_centroids:
                    p2 = p
                    index = bgr_centroids.index(p)
                    distance = calculate_color_distance(p1, p2, coords, yx_centroids[index])
                    if distance < smallest_distance:
                        smallest_distance = distance
                        closest_rgb_centroid = p2
                        best_coordinate = [r,c]

                #print("The closest centroid to point " + str([r,c]) + " is " + str(closest_rgb_centroid) + " with a bgr value of " + str(p1))
                #print(centroids.index(closest_rgb_centroid))
                bgr_cluster_rows[bgr_centroids.index(closest_rgb_centroid)].append(p1)
                yx_cluster_rows[bgr_centroids.index(closest_rgb_centroid)].append(best_coordinate)
        print("I made it out")
        old_bgr_centroids = copy.deepcopy(bgr_centroids)
        old_yx_centroids = copy.deepcopy(yx_centroids)
        bgr_centroids = []
        yx_centroids = []
        for cluster, yx_cluster in zip(bgr_cluster_rows,yx_cluster_rows):
            r_sum = 0
            g_sum = 0
            b_sum = 0
            length = len(cluster)
            y_sum = 0
            x_sum = 0
            # index = bgr_cluster_rows.index(cluster)
            # print("The length of the " + str(index) + "th bgr cluster is " + str(length))
            #print("Index is " + str(index))
            for bgr_value,yx_value in zip(cluster,yx_cluster):
                r_sum+= bgr_value[2]
                g_sum+= bgr_value[1]
                b_sum+= bgr_value[0]
                #index_2 = cluster.index(bgr_value)
                #print("Index_2 is " + str(index_2))
                y_sum += yx_value[0]
                x_sum += yx_value[1]

            r = r_sum/length
            g = g_sum/length
            b = b_sum/length
            y = y_sum/length
            x = x_sum/length
            bgr_centroids.append([b,g,r])
            yx_centroids.append([y,x])

            #centroids[cluster_rows.index(cluster)] = new_centroid
        # for cluster in yx_cluster_rows:
        #     x_sum = 0
        #     y_sum = 0
        #     length = len(cluster)
        #     # index = yx_cluster_rows.index(cluster)
        #     # print("The length of the " + str(index) + "th yx cluster is " + str(length))
        #     for yx_coord in cluster:
        #         y_sum += yx_coord[0]
        #         x_sum += yx_coord[1]
        #
        #     y = y_sum / length
        #     x = x_sum/length
        #     yx_centroids.append([y, x])

        for cluster in bgr_cluster_rows:
            index = bgr_cluster_rows.index(cluster)
            distance = calculate_color_distance(bgr_centroids[index], old_bgr_centroids[index], yx_centroids[index], old_yx_centroids[index])
            if distance < threshold:
                converged = True
            else:
                converged = False
        iterations+=1
        print("The number of iterations that has happened is " + str(iterations))

    image_copy = np.copy(image)
    for cluster in yx_cluster_rows:
        index = yx_cluster_rows.index(cluster)
        for yx in cluster:
            r,c = yx
            image_copy[r][c] = bgr_centroids[index]

    return image_copy

def problem3_k_means_segmentation(image: np.ndarray):
    # K is the number of clusters you want to identify
    k = 10
    threshold = 7
    yx_centroids = []
    rows, columns, _ = image.shape
    # for i in range(k):
    #     yx_centroids.append([np.random.randint(0, rows), np.random.randint(0, columns)])

    bgr_centroids = []
    while len(yx_centroids) < k:
        coords = [np.random.randint(0, rows), np.random.randint(0, columns)]
        r, c = coords
        if list(image[r][c]) not in bgr_centroids:
            yx_centroids.append([r,c])
            bgr_centroids.append(list(image[r][c]))

    print(bgr_centroids)
    print(yx_centroids)

    bgr_cluster_rows = []
    yx_cluster_rows = []

    for i in range(k):
        bgr_cluster_rows.append([])
        yx_cluster_rows.append([])

    converged = False

    iterations = 0
    while not converged:
        for cluster in bgr_cluster_rows:
            index = bgr_cluster_rows.index(cluster)
            bgr_cluster_rows[index] = []
            yx_cluster_rows[index] = []

        for r in range(0, rows):
            for c in range(0, columns):
                closest_rgb_centroid = []
                best_coordinate = []
                smallest_distance = 1000000
                coords = [r,c]
                p1 = list(image[r][c])
                for p in bgr_centroids:
                    p2 = p
                    index = bgr_centroids.index(p)
                    distance = calculate_color_distance(p1, p2, coords, yx_centroids[index])
                    #print(distance)
                    if distance < smallest_distance:
                        smallest_distance = distance
                        closest_rgb_centroid = p2
                        best_coordinate = [r,c]
                        #print('updating distance')

                #print("The closest centroid to point " + str([r,c]) + " is " + str(closest_rgb_centroid) + " with a bgr value of " + str(p1))
                #print(centroids.index(closest_rgb_centroid))
                bgr_cluster_rows[bgr_centroids.index(closest_rgb_centroid)].append(p1)
                yx_cluster_rows[bgr_centroids.index(closest_rgb_centroid)].append(best_coordinate)
        print("I made it out")
        #print(len(bgr_cluster_rows))
        old_bgr_centroids = copy.deepcopy(bgr_centroids)
        old_yx_centroids = copy.deepcopy(yx_centroids)
        bgr_centroids = []
        yx_centroids = []

        for cluster in bgr_cluster_rows:
            print("The length of the cluster is: " + str(len(cluster)))

        for cluster, yx_cluster in zip(bgr_cluster_rows,yx_cluster_rows):
            r_sum = 0
            g_sum = 0
            b_sum = 0
            length = len(cluster)
            y_sum = 0
            x_sum = 0
            # index = bgr_cluster_rows.index(cluster)
            # print("The length of the " + str(index) + "th bgr cluster is " + str(length))
            #print("Index is " + str(index))
            for bgr_value,yx_value in zip(cluster,yx_cluster):
                r_sum+= bgr_value[2]
                g_sum+= bgr_value[1]
                b_sum+= bgr_value[0]
                #index_2 = cluster.index(bgr_value)
                #print("Index_2 is " + str(index_2))
                y_sum += yx_value[0]
                x_sum += yx_value[1]

            r = r_sum/length
            g = g_sum/length
            b = b_sum/length
            y = y_sum/length
            x = x_sum/length
            bgr_centroids.append([b,g,r])
            yx_centroids.append([y,x])

        for cluster in bgr_cluster_rows:
            index = bgr_cluster_rows.index(cluster)
            distance = calculate_color_distance(bgr_centroids[index], old_bgr_centroids[index], yx_centroids[index], old_yx_centroids[index])
            if distance < threshold:
                converged = True
            else:
                converged = False
        iterations+=1
        print("The number of iterations that has happened is " + str(iterations))

    return bgr_centroids,yx_centroids


def pixel_classification(sky_image: np.ndarray, no_sky: np.ndarray, images_list: list, no_sky_bgr_centroids, no_sky_yx_centroids):
    white = [255, 255, 255]
    rows, columns, _ = sky_image.shape
    mask_image = np.copy(sky_image)
    for r in range(0,rows):
        for c in range (0, columns):
            if list(no_sky[r][c]) == white:
                mask_image[r][c] = sky_image[r][c]
            else:
                mask_image[r][c] = white

    print('now doing mask_image k-means')

    mask_image_bgr_centroids, mask_image_yx_centroids = problem3_k_means_segmentation(mask_image)
    no_sky_bgr_centroids_white_removed, no_sky_yx_centroids_white_removed = [],[]
    mask_image_bgr_centroids_white_removed, mask_image_yx_centroids_white_removed = [],[]
    print('I finished???')

    print(no_sky_bgr_centroids)

    for i in range(0,10):
        if no_sky_bgr_centroids[i] != white:
            no_sky_bgr_centroids_white_removed.append(no_sky_bgr_centroids[i])
            no_sky_yx_centroids_white_removed.append(no_sky_yx_centroids[i])
        if mask_image_bgr_centroids[i] != white:
            mask_image_bgr_centroids_white_removed.append(mask_image_bgr_centroids[i])
            mask_image_yx_centroids_white_removed.append(mask_image_yx_centroids[i])

    print(no_sky_bgr_centroids_white_removed)
    print(no_sky_yx_centroids_white_removed)
    print(mask_image_bgr_centroids_white_removed)
    print(mask_image_yx_centroids_white_removed)


    orange = [0, 165, 255]
    classified_images = []
    for image in images_list:
        image_copy = np.copy(image)
        for r in range(0,rows):
            for c in range(0,columns):
                pixel = list(image[r][c])
                coords_image = [r,c]
                no_sky_distance = 10000000
                mask_image_distance = 10000000
                for i in range(0,len(no_sky_bgr_centroids_white_removed)):
                    pixel2 = no_sky_bgr_centroids_white_removed[i]
                    pt2 = no_sky_yx_centroids_white_removed[i]
                    distance = calculate_color_distance(pixel,pixel2,coords_image,pt2)
                    if distance < no_sky_distance:
                        no_sky_distance = distance

                for i in range(0, len(mask_image_bgr_centroids_white_removed)):
                    pixel2 = mask_image_bgr_centroids_white_removed[i]
                    pt2 = mask_image_yx_centroids_white_removed[i]
                    distance = calculate_color_distance(pixel,pixel2,coords_image,pt2)
                    if distance < mask_image_distance:
                        mask_image_distance = distance

                if mask_image_distance < no_sky_distance:
                    image_copy[r][c] = orange
        classified_images.append(image_copy)

    return classified_images


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    image = cv2.imread("/home/bmirisola/PycharmProjects/558/hw4:)/data/white-tower.png")

    segmented_image = k_means_segmentation(image)

    print("It finished")

    no_sky_image = cv2.imread('/home/bmirisola/PycharmProjects/558/hw4:)/data/sky/no_sky.jpg')
    simage = cv2.imread('/home/bmirisola/PycharmProjects/558/hw4:)/data/sky/sky_train.jpg')
    nsi2 = cv2.imread('/home/bmirisola/PycharmProjects/558/hw4:)/data/sky/no_sky.jpg')



    sky_test_1 = cv2.imread('/home/bmirisola/PycharmProjects/558/hw4:)/data/sky/sky_test1.jpg')
    sky_test_2 = cv2.imread('/home/bmirisola/PycharmProjects/558/hw4:)/data/sky/sky_test2.jpg')
    sky_test_3 = cv2.imread('/home/bmirisola/PycharmProjects/558/hw4:)/data/sky/sky_test3.jpg')
    sky_test_4 = cv2.imread('/home/bmirisola/PycharmProjects/558/hw4:)/data/sky/sky_test4.jpg')

    test_images_list = [sky_test_1, sky_test_2, sky_test_3, sky_test_4]
    #test_images_list = [sky_test_1]
    no_sky_bgr_centroids1, no_sky_yx_centroids1 = problem3_k_means_segmentation(nsi2)

    classified_images = pixel_classification(simage, no_sky_image, test_images_list,no_sky_bgr_centroids1,no_sky_yx_centroids1)


    cv2.imshow("image",image)
    cv2.imshow("segmented", segmented_image)
    cv2.imshow('sky_test1', classified_images[0])
    cv2.imshow('sky_test2', classified_images[1])
    cv2.imshow('sky_test3', classified_images[2])
    cv2.imshow('sky_test4', classified_images[3])

    cv2.waitKey(0)
    cv2.destroyAllWindows()