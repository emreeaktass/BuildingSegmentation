import numpy as np
import copy
import matplotlib.pyplot as plt
import random
import cv2
import file_operations as f


def extract(color_data, ndsm_data, building_mask, patch_size):
    object_thresh = 100
    row, col, channel = color_data.shape

    c1 = 0
    c2 = 0
    counter = 0
    path = 'all_datas'
    for i in range(0, row, patch_size):

        for j in range(0, col, patch_size):
            if i + j > 15000:
                sum_of_pixels_in_patch = np.sum(building_mask[i: i + patch_size, j: j + patch_size])
                if sum_of_pixels_in_patch > object_thresh:
                    # write to file

                    # model 1
                    ndvi = calculate_ndvi(color_data[i: i + patch_size, j: j + patch_size,:])
                    gray = cv2.cvtColor(color_data[i: i + patch_size, j: j + patch_size,:], cv2.COLOR_RGB2GRAY)
                    ndsm = ndsm_data[i: i + patch_size, j: j + patch_size]

                    # model 2
                    # ndvi = color_data[i: i + patch_size, j: j + patch_size, 0]
                    # gray = color_data[i: i + patch_size, j: j + patch_size, 1]
                    # ndsm = color_data[i: i + patch_size, j: j + patch_size, 2]

                    # model 3
                    # ndvi = color_data[i: i + patch_size, j: j + patch_size, 0]
                    # gray = color_data[i: i + patch_size, j: j + patch_size, 1]
                    # ndsm = ndsm_data[i: i + patch_size, j: j + patch_size]

                    # model 4
                    # ndvi = color_data[i: i + patch_size, j: j + patch_size, 0]
                    # gray = color_data[i: i + patch_size, j: j + patch_size, 1]
                    # ndsm = calculate_ndvi(color_data[i: i + patch_size, j: j + patch_size,:])


                    mask = building_mask[i: i + patch_size, j: j + patch_size]
                    mask = mask.astype(np.uint8)
                    patch = np.empty((patch_size, patch_size, 3))
                    patch[:, :, 0] = gray
                    patch[:, :, 1] = ndsm
                    patch[:, :, 2] = ndvi
                    patch = patch.astype(np.uint8)
                    f.write(path, counter, 'data', patch)
                    f.write(path, counter, 'mask', mask)
                    counter = counter + 1
                    # print(counter)
                    # kayıt işlemi
                    # print(sum_of_pixels_in_patch)
                    # print(i)
                    # print(j)
                    # plt.figure(1)
                    # plt.imshow(gray, cmap='gray')
                    #
                    # plt.figure(2)
                    # plt.imshow(ndvi, cmap='gray')
                    #
                    # plt.figure(3)
                    # plt.imshow(ndsm, cmap='gray')
                    #
                    # plt.figure(4)
                    # plt.imshow(mask, cmap='gray')
                    #
                    # plt.show()
                    # break

                # if sum_of_pixels_in_patch == 0:
                #     add_fp_prob = random.random()
                #
                #     if add_fp_prob < 0.04:
                #         # add negative samples
                #         ndvi = calculate_ndvi(color_data[i: i + patch_size, j: j + patch_size, :])
                #         gray = cv2.cvtColor(color_data[i: i + patch_size, j: j + patch_size, :], cv2.COLOR_RGB2GRAY)
                #         ndsm = ndsm_data[i: i + patch_size, j: j + patch_size]
                #         mask = np.zeros((patch_size, patch_size))
                #         mask = mask.astype(np.uint8)
                #         patch = np.empty((patch_size, patch_size, 3))
                #
                #         patch[:, :, 0] = gray
                #         patch[:, :, 1] = ndsm
                #         patch[:, :, 2] = ndvi
                #         patch = patch.astype(np.uint8)
                #         f.write(path, counter, 'data', patch)
                #         f.write(path, counter, 'mask', mask)
                #         counter = counter + 1
                #         print(counter)
    # print('c1 : ', c1)
    # print('c2 : ', c2)

def calculate_ndvi(color_data):

    red = color_data[:,:,0].astype('f4')
    green = color_data[:,:,1].astype('f4')

    ndvi_nan = np.divide(np.subtract(green, red), np.add(green, red))
    ndvi = np.nan_to_num(ndvi_nan, nan=-1, posinf=-1, neginf=-1)
    ndvi = cv2.normalize(ndvi, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    ndvi = ndvi.astype(np.uint8)

    return ndvi