import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from tqdm import tqdm

from skimage.feature import greycomatrix, greycoprops
from sklearn.linear_model import LogisticRegression


def crop_samples(folder=f'GLCM_images/original/', sq_lin=100):
    for image_name in tqdm(os.listdir(folder)):
        image = cv2.imread(os.path.join(folder, image_name))
        sdim_1 = image.shape[0] % sq_lin
        sdim_2 = image.shape[1] % sq_lin

        sum_list = []
        coord_list = []
        for i in range(sdim_1, image.shape[0], sq_lin):
            for j in range(sdim_2, image.shape[1], sq_lin):
                image_slice = image[i:i + sq_lin, j:j + sq_lin]
                image_slice = image_slice.flatten()
                slice_sum = np.sum(image_slice)
                sum_list.append(slice_sum)
                coord_list.append([i, j])

        zip_iterator = zip(sum_list, coord_list)
        sum_dic = dict(zip_iterator)
        # помеять ревёрс, чтобы брать самые большие энтропии
        sum_list.sort(reverse=True)
        result = []
        for sum_val in sum_list[:8]:
            result.append(sum_dic[sum_val])

        # fig, maxs = plt.subplots(1, 1, figsize=(12, 8))
        # maxs.imshow(image[:, :, 0], cmap='gray')
        # maxs.axis('off')
        # for i in range(len(result)):
        #     rect2 = patches.Rectangle((result[i][0], result[i][1]),
        #                               100, 100,
        #                               linewidth=1, edgecolor='r', facecolor='none')
        #     maxs.add_patch(rect2)
        # plt.show()

        idx = 0
        for coord in result:
            image_slice = image[coord[0]:coord[0] + sq_lin, coord[1]:coord[1] + sq_lin]
            cv2.imwrite(f'GLCM_images/cropped/{image_name}_{idx}.png', image_slice)
            idx += 1


def glcm_features():
    correlation = []
    dissimilarity = []
    bottom_patches = []

    for image_name in os.listdir(f'GLCM_images/cropped/'):
        image = cv2.imread(os.path.join(f'GLCM_images/cropped/', image_name))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        bottom_patches.append(image)
    for patch in (bottom_patches):
        glcm = greycomatrix(patch, distances=[5], angles=[0], levels=256,
                            symmetric=True, normed=True)
        correlation.append(greycoprops(glcm, 'correlation')[0, 0])
        dissimilarity.append(greycoprops(glcm, 'dissimilarity')[0, 0])
        # create the figure

        # display original image with locations of patches
        # ax = fig.add_subplot(3, 2, 1)
        # ax.imshow(image, cmap=plt.cm.gray,
        #           vmin=0, vmax=255)
        # ax.set_xlabel('Original Image')
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.axis('image')

    # for each patch, plot (dissimilarity, correlation)
    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(1, 1, 1)
    # ax.plot(correlation[:len(bottom_patches)], dissimilarity[:len(bottom_patches)], 'go',
    #         label='Bottom')
    # ax.set_xlabel('GLCM Dissimilarity')
    # ax.set_ylabel('GLCM Correlation')
    # ax.legend()
    #
    # plt.tight_layout()
    # plt.show()
    result_zipping = zip(correlation, dissimilarity)
    result = list(result_zipping)
    return result


def log_regression(glcm_features):
    y = np.ones((glcm_features.shape[0],1))
    print(y.shape)
    print(glcm_features.shape)
    clf = LogisticRegression(random_state=0).fit(glcm_features, y)


if __name__ == '__main__':
    features = glcm_features()
    features = np.array(features)
    log_regression(features)