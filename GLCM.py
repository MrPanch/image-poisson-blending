import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from tqdm import tqdm

from skimage.feature import greycomatrix, greycoprops
from sklearn.linear_model import LogisticRegression


def prepare_data(sq_lin=100):

    for folder in ['train_negative', 'train_positive', 'validation']:
        root_folder = os.path.join('GLCM_images', f'{folder}_class')
        print(f'\n cropping samples for {folder}')
        for image_name in tqdm(os.listdir(root_folder)):
            image = cv2.imread(os.path.join(root_folder, image_name))
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
            if folder == 'train_negative':
                sum_list.sort()
            if folder == 'train_positive':
                sum_list.sort(reverse=True)
            result = []
            for sum_val in sum_list[:8]:
                result.append(sum_dic[sum_val])

            idx = 0
            for coord in result:
                image_slice = image[coord[0]:coord[0] + sq_lin, coord[1]:coord[1] + sq_lin]
                cv2.imwrite(f'GLCM_images/{folder}_cropped/{image_name}_{idx}.png', image_slice)
                idx += 1


def glcm_features():
    result_negative_list = []
    result_positive_list = []
    result_validation_list = []
    for folder in ['train_negative_cropped', 'train_positive_cropped', 'validation_cropped']:
        correlation = []
        dissimilarity = []
        bottom_patches = []
        root_folder = os.path.join(f'GLCM_images/', folder)
        for image_name in os.listdir(root_folder):
            image = cv2.imread(os.path.join(root_folder, image_name))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            bottom_patches.append(image)
        for patch in (bottom_patches):
            glcm = greycomatrix(patch, distances=[5], angles=[0], levels=256,
                                symmetric=True, normed=True)
            correlation.append(greycoprops(glcm, 'correlation')[0, 0])
            dissimilarity.append(greycoprops(glcm, 'dissimilarity')[0, 0])

        if folder == 'train_negative_cropped':
            result_negative_zip = zip(correlation, dissimilarity)
            result_negative_list = list(result_negative_zip)
        if folder == 'train_positive_cropped':
            result_positive_zip = zip(correlation, dissimilarity)
            result_positive_list = list(result_positive_zip)
        if folder == 'validation_cropped':
            result_validation_zip = zip(correlation, dissimilarity)
            result_validation_list = list(result_validation_zip)

    result = {'negative': np.array(result_negative_list), 'positive': np.array(result_positive_list),
              'validation': np.array(result_validation_list)}
    return result


def log_regressor(glcm_features):
    y_positive = np.ones(glcm_features['positive'].shape[0])
    y_negative = np.zeros(glcm_features['negative'].shape[0])
    X = np.append(glcm_features['positive'], glcm_features['negative'], axis=0)
    y = np.append(y_positive, y_negative)
    clf = LogisticRegression(random_state=0).fit(X, y)
    return clf


def clf_validation(clf, glcm_features):
    X = glcm_features['validation']
    prediction = clf.predict(X)
    root_folder = os.path.join(f'GLCM_images', f'validation_cropped')
    data_2_display = list(zip(os.listdir(root_folder), prediction))
    for idx in range(len(data_2_display)//8):
        fig = plt.figure(figsize=(9, 13))
        columns = 4
        rows = 2
        ax = []

        for i in range(columns * rows):
            img = cv2.imread(os.path.join(root_folder, data_2_display[idx*8 + i][0]))
            # create subplot and append to ax
            ax.append(fig.add_subplot(rows, columns, i + 1))
            ax[-1].set_title("image class: " + str(data_2_display[idx*8 + i][1]))  # set title
            plt.imshow(img)

        plt.show()  # finally, render the plot




if __name__ == '__main__':
    prepare_data()
    features = glcm_features()
    clf = log_regressor(features)
    clf_validation(clf, features)