import cv2
import numpy as np
import os



def tesselation_brute_force(image, sq_lin_1, sq_lin_2):
    sdim_1 = image.shape[0] % sq_lin_1
    sdim_2 = image.shape[1] % sq_lin_2
    original = image.copy()

    result = []
    for i in range(sdim_1, image.shape[0], sq_lin_1):
        for j in range(sdim_2, image.shape[1], sq_lin_2):
            image_slice = image[i:i + sq_lin_1-5, j:j + sq_lin_2-5]
            if np.sum(image_slice) == 0:
                image[i:i + sq_lin_1, j:j + sq_lin_2] = 122
                result.append((i,j,sq_lin_1,sq_lin_2))
    # comparison = np.hstack((original, image))
    # cv2.imshow("image_name", comparison)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return result

def extract_contours(image, check_bb=False, check_rect=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blurred, 50, 150)

    cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)


    rects = []
    for c in cnts:
        area = cv2.contourArea(c)
        # TODO: change this constant
        if area > 100:
            (x, y, w, h) = cv2.boundingRect(c)
            rects.append((x, y, w, h))

    if check_bb:
        for coord in rects:
            image = cv2.rectangle(image, (coord[0], coord[1]), (coord[0]+coord[2], coord[1]+coord[3]), (255,0,0), 2)

        cv2.imshow("image_name", image)
        cv2.waitKey()
        cv2.destroyAllWindows()

    if check_rect:
        for coord in rects:
            image_slice = image[coord[1]:coord[1]+coord[3], coord[0]:coord[0]+coord[2]]
            cv2.imshow("image_name", image_slice)
            cv2.waitKey()
            cv2.destroyAllWindows()

    return rects


def find_places(image, img_size):
    rects = extract_contours(image)

    result = []
    for coord in rects:
        image_slice = image[coord[1]:coord[1] + coord[3], coord[0]:coord[0] + coord[2]]
        result.append(tesselation_brute_force(image_slice, img_size[0], img_size[1]))

    return result



if __name__ == '__main__':
    root_folder = f'seg_images/templates'
    for image_name in os.listdir(root_folder):
        image = cv2.imread(os.path.join(root_folder, image_name))
        print(find_places(image, (48, 36)))
