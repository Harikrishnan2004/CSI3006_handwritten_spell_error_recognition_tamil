import cv2
import numpy as np
import matplotlib.pyplot as plt

def resize(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, c = img.shape

    if w > 1000:
        new_w = 1000
        ar = w/h
        new_h = int(new_w/ar)
        
        img = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_AREA)
    return img

def thresholding(image):
    img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(img_gray,80,255,cv2.THRESH_BINARY_INV)
    plt.imshow(thresh)
    plt.show()
    return thresh

def dilation(threshold_image, kernel_size):
    kernel = np.ones(kernel_size, np.uint8)
    dilated = cv2.dilate(threshold_image, kernel, iterations = 1)
    return dilated

def contours_by_y(dilated_image):
    (contours, heirarchy) = cv2.findContours(dilated_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours = sorted(contours, key = lambda ctr : cv2.boundingRect(ctr)[1])
    return sorted_contours

def contours_by_x(dilated_image):
    (contours, heirarchy) = cv2.findContours(dilated_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours = sorted(contours, key = lambda ctr : cv2.boundingRect(ctr)[0])
    return sorted_contours

def word_segmentation(img, dilated_image, sorted_contours_lines):
    img3 = img.copy()
    words_list = []

    for line in sorted_contours_lines:

        x, y, w, h = cv2.boundingRect(line)
        roi_line = dilated_image[y:y+h, x:x+w]
        
        sorted_contour_words = contours_by_x(roi_line)
        
        for word in sorted_contour_words:
            if cv2.contourArea(word) < 500:
                continue
            
            x2, y2, w2, h2 = cv2.boundingRect(word)
            words_list.append([x+x2, y+y2, x+x2+w2, y+y2+h2])
            cv2.rectangle(img3, (x+x2, y+y2), (x+x2+w2, y+y2+h2), (255,0,0),2)
            
    plt.imshow(img3)
    return words_list

def word_images(img, words_list):
    image_list = []
    for i in range(len(words_list)):
        word = words_list[i]
        roi = img[word[1]:word[3], word[0]:word[2]]
        plt.figure()
        plt.imshow(roi)
        image_list.append(roi)
    plt.show()
    return image_list

def generate_word_images(img_path):
    # for line segmentation
    image = resize(img_path)
    thresh_img = thresholding(image)
    dilated = dilation(thresh_img, (20, 85))
    sorted_contour_lines = contours_by_y(dilated)

    plt.imshow(dilated)
    plt.show()

    #for word segmentation
    dilated2 = dilation(thresh_img, (25, 25))
    plt.imshow(dilated2)
    plt.show()
    words_list = word_segmentation(image, dilated2, sorted_contour_lines)
    word_images_list = word_images(image, words_list)
    return word_images_list

