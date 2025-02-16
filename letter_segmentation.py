from word_segmentation import *

def letter_images(image, letter_list):
    image_list = []
    for i in range(len(letter_list)):
        word = letter_list[i]
        roi = image[word[1]:word[3], word[0]:word[2]]
        
        image_list.append(roi)
    return image_list


def generate_letter_segmentation(word_image):
    image = word_image.copy()
    thresh_img = thresholding(image)
    dilated = dilation(thresh_img, (25, 2))
    sorted_contour_letters = contours_by_x(dilated)

    letter_list = []
    image2 = image.copy()

    for letter in sorted_contour_letters:
        x2, y2, w2, h2 = cv2.boundingRect(letter)
        letter_list.append([x2, y2, x2+w2, y2+h2])
        cv2.rectangle(image2, (x2, y2), (x2+w2, y2+h2), (255,0,0), 2)

    plt.imshow(dilated)
    plt.show()
    plt.imshow(image2)
    plt.show()
    letter_images_list = letter_images(image, letter_list)
    return letter_images_list

    
