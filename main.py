from word_segmentation import *
from letter_segmentation import *
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from tamil_character_recognition import *
from class_unicode_mapper import *
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance

sentence = []
resized_sentence = []
word_segmentations_list = generate_word_images("D:/hari files/project/sf_project/sf_project/src/Datasets/handwritten_images/image11.png")
for word in word_segmentations_list:
    letter_segmentation_list = generate_letter_segmentation(word)
    sentence.append(letter_segmentation_list)

for word in sentence:
    resized_letter_list = []
    for letter in word:
        # Convert to PIL image
        img_pil = Image.fromarray(cv2.cvtColor(letter, cv2.COLOR_BGR2RGB))

        # Enhance contrast
        img_contrast = ImageEnhance.Contrast(img_pil)
        enhanced_img = img_contrast.enhance(2)

        # Convert back to numpy array
        enhanced_img_np = cv2.cvtColor(np.array(enhanced_img), cv2.COLOR_RGB2BGR)

        # Convert the image to grayscale
        gray_img = cv2.cvtColor(enhanced_img_np, cv2.COLOR_BGR2GRAY)

        # Find the top boundary
        top_boundary = 0
        for row in range(gray_img.shape[0]):
            if np.any(gray_img[row] < 10):  # Assuming black pixels have values close to 0
                top_boundary = row
                break

        # Find the bottom boundary
        bottom_boundary = gray_img.shape[0]
        for row in range(gray_img.shape[0]-1, -1, -1):
            if np.any(gray_img[row] < 10):  # Assuming black pixels have values close to 0
                bottom_boundary = row
                break

        # Crop the image
        cropped_img = enhanced_img_np[top_boundary:bottom_boundary, :]
        try:
            resized_img = cv2.resize(cropped_img, (64, 64), interpolation=cv2.INTER_NEAREST_EXACT)
        except:
            continue

        resized_letter_list.append(resized_img)
    resized_sentence.append(resized_letter_list)


string = ""
for word in resized_sentence:
    for letter in word:
        plt.imshow(letter)
        plt.show()
        predicted_class = letter_image_to_class(letter)
        string += get_tamil_character(predicted_class) + " "
    string += "  "

print(string)




