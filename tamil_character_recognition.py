import tensorflow as tf
import tensorflow.keras.layers as tfl

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

BATCH_SIZE = 32
IMAGE_SIZE = (64, 64)
directory_train = "D:/hari files/project/sf_project/sf_project/src/Datasets/handwritten_letter_images/train"
directory_test = "D:/hari files/project/sf_project/sf_project/src/Datasets/handwritten_letter_images/test"

training_dataset = image_dataset_from_directory(
    directory_train,
    shuffle = True,
    batch_size = BATCH_SIZE,
    image_size = IMAGE_SIZE,
    validation_split = 0.2,
    subset = "training",
    seed = 42
)

validation_dataset = image_dataset_from_directory(
    directory_train,
    shuffle = True,
    batch_size = BATCH_SIZE,
    image_size = IMAGE_SIZE,
    validation_split = 0.2,
    subset = "validation",
    seed = 42
)

test_dataset = image_dataset_from_directory(
    directory_test,
    image_size=IMAGE_SIZE
)

model = tf.keras.models.Sequential([
    tfl.Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    tfl.MaxPooling2D((2, 2)),
    tfl.Conv2D(128, (3, 3), activation='relu'),
    tfl.MaxPooling2D((2, 2)),
    tfl.Conv2D(256, (3, 3), activation='relu'),
    tfl.MaxPooling2D((2, 2)),
    tfl.Flatten(),
    tfl.Dense(512, activation='relu'),
    tfl.Dense(156, activation='softmax')  # 156 classes for prediction
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(training_dataset,
                    validation_data=validation_dataset,
                    epochs=50)

model.save("src/models/tamil_character_recognition")

model = tf.keras.models.load_model("./src/models/tamil_character_recognition")
mapper = [0,   1,  10, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
    11, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,  12, 120,
    121, 122, 123, 124, 125, 126, 127, 128, 129,  13, 130, 131, 132,
    133, 134, 135, 136, 137, 138, 139,  14, 140, 141, 142, 143, 144,
    145, 146, 147, 148, 149,  15, 150, 151, 152, 153, 154, 155,  16,
    17,  18,  19,   2,  20,  21,  22,  23,  24,  25,  26,  27,  28,
    29,   3,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,   4,
    40,  41,  42,  43,  44,  45,  46,  47,  48,  49,   5,  50,  51,
    52,  53,  54,  55,  56,  57,  58,  59,   6,  60,  61,  62,  63,
    64,  65,  66,  67,  68,  69,   7,  70,  71,  72,  73,  74,  75,
    76,  77,  78,  79,   8,  80,  81,  82,  83,  84,  85,  86,  87,
    88,  89,   9,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99
]

def letter_image_to_class(img):
    global model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = mapper[np.argmax(prediction)]
    return predicted_class





