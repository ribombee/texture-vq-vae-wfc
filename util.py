import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from tensorflow import keras

def normalize_and_hsv_image(image):
    image = image / 255  # Normalize
    image = tf.image.rgb_to_hsv(image)  # Swap to hsv

    return image

def denormalize_and_rgb_image(image):

    # Convert the image tensor back to RGB
    image = tf.image.hsv_to_rgb(image)

    #Denormalize
    image = image * 255

    return image

def get_image_processor(image_size, augment=False):

    data_augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal_and_vertical"),
        keras.layers.RandomBrightness(0.2),
        keras.layers.RandomContrast(0.2),
        keras.layers.RandomRotation(0.2),
        keras.layers.RandomZoom(0.2, 0.2),
        keras.layers.RandomCrop(image_size[0], image_size[1])]
    )

    def process_image(image):
        if augment:
            if np.random.rand() > 0.7:
                image = data_augmentation(image)

        image = tf.image.resize_with_crop_or_pad(image, image_size[0], image_size[1])
        image = normalize_and_hsv_image(image)
        return image

    return process_image



def hsv_tf_to_cv(image):

    hue, sat, val = cv.split(image)
    hue = np.clip(hue * 179, 0, 179)
    sat = np.clip(sat * 255, 0, 255)
    val = np.clip(val * 255, 0, 255)

    scaled = np.stack([hue, sat, val])
    scaled = np.transpose(scaled, (1, 2, 0))

    return scaled

def hsv_cv_to_tf(image):
    hue, sat, val = cv.split(image)
    hue = hue / 179
    sat = sat / 255
    val = val / 255

    normalized_sprite = np.stack([hue, sat, val])
    normalized_sprite = np.transpose(normalized_sprite, (1, 2, 0))

    return normalized_sprite


def plot_results(original, codes, reconstruction, img_save_path, codebook_size=64):
    plt.subplot(1, 3, 1)
    plt.imshow(cv.cvtColor(original, cv.COLOR_HSV2RGB))
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(codes, vmin= 0, vmax = codebook_size)
    # TODO: make this plot bigger or something to let annot=True fit in the plot nicely
    # sns.heatmap(codes, vmin=0, vmax=codebook_size)
    plt.title("Codes")
    plt.axis("Off")

    plt.subplot(1, 3, 3)
    plt.imshow(cv.cvtColor(reconstruction, cv.COLOR_HSV2RGB))
    plt.title("Reconstruction")
    plt.axis("off")

    plt.savefig(img_save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_results1(original, codes, reconstruction, img_save_path, codebook_size=64):
    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(codes, vmin= 0, vmax = codebook_size)
    plt.title("Codes")
    plt.axis("Off")

    plt.subplot(1, 3, 3)
    plt.imshow(reconstruction)
    plt.title("Reconstruction")
    plt.axis("off")

    plt.savefig(img_save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()