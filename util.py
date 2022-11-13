import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 as cv
import seaborn as sns

def get_image_processor(image_size):

    def process_image(image):
        image = tf.image.resize_with_crop_or_pad(image, image_size[0], image_size[1])

        image = image / 255 # Normalize
        image = tf.image.rgb_to_hsv(image) # Swap to hsv

        return image

    return process_image


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
