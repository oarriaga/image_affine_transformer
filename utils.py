import matplotlib.pyplot as plt
import numpy as np

def display_image(image_array):
    image_array = np.squeeze(image_array)
    plt.imshow(image_array)
    plt.show()


