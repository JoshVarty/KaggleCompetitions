
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def gaussBlur(input, width, height, radius):
    output = np.zeros_like(input)
    radius = np.ceil(radius * 2.57)

    for i in range(height):
        for j in range(width):
            val = 0
            wsum = 0

            for index_y in range(i - int(radius), i + int(radius) + 1):
                for index_x in range(j - int(radius), j + int(radius) + 1):
                    x = min(width - 1, max(0, index_x))
                    y = min(height - 1, max(0, index_y))

                    dsq = (index_x - j) * (index_y - j) + (index_y - j) * (index_x - j)
                    weight = np.exp(-dsq / (2 * radius * radius)) / (np.pi * 2 * radius * radius)

                    val += input[y * width+x] * weight
                    wsum += weight;

            output[i * width + j] = np.round(val/wsum)
                    

    return output

def display(img):
    
    one_image = img.reshape(img_size, img_size)
    plt.axis('off')
    plt.imshow(one_image, cmap=cm.binary)
    plt.show()

data = pd.read_csv("../input/train.csv");

#Partition into train/valid sets
img_size = 28
images = data.iloc[:,1:].as_matrix()
first = images[0]

blurred = gaussBlur(first, img_size, img_size, 1)

display(first)
display(blurred)
