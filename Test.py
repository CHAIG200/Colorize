from keras.models import model_from_json
from keras.preprocessing.image import img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb
from skimage.io import imsave
from matplotlib import pyplot as plt
import numpy as np
import os

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")


# Manual Testing of Images
fig, ax = plt.subplots(22, 2, figsize=(16, 100))
row = 0
colorize = []

print('Output of the Model')
for filename in os.listdir('Datasets/Test/'):
    colorize.append(img_to_array(load_img('Datasets/Test/' + filename)))
    ax[row, 0].imshow(load_img('Datasets/Test/' + filename), interpolation='nearest')
    row += 1

colorize = np.array(colorize, dtype=float)
colorize = rgb2lab(1.0 / 255 * colorize)[:, :, :, 0]
colorize = colorize.reshape(colorize.shape + (1,))

# Test model
output = loaded_model.predict(colorize)
output = output * 128

row = 0

# Output colorizations
for i in range(len(output)):
    cur = np.zeros((256, 256, 3))
    cur[:, :, 0] = colorize[i][:, :, 0]
    cur[:, :, 1:] = output[i]
    resImage = lab2rgb(cur)
    ax[row, 1].imshow(resImage, interpolation='nearest')
    row += 1

    imsave("result/img_"+str(i)+".png", resImage)
