"""Visualizing output of layers.


Author: Lucas David -- <ld492@drexel.edu>
License: MIT (c) 2016

"""
import os

from keras.applications import VGG19
from keras.engine import Model
from keras.preprocessing.image import load_img, img_to_array, array_to_img

BASE_IMAGE = './mona.jpg'
LAYERS = ['block1_conv1', 'block5_pool']
N_KERNELS = 10


def main():
    print(__doc__)

    print('loading %s...' % BASE_IMAGE)
    data = load_img(BASE_IMAGE, target_size=(224, 224, 3))
    x = img_to_array(data)
    x = x.reshape((1, 224, 224, 3))

    print('loading VGG-19...')
    base_model = VGG19(include_top=False)

    for layer_id in LAYERS:
        model = Model(input=base_model.input,
                      output=base_model.get_layer(layer_id).output)

        y = model.predict(x)

        for kernel_id in range(N_KERNELS):
            os.makedirs(layer_id, exist_ok=True)
            s = y[0, :, :, kernel_id]
            s = array_to_img(s.reshape(s.shape + (1,)))
            s.save('%s/kernel-%i.jpg' % (layer_id, kernel_id))


if __name__ == '__main__':
    main()
