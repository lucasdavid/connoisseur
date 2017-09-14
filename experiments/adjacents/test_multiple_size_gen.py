from sacred import Experiment

from keras.preprocessing.image import ImageDataGenerator
from connoisseur.utils.image import DirectoryIterator

ex = Experiment('test-multiple-sizes-image-generator')


@ex.config
def config():
    iterations = 1
    data_dir = '/home/ldavid/datasets/vangogh/vgdb_2016/train/'


@ex.automain
def main(iterations, data_dir):
    g = ImageDataGenerator()
    data = DirectoryIterator(data_dir,
                             image_data_generator=g,
                             target_size=[None, None])

    for i in range(iterations):
        x, y = next(data)
        print('shapes:', x.shape, y.shape)

        for _x in x:
            print('x\'s inner shape:', _x.shape)
