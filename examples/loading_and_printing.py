"""
Loading and Plotting Example.

Download, extract and load batches of Paintings91 data set.

Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""

import time
import tensorflow as tf

from connoisseur import datasets


def main():
    print(__doc__)
    started_at = time.time()

    paintings91 = datasets.Paintings91(
        datasets.Parameters(n_epochs=20, batch_size=50, n_threads=1,
                            save_in='/home/ldavid/data'))
    try:
        print('Fetching data set...', end=' ')
        paintings91.load().preprocess()
        print('Done (%.2f sec).' % (time.time() - started_at))

    except KeyboardInterrupt:
        print('Canceled.')
        return

    with tf.Session() as s:
        images, labels = paintings91.as_batches()

        s.run(tf.initialize_all_variables())

        c = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=c)

        try:
            while not c.should_stop():
                i, l = s.run([images, labels])

                print(i.shape)
                print(l.shape)

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            c.request_stop()

        c.join(threads)


if __name__ == '__main__':
    main()
