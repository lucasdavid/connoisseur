"""
Loading and Plotting Example.

Download, extract and load batches of Paintings91 data set.

Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""

import tensorflow as tf

from connoisseur import datasets


def main():
    print(__doc__)

    paintings91 = datasets.Paintings91(datasets.DataSetParameters(
        'Paintings91', batch_size=10, save_in='/home/ldavid/data'))

    try:
        paintings91.load().process()
    except KeyboardInterrupt:
        print('Fetching canceled. Bye.')
        return

    except:
        raise

    init = tf.initialize_all_variables()

    with tf.Session() as s:
        s.run(init)

        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coordinator)

        for i in range(10):
            batch = paintings91.next_batch()

            print(batch)

        coordinator.request_stop()
        coordinator.join(threads)


if __name__ == '__main__':
    main()
