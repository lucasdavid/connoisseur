"""Connoisseur Utils.


Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""
import abc
import argparse
import gc
import json
import time
from datetime import datetime
import multiprocessing

import tensorflow as tf


class Timer:
    """Pretty Time Counter.

    Usage:

    >>> dt = Timer()
    >>> # Do some processing...
    >>> print('time elapsed: ', dt)
    time elapsed: 00:00:05
    """

    def __init__(self):
        self.start = time.time()

    def restart(self):
        self.start = time.time()

    def elapsed(self):
        return time.time() - self.start

    def pretty_elapsed(self):
        m, s = divmod(self.elapsed(), 60)
        h, m = divmod(m, 60)
        time_str = "%02d:%02d:%02d" % (h, m, s)
        return time_str

    def __str__(self):
        return self.pretty_elapsed()


class Constants:
    """Container for constants of an experiment, as well as an automatic
    loader for these.

    Parameters:
        data   -- the dict containing info regarding an execution.

    """

    def __init__(self, data):
        self._data = data

    def __getattr__(self, item):
        if item not in self._data:
            raise AttributeError('%s not an attribute of constants: %s'
                                 % (item, list(self._data.keys())))
        return self._data[item]

    @classmethod
    def from_json(cls):
        pass


class Experiment(metaclass=abc.ABCMeta):
    """Base Class for Experiments.

    Notes:
        the `run` method should be overridden for the experiment to be
        actually performed.

    Usage:

    >>> class ToyExperiment(Experiment):
    ...     def run(self):
    ...         print('Hello World!')
    ...
    >>> with ToyExperiment() as t:
    ...     t.run()
    Hello World
    >>> print(t.started_at)
    2016-10-11 14:40:22.454985
    >>> print(t.ended_at)
    2016-10-11 14:40:22.455061

    """

    def __init__(self, consts=None):
        self.consts = consts
        self.started_at = self.ended_at = None

    def run(self):
        raise NotImplementedError

    def __enter__(self):
        self.started_at = datetime.now()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.ended_at = datetime.now()


class ExperimentSet:
    """Container for experiments.

    Useful when executing multiple experiments or executing from multiple
    different environments.

    Usage:

    >>> class ToyExperiment(Experiment):
    ...     def run(self):
    ...         print('Hello World!')
    ...
    >>> experiments = ExperimentSet(ToyExperiment)
    >>> experiments.load_from_json('toy-experiments.json')
    >>> # Run all experiments!
    >>> experiments.run()
    >>> # Find out more about when these experiments started and ended.
    >>> for e in experiments:
    ...     print('experiment %s started at %s and ended at %s'
    ...           % (e.consts.code_name, e.started_at, e.ended_at))
    ...
    experiment julia started at 2016-10-11 14:52:12.216573
    and ended at 2016-10-11 14:52:14.688444

    """

    def __init__(self, experiment_cls, data=None):
        self.experiment_cls = experiment_cls
        self._data = None
        self._experiment_constants = None
        self.current_experiment_ = -1

        if data: self.load_from_object(data)

    def load_from_json(self, filename='./constants.json',
                       raise_on_not_found=False):
        data = {}

        try:
            with open(filename) as f:
                data = json.load(f)
        except IOError:
            if raise_on_not_found:
                raise

        return self.load_from_object(data)

    def load_from_object(self, data):
        self._data = data = data.copy()

        if isinstance(data, dict):
            base_params = (data['base_parameters']
                           if 'base_parameters' in data
                           else data)

            experiments_params_lst = (data['executions']
                                      if 'executions' in data
                                      else [{'code_name': 'julia'}])
        else:
            base_params = []
            experiments_params_lst = data

        self._experiment_constants = []

        for experiment_params in experiments_params_lst:
            params = base_params.copy()
            params.update(experiment_params)

            self._experiment_constants.append(Constants(params))

        return self

    def __iter__(self):
        self.current_experiment_ = -1
        return self

    def __next__(self):
        self.current_experiment_ += 1

        if self.current_experiment_ >= len(self._experiment_constants):
            self.current_experiment_ = -1
            raise StopIteration

        consts = self._experiment_constants[self.current_experiment_]
        return self.experiment_cls(consts)

    def __len__(self):
        return len(self._experiment_constants)

    def run(self):
        try:
            tf.logging.info('%i experiments in set' % len(self))

            for i, e in enumerate(self):
                with e:
                    tf.logging.info('experiment #%i: %s (%s)',
                                    i, e.consts.code_name, e.started_at)
                    e.run()
                tf.logging.info('experiment completed (%s)' % e.ended_at)

                del e
                gc.collect()

            tf.logging.info('experimentation set has finished')
        except KeyboardInterrupt:
            tf.logging.warning('interrupted by the user')


arg_parser = argparse.ArgumentParser(
    description='Experiments on Art Connoisseurship')

arg_parser.add_argument('--constants', type=str, default='./constants.json',
                        help='JSON file containing definitions for the '
                             'experiment.')
