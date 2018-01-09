""" Training of model. """

import os
from time import time
import numpy as np
from tqdm import tqdm

from dataset.dataset import Dataset, Pipeline, B, V

class MultipleTraining:
    """ Class for training one model repeatedly. """
    def __init__(self, class_model, data, config, feed_dict, preproc_template=None, metrics=None):
        self.class_model = class_model
        self.data = data
        self.config = config
        self.feed_dict = feed_dict
        self.preproc_template = preproc_template
        self.metrics = metrics
        self.results = dict()

        self._make_outputs()

    def _make_outputs(self):
        self.fetches = tuple(['loss'] + ['output_'+name for name in self.metrics])
        self.output = dict(ops=self.metrics)
        self.config = {**self.config, 'output': self.output}

    def _model_template(self):
        template = Pipeline()
        for metric in ['loss']+self.metrics:
            template = template + Pipeline().init_variable(metric, init_on_each_run=list)

        save_to = [V(metric) for metric in ['loss']+self.metrics]

        self.train_template = template + Pipeline().train_model('model',
                                                                fetches=self.fetches,
                                                                feed_dict=self.feed_dict,
                                                                save_to=save_to,
                                                                mode='a')
        self.test_template = template + Pipeline().predict_model('model',
                                                                 fetches=self.fetches,
                                                                 feed_dict=self.feed_dict,
                                                                 save_to=save_to,
                                                                 mode='a')

    def _make_pipelines(self):

        self._model_template()

        self.train_ppl = (self.preproc_template +
                          Pipeline().init_model('dynamic', self.class_model, 'model', config=self.config) +
                          self.train_template)
        self.train_ppl = self.train_ppl << self.data.train

        self.test_ppl = (self.preproc_template +
                         Pipeline().import_model('model', self.train_ppl) +
                         self.test_template)
        self.test_ppl = self.test_ppl << self.data.test

    def _reset_model(self):
        for metric in ['loss']+self.metrics:
            self.train_ppl.set_variable(metric, list())
            self.test_ppl.set_variable(metric, list())
        graph = self.train_ppl.get_model_by_name('model').graph.get_collection('trainable_variables')
        sess = self.train_ppl.get_model_by_name('model').session
        for var in graph:
            sess.run(var.initializer)
        self.train_ppl.reset_iter()
        self.test_ppl.reset_iter()

    def _empty_results(self):
        """ Create empty dict for results of research. """
        self.results = {
          'time': list(),
          'train': dict(),
          'test': dict()  
        }
        for metric in ['loss'] + self.metrics:
            self.results['train'][metric] = list()
            self.results['test'][metric] = list()

    def _append_results(self, results):
        """ Append results of single training. """
        self.results['time'].append(results['time'])
        for metric in ['loss']+self.metrics:
            self.results['train'][metric].append(results['train'][metric])
            self.results['test'][metric].append(results['test'][metric])

    def run(self, batch_size, n_iters, n_reps):
        """ Run training and save statistics into attribute results 
        
        Parameters
        ----------
        n_reps : int
            the number of repetitions for each combination of parameters
        batch_size : int

        n_iters : int
        """
        self._make_pipelines()
        self._empty_results()

        for experiment in tqdm(range(n_reps)):
            train_time = []
            if experiment != 0:
                self._reset_model()
            for _ in range(n_iters):
                start = time()
                self.train_ppl.next_batch(batch_size, shuffle=True, n_epochs=None)
                self.test_ppl.next_batch(batch_size, shuffle=True, n_epochs=None)
                train_time.append(time() - start)
        
            results = {
                'time': sum(train_time),
            }
            results['train'] = dict()
            results['test'] = dict()
            for metric in ['loss'] + self.metrics:
                results['train'][metric] = self.train_ppl.get_variable(metric)
                results['test'][metric] = self.test_ppl.get_variable(metric)

            self._append_results(results)