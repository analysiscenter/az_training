#pylint:disable=too-many-instance-attributes
#pylint:disable=too-few-public-methods
#pylint:disable=attribute-defined-outside-init

""" Training of model. """

import os
from copy import copy
import pickle

from dataset import Config
from distributor import Tasks, Distributor, Worker
from singlerun import SingleRunning

class PipelineWorker(Worker):
    def task(self, item):
        i, task = item
        print('Run task', i)
        single_running = SingleRunning()
        for name, pipeline in task['pipelines'].items():
            single_running.add_pipeline(pipeline['ppl'], pipeline['var'], pipeline['cfg'], name)
        single_running.set_config(task['config'])
        single_running.run(task['n_iters'], task['names'])
        single_running.save_results(task['save_to'])

class MultipleRunning(SingleRunning):
    def add_grid_config(self, grid_config):
        self.grid_config = grid_config

    def _create_tasks(self, n_reps, n_iters, names):
        self.tasks = (
            {'pipelines': self.pipelines, 
             'n_iters': n_iters, 
             'names': names, 
             'config': config.config(),
             'save_to': os.path.join('results', config.alias(as_string=True), str(idx)),
             }
             for config in self.grid_config.gen_configs()
             for idx in range(n_reps)
        )
        self.tasks = Tasks(self.tasks)

    def run(self, n_reps, n_iters, names=None, n_jobs=1, worker_class=None):
        worker_class = worker_class if worker_class is not None else PipelineWorker
        self._create_tasks(n_reps, n_iters, names)
        distr = Distributor(worker_class, n_jobs)
        distr.run(self.tasks)

    def load_results(self, name=None):
        with open(name, 'rb') as file:
            return pickle.load(file)



