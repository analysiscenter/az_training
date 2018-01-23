#pylint:disable=too-many-instance-attributes
#pylint:disable=too-few-public-methods
#pylint:disable=attribute-defined-outside-init

""" Training of model. """

import os
from collections import OrderedDict
from copy import copy, deepcopy
import pickle

from dataset import Config
from distributor import Tasks, Distributor, Worker
from singlerun import SingleRunning

class PipelineWorker(Worker):
    def task(self, item):
        i, task = item
        print('Run task', i)
        
        single_runnings = []        
        for config in task['configs']:
            single_runnings = SingleRunning()
            for name, pipeline in task['pipelines'].items():
                single_running.add_pipeline(deepcopy(pipeline['ppl']), pipeline['var'], config=pipeline['cfg'], name=name)
            single_running.add_common_config(config)
            single_runnings.append(single_running)

        for sr in single_runnings:
            pass

class MultipleRunning:
    def __init__(self):
        self.pipelines = OrderedDict()
        self.config = Config()
        self.results = None
        self.has_preproc = False

    def add_pipeline(self, pipeline, variables, preproc=None, config=None, name=None):
        """ Add new pipeline to research.
        Parameters
        ----------
        pipeline : dataset.Pipeline
        variables : str or list of strs
            names of pipeline variables to remember at each repetition
        config : dict (default None)
            pipeline config
        names : str (default None)
            name of pipeline. If None - name will be 'ppl_{index}'
        """
        name = name or 'ppl_' + str(len(self.pipelines))
        config = config or Config()
        variables = variables or []
        if preproc is not None:
            self.has_preproc = True

        if not isinstance(variables, list):
            variables = [variables]
        if name in self.pipelines:
            raise ValueError('Pipeline with name {} was alredy existed'.format(name))
        self.pipelines[name] = {'ppl': pipeline, 'cfg': config, 'var': variables, 'preproc': preproc}

    def add_grid_config(self, grid_config):
        self.grid_config = grid_config

    def _create_tasks(self, n_reps, n_iters, names, reuse_batch):
        self.tasks = (
            {'pipelines': self.pipelines,
             'n_iters': n_iters, 
             'names': names, 
             'configs': configs,
             'repetition': idx,
             }
             for configs in self.grid_config.gen_configs(reuse_batch)
             for idx in range(n_reps)
        )
        self.tasks = Tasks(self.tasks)

    def run(self, n_reps, n_iters, n_jobs=1, reuse_batch=1, worker_class=None, names=None):
        self._create_tasks(n_reps, n_iters, names, reuse_batch)
        worker_class = worker_class or PipelineWorker
        distr = Distributor(worker_class, n_jobs, reuse_batch)
        distr.run(self.tasks)

    def load_results(self, name=None):
        with open(name, 'rb') as file:
            return pickle.load(file)

    def _filename(self, alias, index):
        return os.path.join('results', alias, str(index))