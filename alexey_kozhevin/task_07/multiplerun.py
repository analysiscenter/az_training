#pylint:disable=too-many-instance-attributes
#pylint:disable=too-few-public-methods
#pylint:disable=attribute-defined-outside-init

""" Training of model. """

import os
import sys
from itertools import product
from collections import OrderedDict
from copy import copy, deepcopy
import pickle
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from dataset import Config, Pipeline, inbatch_parallel
from distributor import Tasks, Distributor, Worker
from singlerun import SingleRunning

class PipelineWorker(Worker):

    @inbatch_parallel(init='_parallel_init')
    def _parallel_run(self, sr, single_runnings, batch, name):
        sr.run_on_batch(batch, name)

    def _parallel_init(self, single_runnings, batch, name):
        return [sr for sr in single_runnings]

    def task(self, item):
        i, task = item
        single_runnings = []
        for idx, config in enumerate(task['configs']):
            single_running = SingleRunning()
            for name, pipeline in task['pipelines'].items():
                pipeline_copy = pipeline['ppl'] + Pipeline()
                single_running.add_pipeline(pipeline_copy, pipeline['var'], config=pipeline['cfg'],
                                            name=name, import_model_from=pipeline['import_model_from'])
            if isinstance(task['model_per_preproc'], list):
                model_per_preproc = task['model_per_preproc'][idx]
            else:
                model_per_preproc = Config()
            single_running.add_common_config(config.config()+model_per_preproc)
            single_running.init()
            single_runnings.append(single_running)

        @inbatch_parallel(init=single_runnings)
        def _parallel_run(self, sr, batch, name):
            sr.run_on_batch(batch, name)

        for i in range(task['n_iters']):
            for name, pipeline in task['pipelines'].items():
                if pipeline['preproc'] is not None:
                    batch = pipeline['preproc'].next_batch()
                    self._parallel_run(single_runnings, batch, name)
                else:
                    for sr in single_runnings:
                        sr.next_batch(name)
        for sr, config in zip(single_runnings, task['configs']):
            sr.save_results(os.path.join(task['name'], 'results', config.alias(as_string=True), str(task['repetition'])))

class Research:
    def __init__(self):
        self.pipelines = OrderedDict()
        self.config = Config()
        self.results = None
        self.has_preproc = False

    def add_pipeline(self, pipeline, variables, preproc=None, config=None, name=None, import_model_from=None):
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
        self.pipelines[name] = {'ppl': pipeline, 'cfg': config, 'var': variables, 'preproc': preproc, 'import_model_from': import_model_from}

    def add_grid_config(self, grid_config):
        self.grid_config = grid_config

    def _create_tasks(self, n_reps, n_iters, model_per_preproc, name):
        if isinstance(model_per_preproc, int):
            n_models = model_per_preproc
        elif model_per_preproc is None:
            n_models = 1
        else:
            n_models = len(model_per_preproc)
        self.tasks = (
            {'pipelines': self.pipelines,
             'n_iters': n_iters,
             'configs': configs,
             'repetition': idx,
             'model_per_preproc': model_per_preproc,
             'name': name
             }
             for configs in self.grid_config.gen_configs(n_models)
             for idx in range(n_reps)
        )
        self.tasks = Tasks(self.tasks)

    def run(self, n_reps, n_iters, n_jobs=1, model_per_preproc=1, name=None):
        self.n_reps = n_reps
        self.n_iters = n_iters
        self.n_jobs = n_jobs
        self.model_per_preproc = model_per_preproc

        self.name = self._does_exist(name)

        self.save()
        self._create_tasks(n_reps, n_iters, model_per_preproc, self.name)
        distr = Distributor(n_jobs, model_per_preproc, PipelineWorker)
        distr.run(self.tasks, dirname=self.name)

    def _does_exist(self, name):
        name = name or 'research'
        if not os.path.exists(name):
            dirname = name
        else:
            i = 1
            while os.path.exists(name + '_' + str(i)):
                i += 1
            dirname = name + '_' + str(i)
        os.makedirs(dirname)
        return dirname        

    def save(self):
        with open(os.path.join(self.name, 'description'), 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, name):
        with open(os.path.join(name, 'description'), 'rb') as file:
            return pickle.load(file)