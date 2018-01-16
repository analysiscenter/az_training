#pylint:disable=too-many-instance-attributes
#pylint:disable=too-few-public-methods
#pylint:disable=attribute-defined-outside-init

""" Training of model. """

import os
from copy import copy, deepcopy
from tqdm import tqdm
import numpy as np
import pickle

from dataset.dataset.models import BaseModel

class SingleRunning:
    """ Class for training one model repeatedly. """
    def __init__(self, prefix=None):
        self.pipelines = []
        self.results = Results()

    def add_pipeline(self, pipeline, variables, config=None, name=None):
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
        if name is None:
            name = 'ppl_' + str(len(self.pipelines))
        if config is None:
            config = dict()
        if variables is None:
            variables = []
        if not isinstance(variables, list):
            variables = [variables]
        if name in [pipeline['name'] for pipeline in self.pipelines]:
            raise ValueError('Pipeline with name {} was alredy existed'.format(name))
        self.pipelines.append({'name': name, 'ppl': pipeline, 'cfg': config, 'var': variables})

    def run(self, n_iters, names=None, additional_config=None, save_to=None):
        """ Run pipelines repeatedly. Pipelines will be executed simultaneously in the following sense:
        next_batch is applied successively to each pipeline in names list at each iteration. Pipelines
        at each repetition are copies of the initial pipelines.

        Parameters
        ----------
        n_iters : int
            number of iterations at each repetition
        n_reps : int (default 1)
            number of repeated runs of each pipeline
        names : str, int or list (default None)
            pipelines to run. If str - name of the pipeline. If int - index at self.pipelines.
            If list - list of names or list of indices. If None - all pipelines will be run.
        """
        results = Results()
        additional_config = additional_config if additional_config is not None else dict()
        pipelines = []
        names = names if names is not None else list(range(len(self.pipelines)))
        names = [names] if isinstance(names, (str, int)) else names
        
        if isinstance(names[0], int):
            pipelines = [self.pipelines[i] for i in names]
        else:
            pipelines = [pipeline for pipeline in self.pipelines if pipeline['name'] in names]

        for pipeline in pipelines:
            pipeline['ppl'].config = {**BaseModel.flatten(pipeline['cfg']), **BaseModel.flatten(additional_config)}
            pipeline['ppl'].config = BaseModel.parse(pipeline['ppl'].config)

        for _ in range(n_iters):
            for pipeline in pipelines:
                pipeline['ppl'].next_batch()

        for pipeline in pipelines:
            if len(pipeline['var']) != 0:
                _results = {variable: copy(pipeline['ppl'].get_variable(variable)) for variable in pipeline['var']}
                results.append(pipeline['name'], _results)

        save_to = '.' if save_to is None else save_to
        self._save_results(results, save_to)

    def _save_results(self, results, save_to=None):
        name = 'results' if save_to is None else save_to

        foldername, _ = os.path.split(name)
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        with open(name, 'wb') as file:
            pickle.dump(results, file)

    def _load_pipeline(self, ppl_name, index=None, name=None, prefix=None):
        prefix = '.' if prefix is None else prefix
        if index is None and name is None:
            raise ValueError('At least one of index and name must be defined.')
        name = str(index) if name is None else name
        ppl_name = 'ppl_'+str(ppl_name) if isinstance(ppl_name, int) else ppl_name

        foldername = os.path.join(prefix, 'pipelines', ppl_name)
        filename = os.path.join(foldername, name)
        with open(filename, 'rb') as file:
            return pickle.load(file)

    def _append_results(self, name, results):
        if name not in self.results:
            self.results[name] = dict()
        for variable in results:
            if variable not in self.results[name]:
                self.results[name][variable] = list()
            self.results[name][variable].append(results[variable])

class Results:
    """ Class for results of an experiment. """
    def __init__(self):
        self.stat = dict()

    def append(self, name, results):
        """ Append results.

        Parameters
        ----------
        name : str
        results : dict or Results
            results to append
        """
        if name not in self.stat:
            self.stat[name] = dict()
        for variable in results:
            if variable not in self.stat[name]:
                self.stat[name][variable] = list()
            if isinstance(results, dict):
                self.stat[name][variable].append(results[variable])
            else:
                self.stat[name][variable].extend(results.stat[variable])

    def __getitem__(self, index):
        return BaseModel.get(index, self.stat)
