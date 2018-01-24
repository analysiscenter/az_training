#pylint:disable=too-many-instance-attributes
#pylint:disable=too-few-public-methods
#pylint:disable=attribute-defined-outside-init

""" Training of model. """

import os
from copy import copy
from collections import OrderedDict
import numpy as np
import pickle

from dataset import Config, Pipeline

class SingleRunning:
    """ Class for training one model repeatedly. """
    def __init__(self):
        self.pipelines = OrderedDict()
        self.config = Config()
        self.results = None

    def add_pipeline(self, pipeline, variables, config=None, name=None, import_model_from=None):
        """ Add new pipeline to research.
        Parameters
        ----------
        pipeline : dataset.Pipeline
        variables : str or list of strs
            names of pipeline variables to remember at each repetition
        config : Config or dict (default None)
            pipeline config
        name : str (default None)
            name of pipeline. If None - name will be 'ppl_{index}'
        import_model_from : str (default None)
            name of pipeline for model importing. Parameters in actions 'import_model' must have parameter
            'C('import_model_from')'
        """
        name = name or 'ppl_' + str(len(self.pipelines))
        config = Config(config) or Config()
        variables = variables or []
        if not isinstance(variables, list):
            variables = [variables]
        if name in self.pipelines:
            raise ValueError('Pipeline with name {} was alredy existed'.format(name))
        if import_model_from is not None:
            import_dict = Config(import_model_from=self.pipelines[import_model_from]['ppl'])
        else:
            import_dict = dict()
        self.pipelines[name] = {
            'ppl': pipeline,
            'cfg': config + import_dict,
            'var': variables
        }

    def get_pipeline(self, name):
        return self.pipelines[name]

    def add_common_config(self, config):
        self.config = Config(config)

    def get_results(self):
        results = dict()
        for name, pipeline in self.pipelines.items():
            if len(pipeline['var']) != 0:
                results[name] = {variable: copy(pipeline['ppl'].get_variable(variable)) for variable in pipeline['var']}
        return results

    def init(self):
        for _, pipeline in self.pipelines.items():
            pipeline['ppl'].set_config(pipeline['cfg'] + self.config)

    def run_on_batch(self, batch, name):
        self.pipelines[name]['ppl'].execute_for(batch)

    def next_batch(self, name):
        self.pipelines[name]['ppl'].next_batch()
        
    def run(self, n_iters):
        """ Run pipelines repeatedly. Pipelines will be executed simultaneously in the following sense:
        next_batch is applied successively to each pipeline in names list at each iteration. Pipelines
        at each repetition are copies of the initial pipelines.

        Parameters
        ----------
        n_iters : int
            number of iterations at each repetition
        n_reps : int (default 1)
            number of repeated runs of each pipeline
        names : str or list (default None)
            pipelines to run. If str - name of the pipeline. If int - index at self.pipelines.
            If list - list of names or list of indices. If None - all pipelines will be run.
        """
        pipelines = self.pipelines.values()
        for _ in range(n_iters):
            for pipeline in pipelines:
                pipeline['ppl'].next_batch()
        self.results = self.get_results()

    def save_results(self, save_to=None):
        self.results = self.get_results()
        foldername, _ = os.path.split(save_to)
        if len(foldername) != 0:
            if not os.path.exists(foldername):
                os.makedirs(foldername)
        with open(save_to, 'wb') as file:
            pickle.dump(self.results, file)
