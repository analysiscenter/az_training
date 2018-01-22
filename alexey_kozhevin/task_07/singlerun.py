#pylint:disable=too-many-instance-attributes
#pylint:disable=too-few-public-methods
#pylint:disable=attribute-defined-outside-init

""" Training of model. """

import os
from copy import copy
from collections import OrderedDict
import pickle

from dataset import Config

class SingleRunning:
    """ Class for training one model repeatedly. """
    def __init__(self):
        self.pipelines = OrderedDict()
        self.config = Config()
        self.results = None

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
            config = Config()
        if variables is None:
            variables = []
        if not isinstance(variables, list):
            variables = [variables]
        if name in self.pipelines:
            raise ValueError('Pipeline with name {} was alredy existed'.format(name))
        self.pipelines[name] = {'ppl': pipeline, 'cfg': config, 'var': variables}

    def get_pipeline(self, name):
        return self.pipelines[name]

    def set_config(self, config):
        self.config = config

    def run(self, n_iters, names=None):
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
        pipelines = []
        names = names if names is not None else self.pipelines.keys()
        names = [names] if isinstance(names, str) else names

        pipelines = [self.pipelines[name] for name in names]

        for pipeline in pipelines:
            pipeline['ppl'].config = (pipeline['cfg'] + self.config).config

        for _ in range(n_iters):
            for pipeline in pipelines:
                pipeline['ppl'].next_batch()

        results = dict()
        for name, pipeline in zip(names, pipelines):
            if len(pipeline['var']) != 0:
                results[name] = {variable: copy(pipeline['ppl'].get_variable(variable)) for variable in pipeline['var']}
        self.results = results

    def save_results(self, save_to=None):
        foldername, _ = os.path.split(save_to)
        if len(foldername) != 0:
            if not os.path.exists(foldername):
                os.makedirs(foldername)
        with open(save_to, 'wb') as file:
            pickle.dump(self.results, file)
