#pylint:disable=too-many-instance-attributes
#pylint:disable=too-few-public-methods
#pylint:disable=attribute-defined-outside-init

""" Training of model. """

import os
from copy import copy
from collections import OrderedDict
import pickle

from dataset.dataset import Config

class SingleRunning:
    """ Class for training one model repeatedly. """
    def __init__(self):
        self.pipelines = OrderedDict()
        self.config = Config()
        self.results = None

    def add_pipeline(self, pipeline, variables=None, config=None, name=None, import_model_from=None):
        """ Add new pipeline to research.
        Parameters
        ----------
        pipeline : dataset.Pipeline
            if preproc is None pipeline must have run action with lazy=True.
        variables : str or list of str or None
            names of pipeline variables to remember at each repetition.
        config : Config or dict (default None)
            pipeline config
        name : str (default None)
            name of pipeline. If name is None pipeline will have name 'ppl_{index}'
        import_model_from : str or None
            name of pipeline in SingleRunning to import model from. If pipeline imports model from other pipeline,
            corresponding parameter in import_model must have name 'import_model_from'.
        """
        name = name or 'ppl_' + str(len(self.pipelines))
        config = Config(config) or Config()
        variables = variables or []
        if not isinstance(variables, list):
            variables = [variables]
        if name in self.pipelines:
            raise ValueError('Pipeline with name {} was alredy existed'.format(name))
        if import_model_from is not None:
            import_model = Config(import_model_from=self.pipelines[import_model_from]['ppl'])
        else:
            import_model = Config()
        self.pipelines[name] = {
            'ppl': pipeline,
            'cfg': config + import_model,
            'var': variables
        }

    def get_pipeline(self, name):
        """
        Parameters
        ----------
        name : str
        """
        return self.pipelines[name]

    def add_common_config(self, config):
        """
        Add config that is common for all pipelines.

        Parameters
        ----------
        config : Config or dict
        """
        self.config = Config(config)

    def get_results(self):
        """ Get values of variables from pipelines.
        Returns
        -------
        dict
            key : pipeline name
            value : dict
                key : variable name
                value : variable

        If some pipeline was added without variables it will not be included into results.
        """
        results = dict()
        for name, pipeline in self.pipelines.items():
            if len(pipeline['var']) != 0:
                results[name] = {variable: copy(pipeline['ppl'].get_variable(variable)) for variable in pipeline['var']}
        return results

    def init(self):
        """
        Add common config to all pipelines.
        """
        for _, pipeline in self.pipelines.items():
            pipeline['ppl'].set_config(pipeline['cfg'] + self.config)

    def run_on_batch(self, batch, name):
        """
        Run pipeline on prepared batch.

        Parameters
        ----------
        batch : dataset.Batch

        name : str
            pipeline name
        """
        self.pipelines[name]['ppl'].execute_for(batch)

    def next_batch(self, name):
        """
        Get next batch from pipleine.

        Parameters
        ----------
        name : str
            pipeline name
        """
        return self.pipelines[name]['ppl'].next_batch()

    def run(self, n_iters):
        """ Run all pipelines. Pipelines will be executed simultaneously in the following sense:
        next_batch is applied successively to each pipeline at each iteration.

        Parameters
        ----------
        n_iters : int
            number of iterations at each repetition
        """
        pipelines = self.pipelines.values()
        for _ in range(n_iters):
            for pipeline in pipelines:
                pipeline['ppl'].next_batch()
        self.results = self.get_results()

    def save_results(self, save_to):
        """ Pickle results to file.

        Parameters
        ----------
        save_to : str
        """
        self.results = self.get_results()
        foldername, _ = os.path.split(save_to)
        if len(foldername) != 0:
            if not os.path.exists(foldername):
                os.makedirs(foldername)
        with open(save_to, 'wb') as file:
            pickle.dump(self.results, file)
