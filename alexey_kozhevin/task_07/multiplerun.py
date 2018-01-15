#pylint:disable=too-many-instance-attributes
#pylint:disable=too-few-public-methods
#pylint:disable=attribute-defined-outside-init

""" Training of model. """

from time import time
from tqdm import tqdm
from copy import deepcopy, copy

from dataset.dataset import Pipeline, V, C
from dataset.dataset.models import BaseModel

class MultipleRunning:
    """ Class for training one model repeatedly. """
    def __init__(self):
        self.pipelines = []
        self.results = Results()

    def pipeline(self, pipeline, variables, config=None, name=None):
        """ Add new pipeline to research.
        Parameters
        ----------
        pipeline : dataset.Pipeline
        variables : str or list of strs
            names of pipeline variables to remember at each repetition
        config : dict (default None)
            pipeline config
        run_kwargs : dict (default None)
            kwargs for next_batch of that pipeline
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

    def run(self, n_iters, n_reps=1, names=None, additional_config=None, return_results=False, *args, **kwargs):
        """ Run pipelines repeatedly. Pipelines will be executed simultaneously in the following sense:
        next_batch is applied successively to each pipeline in names list at each iteration. Pipelines at each repetition
        are copies of the initial pipelines.

        Parameters
        ----------
        n_reps : int
            number of repeated runs of each pipeline
        n_iters : int (default 1)
            number of iterations at each repetition
        names : str, int or list (default None)
            pipelines to run. If str - name of the pipeline. If int - index at self.pipelines.
            If list - list of names or list of indices. If None - all pipelines will be run.
        *args, **kwargs 
            parameters for next_batch which are common for each pipeline
        """
        results = Results()
        if additional_config is None:
            additional_config = dict()
        for experiment in tqdm(range(n_reps)):
            pipelines = []
            if names is None:
                names = list(range(len(self.pipelines)))
            elif isinstance(names, (str, int)):
                names = [names]
            if isinstance(names[0], int):
                pipelines = [self.pipelines[i] for i in names]
            else:
                pipelines = [pipeline for pipeline in self.pipelines if pipeline['name'] in names]
            # pipelines = [deepcopy(pipeline) for pipeline in pipelines]
            pipelines = [pipeline for pipeline in pipelines]
            for pipeline in pipelines:
                for variable in pipeline['var']:
                    pipeline['ppl'].set_variable(variable, list())

            for pipeline in pipelines:
                pipeline['ppl'].config = {**BaseModel.flatten(pipeline['cfg']), **BaseModel.flatten(additional_config)}
                pipeline['ppl'].config = BaseModel.parse(pipeline['ppl'].config)

            for i in range(n_iters):
                for pipeline in pipelines:
                    pipeline['ppl'].next_batch()
            
            for pipeline in pipelines:
                if len(pipeline['var']) != 0:
                    _results = {variable: copy(pipeline['ppl'].get_variable(variable)) for variable in pipeline['var']}
                    results.append(pipeline['name'], _results)
        if return_results:
            return results
        else:
            self.results = results

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
        results : dict
            results of new repetition which have structure dict(var1=[...], var2=[...], ...)
        """
        if name not in self.stat:
            self.stat[name] = dict()
        for variable in results:
            if variable not in self.stat[name]:
                self.stat[name][variable] = list()
            self.stat[name][variable].append(results[variable])

    def __getitem__(self, index):
        return BaseModel.get(index, self.stat)
