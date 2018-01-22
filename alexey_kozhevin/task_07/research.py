#pylint:disable=attribute-defined-outside-init
#pylint:disable=too-many-instance-attributes
#pylint:disable=too-many-arguments
#pylint:disable=too-many-locals

""" Experiments with models. """

import os
from copy import deepcopy
import pickle
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from singlerun import SingleRunning
from grid import Grid, Option

class ResearchResults:
    """ Class for results of research. """
    def __init__(self, results=None):
        self.results = results if results is not None else list()

    def append(self, config, new_results):
        """ Append results.

        Parameters
        ----------
        config : Config
        new_results : Results
        """
        self.results.append((config, new_results))

    def __iter__(self):
        return self.results

    def _index_by_aliases(self, aliases):
        output = []
        for alias in aliases:
            if isinstance(alias, dict):
                index = self._index_by_alias(alias)
            else:
                index = [alias]
            output.extend(index)
        return output

    def _index_by_alias(self, alias):
        index = []
        for i, (config, _) in enumerate(self.results):
            if self._is_subset(alias, config.alias()):
                index.append(i)
        return index

    def _index_by_grid(self, grid):
        aliases = [config.alias() for config in grid.gen_configs()]
        return self._index_by_aliases(aliases)

    def _is_subset(self, subset, superset):
        return all(item in superset.items() for item in subset.items())

    def _load_results(self, filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)

    def __getitem__(self, ind):
        res = self.results[ind]
        if isinstance(res[1], str):
            res = (res[0], self._load_results(res[1]))
        return res

    def get_results(self, cond):
        """ Get results for configs that satisfy cond

        Parameters
        ----------
        cond : Grid, Option; dict, int or list
        """
        if isinstance(cond, (Grid, Option)):
            cond = self._index_by_grid(cond)
        elif isinstance(cond, list):
            cond = self._index_by_aliases(cond)
        else:
            cond = self._index_by_aliases([cond])
        cond = np.unique(cond)
        return [self[i] for i in self._index_by_aliases(cond)]


class Research:
    """ Class for multiple experiments with models. """
    def __init__(self, name=None):
        """ Initial experiment settings.

        Parameters
        ----------
        pipelines : dataset.Pipeline
            base_config : dict
                model config with parameters which are not changes between experiments
            grid_config : dict
                key : str or tuple of str
                    if str - parameter name. if tuple - (parameter name, parameter alias)
                value : list
                    if aliases=True it must be list of tuples (parameter value, value alias)
                    if aliases=False list of parameter values
        feed_dict : dict

        data : str or Dataset
            input data. If str, must be 'mnist', 'cifar', 'cifar10' or 'cifar100'. If Dataset, it must has attributes
            train and test.
        preproc_template : Pipeline
            pipeline to preprocess data which is applied to data.test and data.train. Default - None (empty pipeline).
        metrics : str or list of str
            metrics to compute on train and test. If None 'loss' will be assigned.
        aliases : bool
            see models
        name : str
        """
        super().__init__()
        if name is None:
            name = 'research'
        self.name = name

        if not os.path.exists(self.name):
            os.makedirs(self.name)
        tmp_dir = os.path.join(self.name, '.tmp')
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        self.pipelines = list()
        self.grid = None
        self.results = ResearchResults()

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

    def add_grid_config(self, grid):
        """ Add Grid.

        Parameters
        ----------
        grid: Grid
        """
        self.grid = grid

    def run(self, n_iters, n_reps=1, names=None, max_workers=None):
        """ Run experiments.

        Parameters
        ----------
        n_reps : int
            the number of repetitions for each combination of parameters
        batch_size : int

        n_iters : int
        """
        self.results = ResearchResults()
        self._clear_tmp_folder()

        def _run(arg):
            indices, additional_config = arg
            single_run = SingleRunning()
            results = dict()
            for experiment in indices:
                single_run.pipelines = deepcopy(self.pipelines)
                save_to = os.path.join('.', self.name, 'results',
                                       additional_config.alias(as_string=True), str(experiment))
                experiment_result = single_run.run(n_iters, names, additional_config.config())
                self._save_results(experiment_result, save_to)
                results[experiment] = save_to
            del single_run
            return results

        for additional_config in self.grid.gen_configs():
            tasks = [([i], additional_config) for i in range(n_reps)]
            with ThreadPoolExecutor(max_workers) as executor:
                exec_output = executor.map(_run, tasks)
                exec_output = dict(item for dct in exec_output for item in dct.items())
                self.results.append(additional_config, exec_output)

    def _save_results(self, results, name=None):
        foldername, _ = os.path.split(name)
        if len(foldername) != 0:
            if not os.path.exists(foldername):
                os.makedirs(foldername)
        with open(name, 'wb') as file:
            pickle.dump(results, file)

    def _clear_tmp_folder(self):
        dirname = os.path.join(self.name, '.tmp')
        for root, dirs, files in os.walk(dirname, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

    @classmethod
    def load_results(cls, name):
        """ Load experiment. """
        with open(os.path.join(name, 'research'), 'rb') as file:
            res = pickle.load(file)
        return res
