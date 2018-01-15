#pylint:disable=attribute-defined-outside-init
#pylint:disable=too-many-instance-attributes
#pylint:disable=too-many-arguments
#pylint:disable=too-many-locals

""" Experiments with models. """

import os
import pickle
import numpy as np

from multiplerun import MultipleRunning
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

    def _indices_by_alias(self, aliases):
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
        for i, (config, _)  in enumerate(self.results):
            if self._is_subset(alias, config.alias()):
                index.append(i)
        return index

    def _index_by_grid(self, grid):
        aliases = [config.alias() for config in grid.gen_configs()]
        return self._indices_by_alias(aliases)

    def _is_subset(self, subset, superset):
        return all(item in superset.items() for item in subset.items())

    def __getitem__(self, ind):
        return self.results[ind]

    def get_results(self, cond):
        """ Get results for configs that satisfy cond
        
        Parameters
        ----------
        cond : Grid, Option; dict, int or list 
        """
        if isinstance(cond, (Grid, Option)):
            cond = self._index_by_grid(cond)
        elif isinstance(cond, list):
            cond = self._indices_by_alias(cond)
        else:
            cond = self._indices_by_alias([cond])
        cond = np.unique(cond)
        return [self.results[i] for i in self._indices_by_alias(cond)]


class Research(MultipleRunning):
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

        self._build()

    def grid_config(self, grid):
        """ Add Grid.

        Parameters
        ----------
        grid: Grid
        """
        self.grid = grid

    def _build(self):
        if not os.path.exists(self.name):
            os.makedirs(self.name)
        tmp_dir = os.path.join(self.name, '.tmp')
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        self.pipelines = list()
        self.grid = None
        self.results = ResearchResults()

    def run(self, n_iters, n_reps=1, names=None, *args, **kwargs):
        """ Run experiments.

        Parameters
        ----------
        n_reps : int
            the number of repetitions for each combination of parameters
        batch_size : int

        n_iters : int
        """
        self._clear_tmp_folder()
        for additional_config in self.grid.gen_configs():
            results = super().run(n_iters, n_reps, names, additional_config.config(), return_results=True)
            self.results.append(additional_config, results)
            self._save_results()

    def _clear_tmp_folder(self):
        dirname = os.path.join(self.name, '.tmp')
        for root, dirs, files in os.walk(dirname, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

    def _save_results(self):
        name = os.path.join(self.name, 'results')
        with open(name, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, name):
        """ Load experiment. """
        with open(os.path.join(name, 'results'), 'rb') as file:
            res = pickle.load(file)
        return res

    def __getstate__(self):
        """ Save all attributes except data. """
        _dict = self.__dict__.copy()
        #_dict['data'] = _dict['data'].__class__.__name__
        ppls = list()
        for pipeline in _dict['pipelines']:
            dct = pipeline.copy()
            dct['ppl'] = None
            ppls.append(dct)
        _dict['pipelines'] = dct
        return _dict
