#pylint:disable=attribute-defined-outside-init
#pylint:disable=too-many-instance-attributes
#pylint:disable=too-many-arguments
#pylint:disable=too-many-locals

""" Experiments with models. """

import os
from itertools import product
from subprocess import call
from collections import OrderedDict
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from dataset.dataset import Dataset, Pipeline
from dataset.dataset.opensets import MNIST, CIFAR10, CIFAR100

from training import MultipleTraining

_DATASETS = {
    'mnist': [MNIST, (28, 28, 1), 10],
    'cifar': [CIFAR10, (32, 32, 3), 10],
    'cifar10': [CIFAR10, (32, 32, 3), 10],
    'cifar100': [CIFAR100, (32, 32, 3), 100],
}

class Research:
    """ Class for multiple experiments with models. """
    def __init__(self, models, data, feed_dict, preproc_template=None, metrics=None, aliases=False, name=None):
        """ Initial experiment settings.

        Parameters
        ----------
        models : list of tuples (class_model, base_config, grid_config)
            class_model : TFModel
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
            input data. If str, must be 'mnist', 'cifar', 'cifar10' or 'cifar100'
        preproc_template : Pipeline
            pipeline to preprocess data. Default - None
        base_config : dict
        metrics : str or list of str
            metrics to compute on train and test. If None 'loss' will be assigned.
        aliases : bool
            see models
        name : str
        """
        self.models = models
        self.data = data
        self.feed_dict = feed_dict
        self.preproc_template = preproc_template
        self.aliases = aliases
        self.metrics = metrics
        self.name = name

        self._build()

    def _build(self):
        if not os.path.exists(self.name):
            os.makedirs(self.name)
        tmp_dir = os.path.join(self.name, '.tmp')
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        self.results = list()
        self.vizualizations = {
            'density': self._plot_density
        }

        self._defaults()
        self._dataset()
        self._add_aliases()

    def run(self, batch_size, n_iters, n_reps):
        """ Run experiments.

        Parameters
        ----------
        n_reps : int
            the number of repetitions for each combination of parameters
        batch_size : int

        n_iters : int
        """
        self.n_reps = n_reps
        self.batch_size = batch_size
        self.n_iters = n_iters
        for class_model, base_config, grid_config in self.models:
            for additional_parameters in self._gen_config(grid_config):
                config = {**base_config, **additional_parameters[0]}
                training_class = MultipleTraining(class_model, self.data, config, self.feed_dict,
                                                  self.preproc_template, self.metrics)
                training_class.run(batch_size, n_iters, n_reps)
                self.results.append((class_model.__name__, *additional_parameters, training_class.results))
                self._save_results()

    def summary(self):
        """ Get description of the experiment.
        Returns
        -------
        pd.DataFrame
        """
        print('Number of repetitions:', self.n_reps)
        print('Number of iterations:', self.n_iters)
        print('Batch size:', self.batch_size)
        summ = OrderedDict()
        for model, _, alias, stat in self.results:
            row = OrderedDict()
            alias = model + '_' + self._alias_to_str(alias)
            for metric in ['loss']+self.metrics:
                mean = self._mean_metrics(stat, metric, iteration=-1)
                row['Train '+metric] = mean[0]
                row['Test '+metric] = mean[1]
            row['Time'] = np.mean(stat['time'])
            summ[alias] = row
        return pd.DataFrame(summ, columns=summ.keys(), index=row.keys()).transpose()

    def _defaults(self):
        """ Assign default values. """
        if self.name is None:
            self.name = 'research'
        if self.metrics is None:
            self.metrics = []
        elif isinstance(self.metrics, str):
            self.metrics = [self.metrics]
        if self.preproc_template is None:
            self.preproc_template = Pipeline()

    def _dataset(self):
        """ Transform str value of self.data to Dataset. """
        if isinstance(self.data, str):
            self.data = _DATASETS[self.data][0]()
        elif issubclass(self.data, Dataset):
            pass
        else:
            raise ValueError('data must be str or Dataset subclass')

    def _add_aliases(self):
        """ Add aliases to grid.config if needed. """
        models_with_aliases = []
        for class_model, base_config, grid_config in self.models:
            grid_config = dict([self._item_aliases(*item) for item in grid_config.items()])
            models_with_aliases.append((class_model, base_config, grid_config))
        self.models = models_with_aliases

    def _item_aliases(self, key, value):
        """ Transform dict items to (key, value) where
            key = (parameter_name, parameter_alias)
            value = list of (parameter_value, value_alias)
        """
        if isinstance(key, tuple):
            pass
        elif isinstance(key, str):
            key = (key, key)
        if not self.aliases:
            value = [(item, i) for i, item in enumerate(value)]
        return key, value

    def _gen_config(self, grid_config):
        """ Generate tuples (config, config_alias) from grid_config. """
        keys = grid_config.keys() # it is important that keys and values are in the same order
        values = grid_config.values()
        return (self._get_dict_and_alias(keys, parameters) for parameters in product(*values))

    def _get_dict_and_alias(self, keys, parameters):
        """ Create dict of parameters and corresponding dict of aliases. """
        _dict = dict(zip(keys, parameters))
        alias = dict()
        for key, value in _dict.items():
            alias[key[1]] = value[1]
        _dict = dict([(key[0], value[0]) for key, value in _dict.items()])
        return _dict, alias

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
        _dict['data'] = _dict['data'].__class__.__name__
        return _dict

    def _alias_to_str(self, alias):
        res = ""
        for key, value in alias.items():
            res += str(key) + '-' + str(value) + '_'
        res = res[:-1]
        return res

    def _mean_metrics(self, stat, metric, iteration=-1):
        res = [np.array(stat[part][metric]) for part in ['train', 'test']]
        res = [np.mean(x[:, iteration]) for x in res]
        return res

    def _index_by_alias(self, model_name, alias):
        index = -1
        for i, (model, _, config, _)  in enumerate(self.results):
            if config == alias and model == model_name:
                index = i
                break
        return index

    def _indices_by_alias(self, model_name, aliases):
        output = []
        for alias in aliases:
            index = self._index_by_alias(model_name, alias)
            if index == -1:
                raise ValueError("Config wasn't found in experiment results.")
            output.append(index)
        return output

    def _plot_density(self, iteration, params_ind=None, metric='loss', window=0,
                      mode=None, xlim=None, ylim=None, axes=None, figsize=None,
                      show=True, *args, **kwargs):
        """ Plot histogram of the metric at the fixed iteration. """
        if isinstance(params_ind, int):
            params_ind = [params_ind]
        left = max(iteration-window, 0)
        right = min(iteration+window+1, self.n_iters)
        if figsize is None:
            fig = plt.figure()
        else:
            fig = plt.figure(figsize=figsize)
        if axes is None:
            axes = [0.1, 0.4, 0.8, 0.5]
        ax = fig.add_axes(axes)

        if params_ind is None:
            params_ind = list(range(len(self.results)))

        for ind in params_ind:
            stat = self.results[ind]
            x = np.array(stat[-1][mode][metric])[:, left:right]
            x = x.reshape(-1)
            label = stat[0]+'_'+self._alias_to_str(stat[2])
            sns.distplot(x, label=label, ax=ax, *args, **kwargs)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_title("{} {}: iteration {}".format(mode, metric, iteration+1))
        ax.legend(loc=9, bbox_to_anchor=(0.5, -0.1))
        if show:
            plt.show()
        return ax

    def make_video(self, vizualization, name, params_ind=None, plots_per_sec=1., key_frames=None, *args, **kwargs):
        """ Creates video with distribution. """
        name = os.path.join(self.name, name)
        if os.path.isfile(name):
            os.remove(name)
        tmp_folder = os.path.join(self.name, '.tmp')

        try:
            call(['ffmpeg.exe'])
        except FileNotFoundError:
            raise FileNotFoundError("ffmpeg.exe was not found.")

        if not os.path.exists(tmp_folder):
            os.makedirs(tmp_folder)

        self._clear_tmp_folder()

        for iteration in range(self.n_iters):
            if key_frames is not None:
                frame = self._get_frame(iteration, key_frames)
                kwargs = {**kwargs, **frame}
            mask = '{:0' + str(int(np.ceil(np.log10(self.n_iters)))) + 'd}.png'
            mask = os.path.join(tmp_folder, '') + mask
            self.vizualizations[vizualization](iteration, params_ind, show=False, *args, **kwargs)
            plt.savefig(mask.format(iteration))
            plt.close()

        mask = '%0{}d.png'.format(int(np.ceil(np.log10(self.n_iters))))
        mask = os.path.join(tmp_folder, mask)
        res = call(["ffmpeg.exe", "-r", str(plots_per_sec), "-i", mask, "-c:v", "libx264", "-vf",
                    "fps=25", "-pix_fmt", "yuv420p", name])
        self._clear_tmp_folder()
        if res != 0:
            raise OSError("Video can't be created")

    def _get_frame(self, iteration, key_frames):
        output = dict()
        for parameter, parameter_values in key_frames.items():
            if callable(parameter_values):
                output[parameter] = parameter_values(iteration)
            elif isinstance(parameter_values, list):
                indices = np.array([value[0] for value in parameter_values])
                frames = np.array([value[1] for value in parameter_values])
                output[parameter] = frames[np.where(indices <= iteration)][-1]
        return output
