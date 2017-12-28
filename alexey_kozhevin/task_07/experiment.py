#pylint:disable=attribute-defined-outside-init
#pylint:disable=too-many-instance-attributes
#pylint:disable=too-many-arguments

""" Numerical experiments with networks. """
import os
from time import time
from itertools import product
from subprocess import call
from collections import OrderedDict
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from ipywidgets import interactive
import pandas as pd

from dataset.dataset import Dataset, Pipeline, B, V
from dataset.dataset.opensets import MNIST, CIFAR10, CIFAR100

_DATASETS = {
    'mnist': [MNIST, (28, 28, 1), 10],
    'cifar': [CIFAR10, (32, 32, 3), 10],
    'cifar10': [CIFAR10, (32, 32, 3), 10],
    'cifar100': [CIFAR100, (32, 32, 3), 100],
}

_TASK_METRICS = {
    'cls': ['accuracy'],
}


class Experiment:
    """ Class for multiple experiments with models. """

    def build(self, model, data, data_config=None,
              task='cls', preproc_template=None, base_config=None, grid_config=None,
              aliases=False, metrics=None, name=None):
        """ Prepare experiment to run. """
        self.model = model
        self.data = data
        self.data_config = data_config
        self.task = task
        self.preproc_template = preproc_template
        self.base_config = base_config
        self.grid_config = grid_config
        self.aliases = aliases
        self.metrics = metrics
        self.dirname = name

        self._build()

    def _build(self):
        if self.dirname is None:
            self.dirname = self.model.__name__+'_experiment'
        if not os.path.exists(self.dirname):
            os.makedirs(self.dirname)

        if self.grid_config is None:
            self.grid_config = OrderedDict()
        else:
            self.grid_config = OrderedDict(self.grid_config)

        self._add_aliases()
        self._metrics()
        self._placeholders()
        self._dataset()
        self._preproc_template()
        self._base_config()
        self._model_template()
        self._create_description()

    def _metrics(self):
        if self.metrics is None:
            self.metrics = tuple(['loss']+_TASK_METRICS[self.task])
            self.fetches = tuple(['loss'] + ['output_'+name for name in _TASK_METRICS[self.task]])
            self.output = dict(ops=_TASK_METRICS[self.task])

    def _placeholders(self):
        if isinstance(self.data, str):
            shape = _DATASETS[self.data][1]
            n_classes = _DATASETS[self.data][2]
            self.placeholders_config = {
                'images': {'shape': shape,
                           'type': 'float32',
                           'name': 'reshaped_images'
                          },
                'labels': {'classes': n_classes,
                           'type': 'int32',
                           'transform': 'ohe',
                           'name': 'targets'
                          }
            }

            self.feed_dict = {'images': B('images'),
                              'labels': B('labels')}
        else:
            self.placeholders_config = None

    def _dataset(self):
        if isinstance(self.data, str):
            self.data = _DATASETS[self.data][0]()
        elif issubclass(self.data, Dataset):
            if self.base_config is None:
                raise ValueError('If data is not str base_config must be dict')
        else:
            raise ValueError('data must be str or Dataset subclass')
        # TODO: add arrays

    def _preproc_template(self):
        if self.preproc_template is None:
            self.preproc_template = Pipeline()

    def _model_template(self):
        template = Pipeline()
        for metric in self.metrics:
            template = template + Pipeline().init_variable(metric, init_on_each_run=list)

        save_to = [V(metric) for metric in self.metrics]

        self.train_template = template + Pipeline().train_model('model',
                                                                fetches=self.fetches,
                                                                feed_dict=self.feed_dict,
                                                                save_to=save_to,
                                                                mode='a')
        self.test_template = template + Pipeline().predict_model('model',
                                                                 fetches=self.fetches,
                                                                 feed_dict=self.feed_dict,
                                                                 save_to=save_to,
                                                                 mode='a')

    def _base_config(self):
        if self.base_config is None:
            self.base_config = {'inputs': self.placeholders_config,
                                'input_block/inputs': 'images',
                                'batch_norm': {'momentum': 0.1},
                                'output': self.output,
                                'loss': 'ce',
                                'optimizer': 'Adam',
                               }

    def _add_aliases(self):
        self.grid_config = [self._parse_parameter(*item) for item in self.grid_config.items()]
        self.grid_config = OrderedDict(self.grid_config)

    def _parse_parameter(self, key, value):
        if isinstance(key, tuple):
            pass
        elif isinstance(key, str):
            key = (key, key)
        if not self.aliases:
            value = [(item, i) for i, item in enumerate(value)]
        return key, value


    def _gen_config(self):
        keys = self.grid_config.keys()
        values = self.grid_config.values()
        return (self._get_dict_and_alias(keys, parameters) for parameters in product(*values))

    def _get_dict_and_alias(self, keys, parameters):
        _dict = OrderedDict(zip(keys, parameters))
        alias = OrderedDict()
        for key, value in _dict.items():
            alias[key[1]] = value[1]
        _dict = OrderedDict([(key[0], value[0]) for key, value in _dict.items()])
        return _dict, alias


    def _create_description(self):
        _dict = OrderedDict()
        for i, config in enumerate(self._gen_config()):
            _dict[i] = config[0]
        _dict = pd.DataFrame(_dict).transpose()
        filename = os.path.join(self.dirname, 'description.csv')
        _dict.to_csv(filename)

    def _reset_model(self):
        for metric in self.metrics:
            self.train_ppl.set_variable(metric, list())
            self.test_ppl.set_variable(metric, list())
        graph = self.train_ppl.get_model_by_name(
            'model').graph.get_collection('trainable_variables')
        sess = self.train_ppl.get_model_by_name('model').session
        for var in graph:
            sess.run(var.initializer)
        self.train_ppl.reset_iter()
        self.test_ppl.reset_iter()

    def _start_train(self):
        train_time = []
        for _ in range(self.n_iters):
            start = time()
            self.train_ppl.next_batch(self.batch_size, shuffle=True)
            self.test_ppl.next_batch(self.batch_size, shuffle=True)
            train_time.append(time() - start)
        return np.sum(train_time)

    def _multiple_train(self):
        train_history = {metric: list() for metric in self.metrics}
        test_history = {metric: list() for metric in self.metrics}
        train_time = 0
        for experiment in tqdm(range(self.n_reps)):
            if experiment != 0:
                self._reset_model()
            train_time += self._start_train()
            for metric in self.metrics:
                train_history[metric].append(self.train_ppl.get_variable(metric))
                test_history[metric].append(self.test_ppl.get_variable(metric))
            self._save_to('train', train_history)
            self._save_to('test', test_history)
        return train_time / self.n_reps, train_history, test_history

    def _save_to(self, filename, obj, tmp=True):
        subdir = '.tmp' if tmp else ''
        _dir = os.path.join(self.dirname, subdir)
        if not os.path.exists(_dir):
            os.makedirs(_dir)

        with open(os.path.join(_dir, filename), 'wb') as file:
            pickle.dump(obj, file)

    def run(self, batch_size, n_iters, n_reps=10):
        """ Run experiments. """
        self.batch_size = batch_size
        self.n_iters = n_iters
        self.n_reps = n_reps
        self.stat = []

        for additional_parameters in self._gen_config():
            config = {**self.base_config, **additional_parameters[0]}

            self.train_ppl = (self.preproc_template +
                              Pipeline().init_model('dynamic', self.model, 'model', config=config) +
                              self.train_template)
            self.train_ppl = self.train_ppl << self.data.train

            self.test_ppl = (self.preproc_template +
                             Pipeline().import_model('model', self.train_ppl) +
                             self.test_template)
            self.test_ppl = self.test_ppl << self.data.test

            stat = dict()
            train_time, train_history, test_history = self._multiple_train()
            stat['time'] = train_time
            stat['train'] = {key: np.array(value) for key, value in train_history.items()}
            stat['test'] = {key: np.array(value) for key, value in test_history.items()}
            stat['iter_time'] = stat['time'] / n_iters
            self.stat.append([additional_parameters, stat])
            self._clear_folder()
            self._save_to('stat', self.stat)
        self._clear_folder()
        self._dump()

    def summary(self, verbose=True):
        """ Get description of the experiment. """
        print('Model:', self.model.__name__)
        print('Number of repetitions:', self.n_reps)
        print('Number of iterations:', self.n_iters)
        print('Batch size:', self.batch_size)
        summ = OrderedDict()
        for parameters, stat in self.stat:
            row = OrderedDict()
            if verbose:
                print('='*30)
                alias = self._alias_to_str(parameters[1])
                print(alias)
                print('Mean train time: {0:4.2f} s'.format(stat['time']))
                print('Mean time per train step: {0:4.2f} s'.format(stat['iter_time']))
                for metric in self.metrics:
                    mean = self._mean_metrics(stat, metric, iteration=-1)
                    print('Train {}: {:4.2f}'.format(metric, mean[0]))
                    print('Test  {}: {:4.2f}'.format(metric, mean[1]))
            row['Time per train step'] = stat['iter_time']
            for metric in self.metrics:
                mean = self._mean_metrics(stat, metric, iteration=-1)
                row['Train '+metric] = mean[0]
                row['Test '+metric] = mean[1]
            summ[alias] = row
        return pd.DataFrame(summ, columns=summ.keys(), index=row.keys())#.transpose()

    def _alias_to_str(self, alias):
        res = ""
        for key, value in alias.items():
            res += str(key) + '-' + str(value) + '_'
        res = res[:-1]
        return res

    def _mean_metrics(self, stat, metric, iteration=-1):
        res = [np.array(stat[history][metric]) for history in ['train', 'test']]
        res = [np.mean(x[:, iteration]) for x in res]
        return res

    def _plot_density(self, iteration, metric, params_ind, window=0,
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
        for ind in params_ind:
            x = np.array(self.stat[ind][1][mode][metric])[:, left:right]
            x = x.reshape(-1)
            label = self._alias_to_str(self.stat[ind][0][1])
            sns.distplot(x, label=label, ax=ax, *args, **kwargs)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_title("{} {}: iteration {}".format(mode, metric, iteration+1))
        ax.legend(loc=9, bbox_to_anchor=(0.5, -0.1))
        fig.tight_layout()
        if show:
            plt.show()
        return ax

    def plot_density_interactive(self, metric, params_ind, window=0, mode=None, *args, **kwargs):
        """ Interactive version of plot_density for different iter values.

        Parameters
        ----------
        params_ind : int or dict
            parameters combination or index of that combination
        metric : str

        window : int
            distribution computed not exactly for the fixed iteration but for iterations
            in the [iteration-window, iteration+window+1]
        mode : str, list or None
            'test' and/or 'train'
        xlim : tuple

        ylim : tuple
        """
        def _interactive_f(iteration):
            self._plot_density(iteration, metric, params_ind, window, mode, *args, **kwargs)
        interactive_plot = interactive(_interactive_f, iteration=(0, self.n_iters-1))
        output = interactive_plot.children[-1]
        output.layout.height = str(300*(len(mode)))+'px'
        return interactive_plot

    def make_video(self, name, metric, params_ind, plots_per_sec=1.,
                   window=0, mode=None, key_frames=None, *args, **kwargs):
        """ Creates video with distribution. """
        name = os.path.join(self.dirname, name)
        if os.path.isfile(name):
            raise OSError("File {} is already created.".format(name))

        tmp_folder = os.path.join(self.dirname, '.tmp')

        try:
            call(['ffmpeg.exe'])
        except FileNotFoundError:
            raise FileNotFoundError("ffmpeg.exe was not found.")

        if not os.path.exists(tmp_folder):
            os.makedirs(tmp_folder)

        self._clear_folder()

        for iteration in range(self.n_iters):
            if key_frames is not None:
                frame = self._get_frame(iteration, key_frames)
                kwargs = {**kwargs, **frame}
            mask = '{:0' + str(int(np.ceil(np.log10(self.n_iters)))) + 'd}.png'
            mask = os.path.join(tmp_folder, '') + mask
            self._plot_density(iteration, metric, params_ind, window, mode, show=False, *args, **kwargs)
            plt.savefig(mask.format(iteration))
            plt.close()

        mask = '%0{}d.png'.format(int(np.ceil(np.log10(self.n_iters))))
        mask = os.path.join(tmp_folder, mask)
        res = call(["ffmpeg.exe", "-r", str(plots_per_sec), "-i", mask, "-c:v", "libx264", "-vf",
                    "fps=25", "-pix_fmt", "yuv420p", name])
        self._clear_folder()
        if res != 0:
            raise OSError("Video can't be created")

    def _get_frame(self, iteration, key_frames):
        indices = np.array([frame[0] for frame in key_frames])
        frames = np.array([frame[1] for frame in key_frames])
        return frames[np.where(indices <= iteration)][-1]

    def _clear_folder(self):
        dirname = os.path.join(self.dirname, '.tmp')
        for root, dirs, files in os.walk(dirname, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

    def _dump(self):
        with open(os.path.join(self.dirname, '.dump'), 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, name):
        """ Load experiment from dump. """
        with open(os.path.join(name, '.dump'), 'rb') as file:
            res = pickle.load(file)
        return res

    def __getstate__(self):
        _dict = self.__dict__
        del _dict['train_ppl']
        del _dict['test_ppl']
        del _dict['preproc_template']
        del _dict['train_template']
        del _dict['test_template']
        return _dict

