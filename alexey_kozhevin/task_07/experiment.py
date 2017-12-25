#pylint:disable=attribute-defined-outside-init
#pylint:disable=too-many-instance-attributes

""" Numerical experiments with networks. """
from time import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product
from ipywidgets import interactive

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
    'sgm': ['mse']
}


class Experiment:
    """ Class for multiple experiments with models. """
    def __init__(self, model, data, data_config=None,
                 task='cls', preproc_template=None, base_config=None, grid_config=None,
                 metrics=None):
        self.model = model
        self.data = data
        self.data_config = data_config
        self.task = task
        self.preproc_template = preproc_template
        self.base_config = base_config
        self.grid_config = grid_config
        self.metrics = metrics

        self._build()

    def _build(self):
        self._metrics()
        self._placeholders()
        self._dataset()
        self._preproc_template()
        self._base_config()
        self._model_template()

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

        if self.grid_config is not None:
            for parameter in self.grid_config.keys():
                template = template + Pipeline().init_variable(parameter, init_on_each_run=list)

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

    def _gen_config(self):
        if self.grid_config is not None:
            keys = self.grid_config.keys()
            values = self.grid_config.values()
            return (dict(zip(keys, parameters)) for parameters in product(*values))
        else:
            return [dict()]

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
                train_history[metric].append(
                    self.train_ppl.get_variable(metric))
                test_history[metric].append(self.test_ppl.get_variable(metric))
        return train_time / self.n_reps, train_history, test_history

    def run(self, batch_size, n_iters, n_reps=10):
        """ Run experiments. """
        self.batch_size = batch_size
        self.n_iters = n_iters
        self.n_reps = n_reps
        self.stat = []

        for additional_parameters in self._gen_config():
            config = {**self.base_config, **additional_parameters}

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
            stat['train_history'] = {key: np.array(value) for key, value in train_history.items()}
            stat['test_history'] = {key: np.array(value) for key, value in test_history.items()}
            stat['iter_time'] = stat['time'] / n_iters
            self.stat.append([additional_parameters, stat])

    def summary(self):
        """ Get description of the experiment. """
        print('Model:', self.model.__name__)
        print('Number of repetitions:', self.n_reps)
        print('Number of iterations:', self.n_iters)
        print('Batch size:', self.batch_size)
        for parameters, stat in self.stat:
            print('='*30)
            print(parameters)
            print('Mean train time: {0:4.2f} s'.format(stat['time']))
            print('Mean time per train step: {0:4.2f} s'.format(stat['iter_time']))
            for metric in self.metrics:
                mean = self._mean_metrics(stat, metric, iteration=-1)
                print('Train {}: {:4.2f}'.format(metric, mean[0]))
                print('Test  {}: {:4.2f}'.format(metric, mean[1]))

    def _mean_metrics(self, stat, metric, iteration=-1):
        res = [np.array(stat[history][metric]) for history in ['train_history', 'test_history']]
        res = [np.mean(x[:, -1]) for x in res]
        return res

    def get_plots(self, metric, params_ind, *args, **kwargs):
        """ Plot mean metrics history with confidence. """
        sns.set(color_codes=True)
        stat = self.stat[params_ind][1]
        sns.tsplot(stat['train_history'][metric], *args, **kwargs)
        plt.title("Train " + metric)
        plt.show()
        sns.tsplot(stat['test_history'][metric], *args, **kwargs)
        plt.title("Test " + metric)
        plt.show()

    def plot_density(self, iter, params_ind, metric, mode=None, xlim=None, ylim=None, *args, **kwargs):
        """ Plot histogram of the metric at the fixed iteration. 
        
        Parameters
        ----------
        iter : int
            iteration of interest
        params_ind : int
            index of the parameters combination
        metric : str

        mode : str, list or None
            'test' and/or 'train'
        xlim : tuple

        ylim : tuple
        """
        for name in mode:
            x = np.array(self.stat[params_ind][1][name][metric])[:, iter]
            sns.distplot(x, *args, **kwargs)
            if xlim is not None:
                plt.xlim(xlim)
            if ylim is not None:
                plt.ylim(ylim)
            plt.title("{}: {}".format(name, metric))
            plt.show()

    def plot_density_interactive(self, params_ind, metric, mode=None, *args, **kwargs):
        """ Interactive version of plot_density for different iter values. """
        if mode is None:
            mode = ['train', 'test']
        elif isinstance(mode, str):
            mode = [mode]
        mode = [name+'_history' for name in mode]
        def _interactive_f(iteration):
            self.plot_density(iteration, params_ind, metric, mode, *args, **kwargs)
        interactive_plot = interactive(_interactive_f, iteration=(0, self.n_iters-1))
        output = interactive_plot.children[-1]
        output.layout.height = str(300*(len(mode)))+'px'
        return interactive_plot
