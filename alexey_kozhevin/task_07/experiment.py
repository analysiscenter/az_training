""" Numerical experiments with networks. """
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from dataset.dataset import Dataset, Pipeline, action, inbatch_parallel, B, V
from dataset.opensets import MNIST, CIFAR10, CIFAR100

_DATASETS = {
    'mnist': [MNIST, (28, 28, 1), 10],
    'cifar': [CIFAR10, (32, 32, 3), 10],
    'cifar10': [CIFAR10, (32, 32, 3), 10],
    'cifar100': [CIFAR10, (32, 32, 3), 100],
}

class Experiment:
    def __init__(self, model, data, preproc_template=None, model_config=None, metrics=None):
        self.data = data
        self.model = model
        self.metrics = metrics
        self.preproc_template = preproc_template
        self.model_template = None
        self.model_config = None
        self.pipeline = None
        self.train_history = None
        self.test_history = None

        self._build()

    def _build(self):
        self._create_metrics()
        self._create_dataset()
        self._create_preproc_template()
        self._create_config()
        self._create_model_template()

        train_template = self.train_template
        test_template = self.train_template

        self.train_ppl = (self.preproc_template +
                                     Pipeline().init_model('dynamic', self.model, 'model', config=self.model_config) +
                                     self.train_template)
        self.train_ppl = self.train_ppl << self.data.train

        self.test_ppl = (self.preproc_template +
                                    Pipeline().import_model('model', self.train_ppl) +
                                    self.test_template)
        self.test_ppl = self.test_ppl << self.data.test

    def _create_metrics(self):
        if self.metrics is None:
            self.metrics = ('loss', 'accuracy')
            self.fetches = ('loss', 'output_accuracy')
            self.output = dict(ops=['accuracy'])

    def _create_dataset(self):
        if isinstance(self.data, str):
            self._images_labels(_DATASETS[self.data][1], _DATASETS[self.data][2])
            self.data = _DATASETS[self.data][0]()
        elif isinstance(data, Dataset):
            pass
        else:
            raise ValueError('data must be str or Dataset but {} was given'.format(type(data)))
        # TODO: add arrays

    def _images_labels(self, shape, n_classes):
        self.placeholders_config = {
           'images': {'shape': shape,
                      'type': 'float32',
                      'name': 'reshaped_images'},
    
           'labels': {'classes': n_classes,
                      'type': 'int32',
                      'transform': 'ohe',
                      'name': 'targets'}
           }
        self.feed_dict = {'images': B('images'),
                  'labels': B('labels')} 

    def _create_preproc_template(self):
        if self.preproc_template == None:
            self.preproc_template = Pipeline()


    def _create_model_template(self):
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

    def _create_config(self):
        if self.model_config is None:
            self.model_config = {'inputs': self.placeholders_config,
                                 'input_block/inputs': 'images',
                                 'batch_norm': {'momentum': 0.1},
                                 'output': self.output,
                                 'loss': 'ce',
                                 'optimizer': 'Adam',
                                 }


    def run(self, batch_size, n_iters, reps=10):
        train_history = {metric: list() for metric in self.metrics}
        test_history = {metric: list() for metric in self.metrics}
        
        for experiment in range(reps):
            if experiment != 0:
                for metric in self.metrics:
                    self.train_ppl.set_variable(metric, list())
                    self.test_ppl.set_variable(metric, list())
                graph = self.train_ppl.get_model_by_name('model').graph.get_collection('trainable_variables')
                sess = self.train_ppl.get_model_by_name('model').session
                for v in graph:
                    sess.run(v.initializer)
            print("Experiment {}".format(experiment))
            self.train_ppl.reset_iter()
            self.test_ppl.reset_iter()
            for i in range(n_iters):
                self.train_ppl.next_batch(batch_size, shuffle=True)
                self.test_ppl.next_batch(batch_size, shuffle=True)
            for metric in self.metrics:
                train_history[metric].append(self.train_ppl.get_variable(metric))
                test_history[metric].append(self.test_ppl.get_variable(metric))
        self.train_history = {k: np.array(v) for k, v in train_history.items()}
        self.test_history = {k: np.array(v) for k, v in test_history.items()}

    def plot(self):
        sns.set(color_codes=True)
        for metric in self.metrics:
            ax = sns.tsplot(self.train_history[metric])
            plt.title("Train " + metric)
            plt.show()
            ax = sns.tsplot(self.test_history[metric])
            plt.title("Test " + metric)
            plt.show()
