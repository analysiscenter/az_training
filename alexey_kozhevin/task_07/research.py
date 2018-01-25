#pylint:disable=too-many-instance-attributes
#pylint:disable=too-few-public-methods
#pylint:disable=attribute-defined-outside-init
#pylint:disable=bare-except

""" Class Research and auxiliary classes for multiple experiments. """

import os
import sys
from collections import OrderedDict
import pickle

from dataset.dataset import Config, Pipeline, inbatch_parallel
from distributor import Tasks, Distributor, Worker
from grid import Grid
from singlerun import SingleRunning

class PipelineWorker(Worker):
    """ Worker that run pipelines. """
    @inbatch_parallel(init='_parallel_init')
    def _parallel_run(self, sr, single_runnings, batch, name):
        _ = single_runnings
        sr.run_on_batch(batch, name)

    def _parallel_init(self, single_runnings, batch, name):
        _ = batch, name
        return single_runnings

    def task(self, item):
        """
        Parameters
        ----------
        item : tuple (index, task)
        """
        i, task = item
        single_runnings = []
        print('Task', i)
        for idx, config in enumerate(task['configs']):
            print(config)
            single_running = SingleRunning()
            for name, pipeline in task['pipelines'].items():
                pipeline_copy = pipeline['ppl'] + Pipeline()
                single_running.add_pipeline(pipeline_copy, pipeline['var'], config=pipeline['cfg'],
                                            name=name, import_model_from=pipeline['import_model_from'])
            if isinstance(task['model_per_preproc'], list):
                model_per_preproc = task['model_per_preproc'][idx]
            else:
                model_per_preproc = Config()
            single_running.add_common_config(config.config()+model_per_preproc)
            single_running.init()
            single_runnings.append(single_running)

        for i in range(task['n_iters']):
            for name, pipeline in task['pipelines'].items():
                if pipeline['preproc'] is not None:
                    batch = pipeline['preproc'].next_batch()
                    self._parallel_run(single_runnings, batch, name)
                else:
                    for item in single_runnings:
                        item.next_batch(name)
        for item, config in zip(single_runnings, task['configs']):
            item.save_results(os.path.join(task['name'], 'results',
                              config.alias(as_string=True), str(task['repetition'])))

class Research:
    """ Class Research for multiple experiments with pipelines. """
    def __init__(self):
        self.pipelines = OrderedDict()
        self.config = Config()
        self.results = None
        self.has_preproc = False

    def add_pipeline(self, pipeline, variables, preproc=None, config=None, name=None, import_model_from=None):
        """ Add new pipeline to research.

        Parameters
        ----------
        pipeline : dataset.Pipeline
            if preproc is None pipeline must have run action with lazy=True. All parameters that are
            getted from grid should be defined as C('parameter_name'). Corresponding parameter in grid
            must have name 'parameter_name'
        variables : str or list of str
            names of pipeline variables to remember at each repetition. All of them must be defined in pipeline,
            not in preproc.
        preproc : dataset.Pipeline or None
            if preproc is not None it must have run action with lazy=True. For resulting batch of the preproc
            pipeline.execute_for(batch) will be called.
        config : Config or dict (default None)
            pipeline config which doesn't change between experiments
        name : str (default None)
            name of pipeline. If name is None pipeline will have name 'ppl_{index}'
        import_model_from : str or None
            name of pipeline in Research to import model from. If pipeline imports model from other pipeline,
            corresponding parameter in import_model must have name 'import_model_from'.
        """
        name = name or 'ppl_' + str(len(self.pipelines))
        config = config or Config()
        variables = variables or []
        if preproc is not None:
            self.has_preproc = True

        if not isinstance(variables, list):
            variables = [variables]
        if name in self.pipelines:
            raise ValueError('Pipeline with name {} was alredy existed'.format(name))
        self.pipelines[name] = {'ppl': pipeline, 'cfg': config, 'var': variables,
                                'preproc': preproc, 'import_model_from': import_model_from}

    def add_grid_config(self, grid_config):
        """ Add grid of pipeline parameters.

        Parameters
        ----------
        grid_config : dict or Grid
            if dict it should have items parameter_name: list of values.
        """
        self.grid_config = Grid(grid_config)

    def _create_tasks(self, n_reps, n_iters, model_per_preproc, name):
        if isinstance(model_per_preproc, int):
            n_models = model_per_preproc
        elif model_per_preproc is None:
            n_models = 1
        else:
            n_models = len(model_per_preproc)
        self.tasks = (
            {'pipelines': self.pipelines,
             'n_iters': n_iters,
             'configs': configs,
             'repetition': idx,
             'model_per_preproc': model_per_preproc,
             'name': name
            }
            for configs in self.grid_config.gen_configs(n_models)
            for idx in range(n_reps)
        )
        self.tasks = Tasks(self.tasks)

    def run(self, n_reps, n_iters, n_jobs=1, model_per_preproc=1, name=None):
        """ Run research.

        Parameters
        ----------
        n_reps : int
            number of repetitions with each combination of parameters
        n_iters: int
            number of iterations for each configurations of each pipeline.
        n_jobs : int (default 1) or list
            If int - number of workers to run pipelines or workers that will run them. By default,
            PipelineWorker will be used. If list - instances of Worker class.
        model_per_preproc: int or list of dicts
            If int - number of pipelines with different configs that will use the same prepared batch
            from preproc. If model_per_preproc - list of dicts with additional configs to each pipeline.
            For example, if there are 2 GPUs, we can define parameter devicse in model config as C('devise')
            and define model_per_preproc as [{'device': 0}, {'device': 1}].
        name : str or None
            name folder to save research. By default is 'research'.
        
        At each iteration all add pipelines will be runned with some config from grid.
        """
        self.n_reps = n_reps
        self.n_iters = n_iters
        self.n_jobs = n_jobs
        self.model_per_preproc = model_per_preproc

        self.name = self._does_exist(name)

        # self.save()
        self._create_tasks(n_reps, n_iters, model_per_preproc, self.name)
        if isinstance(n_jobs, int):
            worker = PipelineWorker
        else:
            worker = None
        distr = Distributor(worker)
        distr.run(self.tasks, dirname=self.name)

    def _does_exist(self, name):
        name = name or 'research'
        if not os.path.exists(name):
            dirname = name
        else:
            i = 1
            while os.path.exists(name + '_' + str(i)):
                i += 1
            dirname = name + '_' + str(i)
        os.makedirs(dirname)
        return dirname

    def save(self):
        """ Save description of the research to name/description. """
        with open(os.path.join(self.name, 'description'), 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, name):
        """ Load description of the research from name/description. """
        with open(os.path.join(name, 'description'), 'rb') as file:
            return pickle.load(file)
