#pylint:disable=too-few-public-methods
#pylint:disable=bare-except
#pylint:disable=too-many-function-args

""" Classes for multiprocess task running. """

import os
import sys
import multiprocess as mp
import logging

class Tasks:
    """ Tasks to workers. """
    def __init__(self, tasks):
        self.tasks = tasks

    def __iter__(self):
        return self.tasks

class Worker:
    """ Worker that creates subprocess to execute task"""
    def __init__(self, worker_name=None, logfile=None, errorfile=None, *args, **kwargs):
        """
        Parameters
        ----------
        worker_name : str or int

        args, kwargs
            will be used in init, post and task
        """
        self.task = None
        self.args = args
        self.kwargs = kwargs
        if isinstance(worker_name, int):
            self.name  = "Worker " + str(worker_name)
        elif worker_name is None:
            self.name = 'Worker'
        self.logfile = logfile or 'research.log'
        self.errorfile = errorfile or 'errors.log'

    def set_args_kwargs(self, *args, **kwargs):
        """
        Parameters
        ----------
        args, kwargs
            will be used in init, post and task
        """
        self.args = args
        self.kwargs = kwargs

    def init(self):
        """ Run before task. """
        pass

    def post(self):
        """ Run after task. """
        pass

    def run_task(self):
        """ Main part of the worker. """
        pass
        

    def __call__(self, queue):
        """ Run worker.

        Parameters
        ----------
        queue : multiprocessing.Queue
            queue of tasks for worker
        """
        self.log_info('Start ' + self.name, filename=self.logfile)

        try:
            item = queue.get()
        except Exception as e:
            self.log_error(e, filename=self.errorfile)
        else:
            while item is not None:
                sub_queue = mp.JoinableQueue()
                sub_queue.put(item)
                try:
                    self.log_info(self.name + ' creates process', filename=self.logfile)
                    worker = mp.Process(target=self._run, args=(sub_queue, ))
                    worker.start()
                    sub_queue.join()
                except Exception as e:
                    self.log_error(e, filename=self.errorfile)
                queue.task_done()
                item = queue.get()
        queue.task_done()

    def _run(self, queue):
        try:
            self.task = queue.get()
            self.log_info('Task {} started by {}'.format(self.task[0], self.name), filename=self.logfile)
            self.log_info('Task {} is {}'.format(self.task[0], self.task[1]['configs']), filename=self.logfile)
            self.init()
            self.run_task()
            self.post()
        except Exception as e:
            self.log_error(e, filename=self.errorfile)
        self.log_info('Task {} finished by {}'.format(self.task[0], self.name), filename=self.logfile)
        queue.task_done()

    @classmethod
    def log_info(cls, *args, **kwargs):
        pass

    @classmethod
    def log_error(cls, *args, **kwargs):
        pass

class Distributor:
    """ Distributor of tasks between workers. """
    def __init__(self, n_workers, worker_class=None):
        """
        Parameters
        ----------
        n_workers : int or list of Worker instances

        worker_class : Worker subclass or None
        """
        if isinstance(n_workers, int) and worker_class is None:
            raise ValueError('If worker_class is None, n_workers must be list of Worker instances.')
        self.n_workers = n_workers
        self.worker_class = worker_class

    def _tasks_to_queue(self, tasks):
        queue = mp.JoinableQueue()
        for idx, task in enumerate(tasks):
            queue.put((idx, task))
        for _ in range(self.n_workers):
            queue.put(None)
        return queue

    @classmethod
    def log_info(cls, message, filename):
        logging.basicConfig(filename=filename, level=logging.INFO)
        logging.info(message)

    @classmethod
    def log_error(cls, obj, filename):
        logging.basicConfig(filename=filename, level=logging.INFO)
        logging.error(obj, exc_info=True)

    def run(self, tasks, dirname=None, logfile=None, errorfile=None, *args, **kwargs):

        """ Run disributor and workers.

        Parameters
        ----------
        tasks : iterable

        args, kwargs
            will be used in worker
        """
        self.logfile = logfile or 'research1.log'
        self.errorfile = errorfile or 'errors1.log'

        self.logfile = os.path.join(dirname, self.logfile)
        self.errorfile = os.path.join(dirname, self.errorfile)

        kwargs['logfile'] = self.logfile
        kwargs['errorfile'] = self.errorfile

        self.log_info('Prepare workers', filename=self.logfile)

        if isinstance(self.n_workers, int):
            workers = [self.worker_class(worker_name=i, *args, **kwargs) for i in range(self.n_workers)]
        else:
            for worker in self.n_workers:
                worker.set_args_kwargs(args, kwargs)
            workers = self.n_workers
            self.n_workers = len(self.n_workers)
        try:
            self.log_info('Create tasks queue', filename=self.logfile)
            queue = self._tasks_to_queue(tasks)
        except Exception as e:
            logging.error(e, exc_info=True)
        else:
            self.log_info('Run workers', filename=self.logfile)
            for worker in workers:
                worker.log_info = self.log_info
                worker.log_error = self.log_error

                try:
                    mp.Process(target=worker, args=(queue, )).start()
                except Exception as e:
                    logging.error(e, exc_info=True)
                    while not queue.empty():
                        q.get(item)
                        q.task_done()
            queue.join()
    