#pylint:disable=too-few-public-methods
#pylint:disable=bare-except
#pylint:disable=too-many-function-args

""" Classes for multiprocess task running. """

import os
import sys
import multiprocessing as mp
import logging

class Tasks:
    """ Tasks to workers. """
    def __init__(self, tasks):
        self.tasks = tasks

    def __iter__(self):
        return self.tasks

class Worker:
    """ Worker that creates subprocess to execute task"""
    def __init__(self, dirname=None, *args, **kwargs):
        """
        Parameters
        ----------
        dirname : str or None
            folder name to save log file
        args, kwargs
            will be used in init, post and task
        """
        self.task = None
        self.args = args
        self.kwargs = kwargs
        self.dirname = dirname or ''

        logfile = os.path.join(self.dirname, 'errors.log')
        logging.basicConfig(filename=logfile, level=logging.ERROR)

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
        logfile = os.path.join(self.dirname, 'errors.log')
        logging.basicConfig(filename=logfile, level=logging.ERROR)

        item = queue.get()
        while item is not None:
            sub_queue = mp.JoinableQueue()
            sub_queue.put(item)
            try:
                worker = mp.Process(target=self._run, args=(sub_queue, ))
                worker.start()
                sub_queue.join()
            except:
                self._log()
            queue.task_done()
            item = queue.get()
        queue.task_done()

    def _run(self, queue):
        logfile = os.path.join(self.dirname, 'errors.log')
        logging.basicConfig(filename=logfile, level=logging.ERROR)

        self.task = queue.get()
        try:
            self.init()
            self.run_task()
            self.post()
        except:
            self._log()
        queue.task_done()

    def _log(self):
        exc_type, _, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logging.error(exc_type, fname, exc_tb.tb_lineno)

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

    def run(self, tasks, *args, **kwargs):
        """ Run disributor and workers.

        Parameters
        ----------
        tasks : iterable

        args, kwargs
            will be used in worker
        """
        if isinstance(self.n_workers, int):
            workers = [self.worker_class(*args, **kwargs) for _ in range(self.n_workers)]
        else:
            for worker in self.n_workers:
                worker.set_args_kwargs(args, kwargs)
            workers = self.n_workers
            self.n_workers = len(self.n_workers)
        queue = self._tasks_to_queue(tasks)
        for worker in workers:
            mp.Process(target=worker, args=(queue, )).start()

        queue.join()
    