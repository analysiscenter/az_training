import os
import sys
import multiprocessing as mp
import numpy as np
import pickle
import logging

class Tasks:
    def __init__(self, tasks):
        self.tasks = tasks

    def __iter__(self):
        return self.tasks

class Worker:
    def __init__(self, dirname=None, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.dirname = dirname or ''

        logfile = os.path.join(self.dirname, 'errors.log')
        logging.basicConfig(filename=logfile, level=logging.ERROR)

    def init(self, task):
        _ = task

    def post(self, task):
        _ = task

    def task(self, task):
        _ = task

    def __call__(self, queue):
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

        task = queue.get()
        try:
            self.init(task)
            self.task(task, *self.args, **self.kwargs)
            self.post(task)
        except:
            self._log()
        queue.task_done()

    def _log(self):
        logging.error(str(sys.exc_info()))

class Distributor:
    def __init__(self, n_workers, reuse_batch, worker_class, *args, **kwargs):
        self.n_workers = n_workers
        self.worker_class = worker_class

    def _tasks_to_queue(self, tasks):
        queue = mp.JoinableQueue()
        for idx, task in enumerate(tasks):
            queue.put((idx, task))
        for i in range(self.n_workers):
            queue.put(None)
        return queue        

    def run(self, tasks, *args, **kwargs):
        queue = self._tasks_to_queue(tasks)
        for i in range(self.n_workers):
            worker = self.worker_class(*args, **kwargs)
            mp.Process(target=worker, args=(queue, )).start()

        queue.join()
    