import os
import sys
import multiprocessing as mp
import numpy as np
import pickle

class Tasks:
    def __init__(self, tasks):
        self.tasks = tasks

    def __iter__(self):
        return self.tasks

class Worker:
    def __init__(self):
        pass

    def init(self, task):
        _ = task

    def post(self, task):
        _ = task

    def task(self, task):
        _ = task

    def __call__(self, queue):
        item = queue.get()
        while item is not None:
            sub_queue = mp.JoinableQueue()
            sub_queue.put(item)
            try:
                worker = mp.Process(target=self._run, args=(sub_queue, ))
                worker.start()
                sub_queue.join()
            except:
                print(sys.exc_info()[0])
            queue.task_done()
            item = queue.get()
        queue.task_done()

    def _run(self, queue):
        task = queue.get()
        try:
            self.init(task)
            self.task(task)
            self.post(task)
        except:
            print(sys.exc_info())
        queue.task_done()

class Distributor:
    def __init__(self, worker_class, n_workers):
        self.n_workers = n_workers
        self.worker_class = worker_class

    def _tasks_to_queue(self, tasks):
        queue = mp.JoinableQueue()
        for idx, task in enumerate(tasks):
            queue.put((idx, task))
        for i in range(self.n_workers):
            queue.put(None)
        return queue        

    def run(self, tasks):
        queue = self._tasks_to_queue(tasks)
        for i in range(self.n_workers):
            worker = self.worker_class()
            mp.Process(target=worker, args=(queue, )).start()
        queue.join()
    