"""Custom batch class for MNIST images
"""
from time import time

from dataset import action, ImagesBatch #pylint: disable=import-error


class ImagesBatchTimeRecorder(ImagesBatch):
    """
    Batch class for measuring the execution time of a pipeline block

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._time_stamps = {}

    @action
    def noop(self):
        """pass"""
        print('I love pylint so much!!', self)

    @action
    def record_time(self, stat_name='statistics', mode='record'):
        """
        Make a time stamp or record the difference between the previous
        one for a specified list in the pipeline

        Parameters
        ----------
        stat_name : str
                    name of the statistics in the pipeline for which the operation is conducted
        mode : {'record', 'diff'}
               if 'record' is given then the current time is recorded with the handler specified by `stat_name`
               if 'diff' is given then the difference between the current time and the last recorded time for
               the given handler (`stat_name`) is appended to `stat_name` in the pipeline

        Returns
        -------
        self : MNISTBatchTime

        """
        if mode == 'record':
            self._time_stamps[stat_name] = time()
        elif mode == 'diff':
            self.pipeline.update_variable(
                stat_name,
                time()-self._time_stamps[stat_name],
                mode='append')
        return self
