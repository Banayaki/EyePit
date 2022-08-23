import multiprocessing as mp


class Tensorboard:
    @staticmethod
    def run_tb(logdir, port):
        import os
        os.system(f'tensorboard --logdir={logdir} --port={port}')

    def __init__(self, logdir, port):
        # Does not inherit resources of the parent process. Create a fresh
        # python interpreter process
        mp.set_start_method('spawn')
        self._process = mp.Process(target=Tensorboard.run_tb, args=(logdir, port))

    def start(self):
        self._process.start()

    def close(self):
        input('TRAINING HAS ENDED.')
        self._process.terminate()
        self._process.join()
        self._process.close()


