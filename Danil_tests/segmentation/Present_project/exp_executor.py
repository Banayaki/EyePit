from data_provider import get_data_cv3
from makiflow.trainers import SegmentatorTrainer

import os
import makiflow as mf

mf.set_main_gpu(1)

import tensorflow as tf

def do_cv_experiment(Xtest, Ytest, experiment_folder, exp_params_file, gen):
    os.makedirs(experiment_folder, exist_ok=True)

    trainer = SegmentatorTrainer(exp_params_file, experiment_folder)
    trainer.set_generator(gen, iterations=313) 
    trainer.set_test_data(Xtest, Ytest)

    trainer.start_experiments()
    del trainer

if __name__ == "__main__":
    Xtest, Ytest, gen_3 = get_data_cv3()
    do_cv_experiment(Xtest, Ytest, 'test3', 'exp_params.json', gen_3)


