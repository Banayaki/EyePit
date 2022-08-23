from data_provider import get_data_cv1, get_data_cv2, get_data_cv3
from makiflow.trainers import SegmentatorTrainer
import os
import makiflow as mf

mf.set_main_gpu(1)

def do_cv_experiment(Xtest, Ytest, experiment_folder, exp_params_file, gen):
    os.makedirs(experiment_folder, exist_ok=True)

    trainer = SegmentatorTrainer(exp_params_file, experiment_folder)
    trainer.set_generator(gen, iterations=100) 
    trainer.set_test_data(Xtest, Ytest)

    trainer.start_experiments()
    del trainer

if __name__ == "__main__":
    Xtest, Ytest, gen_1 = get_data_cv1()
    do_cv_experiment(Xtest, Ytest, 'exp_cv1', 'exp_params.json', gen_1)

    Xtest, Ytest, gen_2 = get_data_cv2()
    do_cv_experiment(Xtest, Ytest, 'exp_cv2', 'exp_params.json', gen_2)

    Xtest, Ytest, gen_3 = get_data_cv3()
    do_cv_experiment(Xtest, Ytest, 'exp_cv3', 'exp_params.json', gen_3)
