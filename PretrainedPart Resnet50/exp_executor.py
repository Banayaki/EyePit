from data_provider import get_data
from makiflow.trainers import SegmentatorTrainer
import makiflow as mf
import os


if __name__ == "__main__":
    mf.set_main_gpu(2)
    Xtrain, Ytrain, num_pos, Xtest, Ytest = get_data()
    os.makedirs('experiments', exist_ok=True)
    trainer = SegmentatorTrainer('exp_params.json', 'experiments')
    trainer.set_test_data(Xtest, Ytest)
    trainer.set_train_data(Xtrain, Ytrain, num_pos)
    trainer.start_experiments()

