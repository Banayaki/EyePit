import sys
sys.path.append('/home/rustam/EyePit/Danil_tests/MakiFlow/')

from generator_provider import get_generator
from makiflow.trainers import SegmentatorTrainer
from data_provider import get_data
import makiflow as mf
import os

mf.set_main_gpu(1)


if __name__ == "__main__":
    generator = get_generator()
    Xtest, Ytest = get_data()
    os.makedirs('experiments', exist_ok=True)
    trainer = SegmentatorTrainer('exp_params.json', 'experiments_xception_40k')
    trainer.set_generator(generator, iterations=1000) 
    trainer.set_test_data(Xtest, Ytest)
    trainer.start_experiments()
