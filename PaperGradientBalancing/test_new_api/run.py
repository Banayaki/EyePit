from makiflow.gyms import Gym
import tensorflow as tf
import makiflow as mf
from makiflow.models.classificator.trainers import DiceTrainer
from gen_layer import get_gen_layer, get_gen_layer_v3
from tensorboard import Tensorboard
import argparse
import traceback
mf.set_main_gpu(2)

# Path to the config file
CONFIG_PATH = 'config.json'
DEFAULT_PORT = 51129


def parse_args():
    parser = argparse.ArgumentParser(description='Starts the training process and runs the tensorboard.')
    parser.add_argument('-p', '--port',  type=int,  default=DEFAULT_PORT,
                        help='Port on which the tensorboard will be launched.')
    parser.add_argument('-c', '--config', type=str, default=CONFIG_PATH,
                        help='Path to the config file.',)
    args = parser.parse_args()
    return args.port, args.config


# noinspection PyShadowingNames
def run(port, config_path):
    gym = Gym(
        config_path=config_path,
        gen_layer_fabric=get_gen_layer_v3,
        sess=tf.Session()
    )
    # Run Tensorboard
    tb_path = gym.get_tb_path()
    tb = Tensorboard(tb_path, port)
    tb.start()
    try:
        gym.start_training()
    except Exception as ex:
        print(ex)
        print(traceback.print_exc())
    finally:
        tb.close()


if __name__ == '__main__':
    port, config_path = parse_args()
    run(port, config_path)
