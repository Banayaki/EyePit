## MakiPose-Training

This repository contains a ready-to-use code for training a MakiFlow pose estimation model. It does not contain much code
and is not complicated, so it is easy to modify to suit one's needs.

### Structure

The repository consists of 3 main files:

> gen_layer.py

Contains code that creates generator layer that is gonna feed the model with the data.
It has several constants that you have to set, such as `TFRECORDS_PATH` or `BATCH_SIZE`.

> run.py

The actual script to run.

> config.json

Configuration file that contains all the info the training process: experiment folder, number of epochs, skeleton 
configuration, etc.

> model.json

The model's architecture file. It is advisable to put the model's architecture this way, however, you are not
restricted and can change the path to the architecture file in `config.json`.

### How to use

It is assumed the data has already been prepared.

1. Put the model's architecture file and name it as `model.json`.
2. Set the configuration file to suit your training needs.
3. Run in console `python run.py`.
4. Open tensorboard. If you run the script on the local machine, open in the browser: localhost:6006.
If the script is being ran on a remote machine, you can access the board through ssh. In command line
enter `ssh -p PORT USERNAME@REMOTE_MACHINE_ADRESS -N -f -L localhost:16006:localhost:6006` and then open 
localhost:16006 in the browser.
