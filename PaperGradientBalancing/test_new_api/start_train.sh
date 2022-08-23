#!/bin/bash
numRepetition=2
port=6969
pathConfigs=writePathHereToConfigs

if [ -d "$pathConfigs" ]
then
  for singleConfig in "$pathConfigs"/*
  do
    for ((i = 0; i < numRepetition; i++))
    do
      echo "Started training with config: $singleConfig, number: $i"

      python run.py -c "$singleConfig" -p $port
    done
  done
else
  echo "Folder $pathConfigs were not found!"
  echo "Training will be started with default config file, i.e named as config.json"
  python run.py -p $port
fi
