# Neural-Vocoder (HiFi-GAN) project

 This is a repository for Neural-Vocoder project based on one of DLA course (HSE).
## Repository structure

`hw_hifi` - directory included all project files.
* `base` - base classes for model, dataset and train.
* `datasets` - functions and classes for downloading data, audio preprocessing and utils for padding.
* `configs` - configs with params for training.
* `logger` - files for logging.
* `loss` - definition for loss computation (feature, discriminator, generator).
* `model` - architecture for HiFi-GAN (generator, discriminator blocks).
* `test_data` - this folder contains 3 examples of audio files for test and inference during train process.
* `trainer` - train loop, logging in W&B.
* `utils` - configs (dataclasses) with hyperparams of model (both V2 and V1) and other crucial functions (parse_config, object_loading, utils).

## Installation guide

As usual, clone repository, change directory and install requirements:

```shell
!git clone https://github.com/KemmerEdition/hw_hifi.git
!cd /content/hw_hifi
!pip install -r requirements.txt
```
## Train
Train model with command below (if you are not using kaggle for training, you need to open `hw_4/trainer/trainer` and change the path to the test audio inside the file).
   ```shell
   !python -m train -c hw_4/configs/v1_config.json
   ```
If you want to resume training from checkpoint, use command below.
   ```shell
   !python -m train -c hw_4/configs/v1_config.json -r checkpoint-epoch90.pth
   ```
## Test
You only need to run following commands (download checkpoint of my model, run test.py), wait some time and enjoy.
   ```shell
   !gdown --id 1VPOwDbota3_etXSokq6GZll9L4jbKWVu
  ```
   ```shell
!python -m test -c hw_4/configs/v1_config.json -r checkpoint-epoch90.pth 
   ```
Find directory named `results` and run following command, where you'll write path for audio you're interested in (see example):
```shell
from IPython import display
display.Audio('/content/hw_hifi/results/audio_0.wav')
```
## Credits

This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.
