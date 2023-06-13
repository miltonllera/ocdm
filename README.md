# Object-centric disentangled mechanisms

This repository contains code for the final chapter of my PhD thesis. Said chapter deals with the compositional generalisation capabilities of Object Centric models, specifically Slot Attention. Additionally, it can be used to reproduce the results from previous chapters, though there are no scripts provided to do so.

## Setting up

First use the `requirements.txt` to create an environment, either using `conda`, `pyenv` or whatever python enviroment manager you prefer. Appart from the PyTorch libraries (including Torchvision), the repo heavily relies on [Hydra](https://hydra.cc/docs/intro/) to create, compose and run experiment configurations. It also uses [PyTorch Lightning](https://lightning.ai/docs/pytorch/latest/) to define and log model runs.

## Project structure

The code is organized following this nice [template](https://github.com/ashleve/lightning-hydra-template). It foregoes using some of Hydra's features such as Structured Configs and just uses plain `json` files. These are located in the `configs` folder. The main entry points for execution are located in the `bin` folder, including scripts to train and analyse models, and create the Pentomino dataset. All model source codes are included in the `src` folder.

## Running experiments

Each experiment can be run using the `train.py` script by using a commmand with the following structure:

``` bash
python -m bin.train experiment=<name-of-the-experiment> <extra-options>
```

The extra options here can be either parameters already present in the config such as `model.latent.latent_size=12` or new parameters, in which case they must be prepended with a `+`. For example:

``` bash
# change latent size
python -m bin.train experiment=vae_dsprites model.latent.latent_size=16

# debug (use fast dev run)
python -m bin.train experiment=vae_dsprites +debug=fdr
```

All experiment logs will be stored in `data/logs`. Summary information can be accessed using `Tensorboard` if you run the server as:

``` bash
tensorboard --logdir data/logs
```

For remote connctions use the `--bind_all` flag and then connect to the machine IP and port 6006:

```
tensorboard --logdir data/logs --bind_all
```
