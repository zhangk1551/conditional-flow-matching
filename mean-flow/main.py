#!/usr/bin/env python
# coding: utf-8

import os
import random
import argparse
import logging
from omegaconf import OmegaConf

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from mean_flow.datasets import MeanFlowDataModule
from mean_flow.pl_modules import MeanFlowModule
from mean_flow.configs import MainConfig


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


parser = argparse.ArgumentParser()
parser.add_argument('--sweep_id', type=str, default=None)
parser.add_argument('--wandb_mode', type=str, default="offline")
parser.add_argument('--conf', type=str,
                    default="conf/gaussians_moons.yml")


PROJECT = "mean-flow"

def main(config=None):
    wandb_logger = WandbLogger(project=PROJECT, entity="zhangk15", log_model=True)

    seed = config.seed
    if seed is None:
        seed = random.randint(0, 100000)
    pl.seed_everything(seed, workers=True)

    monitor = config.trainer.monitor
    monitor_mode = config.trainer.monitor_mode

    data_module = MeanFlowDataModule(config.data)
    data_module.prepare_datasets()

    checkpoint_callback = ModelCheckpoint(monitor=monitor, mode=monitor_mode, save_top_k=2,
                                          filename="{epoch:03d}-{train_loss:.2f}")

    trainer = pl.Trainer(max_epochs=config.trainer.max_epochs,
                         fast_dev_run=config.trainer.fast_dev_run,
                         overfit_batches=config.trainer.overfit_batches,
                         accelerator='gpu',
                         devices=config.trainer.devices,
                         num_nodes=config.trainer.num_nodes,
                         logger=wandb_logger,
                         log_every_n_steps=5,
                         gradient_clip_val=0.5,
                         callbacks=[checkpoint_callback] if config.trainer.overfit_batches == 0
                         else None)

    mean_flow_module = MeanFlowModule(config)

    trainer.fit(mean_flow_module, datamodule=data_module)


def merge_sweep_config(config, sweep_config):
    parameters = dict(sweep_config).keys()
    groups = dict(config).keys()
    for parameter in parameters:
        for group in groups:
            if config[group] is None:
                continue
            if parameter in config[group]:
                config[group][parameter] = sweep_config[parameter]
    return config


def cli_main():
    default_config = OmegaConf.structured(MainConfig)

    if args.conf is not None:
        yaml_config = OmegaConf.load(args.conf)
        config = OmegaConf.merge(default_config, yaml_config)
    else:
        config = default_config

    if args.sweep_id is not None:
        wandb.init(project=PROJECT, entity="zhangk15")
        sweep_config = wandb.config
        config = merge_sweep_config(config, sweep_config)

    logger.info("config")
    logger.info(config)
    main(config)


if __name__ == "__main__":
    args = parser.parse_args()

    os.environ["WANDB_MODE"] = args.wandb_mode

    if args.sweep_id is not None:
        wandb.agent(args.sweep_id, function=cli_main, project=PROJECT, count=1)
    else:
        cli_main()
