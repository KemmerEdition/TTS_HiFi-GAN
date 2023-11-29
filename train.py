import argparse
import collections
import warnings

import numpy as np
import torch
import torch.utils.data
# import hw_4.loss as module_loss
import hw_4.model as module_arch
from hw_4.trainer import Trainer
from hw_4.utils import prepare_device
from hw_4 import datasets as dataset
from hw_4.utils.parse_config import ConfigParser

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)


def main(config):
    logger = config.get_logger("train")

    # setup data_loader instances
    datasets = config.init_obj(config["datasets"], dataset)
    dataloaders = config.init_obj(config["dataloaders"], torch.utils.data, dataset=datasets)

    # build model architecture, then print to console
    model = config.init_obj(config["arch"], module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    # loss_module = config.init_obj(config["loss"], module_loss).to(device)

    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    g_trainable_params = filter(lambda p: p.requires_grad, model.generator.parameters())
    d_p_trainable_params = list(filter(lambda p: p.requires_grad, model.mpd.parameters()))
    d_s_trainable_params = list(filter(lambda p: p.requires_grad, model.msd.parameters()))
    optimizer = {}
    optimizer["optimizer_g"] = config.init_obj(config["optimizer_g"], torch.optim, g_trainable_params)
    optimizer["optimizer_d"] = config.init_obj(config["optimizer_d"], torch.optim,
                                               d_p_trainable_params + d_s_trainable_params)
    lr_scheduler = {}
    lr_scheduler["lr_scheduler_g"] = config.init_obj(config["lr_scheduler_g"], torch.optim.lr_scheduler,
                                                     optimizer['optimizer_g'])
    lr_scheduler["lr_scheduler_d"] = config.init_obj(config["lr_scheduler_d"], torch.optim.lr_scheduler,
                                                     optimizer['optimizer_d'])

    trainer = Trainer(
        model,
        optimizer,
        config=config,
        device=device,
        dataloaders=dataloaders,
        lr_scheduler=lr_scheduler,
        len_epoch=config["trainer"].get("len_epoch", None)
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
