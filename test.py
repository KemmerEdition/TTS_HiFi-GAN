import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm
import torchaudio
import hw_4.model as module_model
from hw_4.trainer import Trainer
from hw_4.utils import ROOT_PATH
from hw_4 import datasets as dataset
from hw_4.utils.parse_config import ConfigParser
from hw_4.utils.configs import MelSpectrogram

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def main(config, out_file):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build model architecture
    model = config.init_obj(config["arch"], module_model)
    # logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()
    os.makedirs("results", exist_ok=True)
    melspeca = MelSpectrogram()
    test_folder = ['/content/hw_hifi/hw_4/test_data/audio_1.wav',
                        '/content/hw_hifi/hw_4/test_data/audio_2.wav',
                        '/content/hw_hifi/hw_4/test_data/audio_3.wav']
    test_wavs = [torchaudio.load(path)[0] for path in test_folder]
    test_mels = [melspeca(wav) for wav in test_wavs]

    with torch.no_grad():
        for i, mel in tqdm(enumerate(test_mels)):
            path = ROOT_PATH / "results" / f"audio_{i}.wav"
            gen_wav = model(mel.to(device)).squeeze(0).cpu()
            torchaudio.save(str(path), gen_wav, sample_rate=22050)


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
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
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
    args.add_argument(
        "-o",
        "--output",
        default="output.json",
        type=str,
        help="File to write results (.json)",
    )
    args.add_argument(
        "-t",
        "--test-data-folder",
        default=None,
        type=str,
        help="Path to dataset",
    )
    args.add_argument(
        "-b",
        "--batch-size",
        default=20,
        type=int,
        help="Test dataset batch size",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    main(config, args.output)
