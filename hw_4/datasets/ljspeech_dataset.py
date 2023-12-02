import json
import logging
import os
import shutil
import math
import random
from curses.ascii import isascii
from torch.nn.functional import pad
from pathlib import Path
from torch.utils.data import Dataset

import torchaudio
# from hw_4.base.base_dataset import BaseDataset
from hw_4.utils import ROOT_PATH
from hw_4.utils.configs import MelSpectrogramConfig, MelSpectrogram
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm

logger = logging.getLogger(__name__)

URL_LINKS = {
    "dataset": "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2", 
}


class LJspeechDataset(Dataset):
    def __init__(self, part, segment_size, data_dir=None, *args, **kwargs):
        super().__init__()
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "ljspeech"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        self.index = sorted((self._get_or_load_index(part)), key=lambda x: x["audio_len"])
        self.mel_spec_class = MelSpectrogram()
        self.segment_size = segment_size

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        filename = self.index[index]
        file_path = filename["path"]
        audio = self.load_wav(file_path)
        mel = self.mel_spec_class(audio.unsqueeze(0)).squeeze(0)
        frames_per_seg = math.ceil(MelSpectrogramConfig.segment_size / MelSpectrogramConfig.hop_length)
        if audio.size(1) >= MelSpectrogramConfig.segment_size:
            mel_start = random.randint(0, mel.size(1) - frames_per_seg - 1)
            mel_ = mel[:, mel_start:mel_start + frames_per_seg]
            audio_ = audio[:, mel_start * MelSpectrogramConfig.hop_length:(mel_start + frames_per_seg) * MelSpectrogramConfig.hop_length]
        else:
            mel_ = pad(mel, (0, frames_per_seg - mel.size(1)), 'constant')
            audio_ = pad(audio, (0, MelSpectrogramConfig.segment_size - audio.size(1)), 'constant')
        return {"audio": audio_, "spectrogram": mel_}

    def _load_dataset(self):
        arch_path = self._data_dir / "LJSpeech-1.1.tar.bz2"
        print(f"Loading LJSpeech")
        download_file(URL_LINKS["dataset"], arch_path)
        shutil.unpack_archive(arch_path, self._data_dir)
        for fpath in (self._data_dir / "LJSpeech-1.1").iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(self._data_dir / "LJSpeech-1.1"))

        (self._data_dir / "train").mkdir(exist_ok=True, parents=True)

        for i, fpath in enumerate((self._data_dir / "wavs").iterdir()):
            shutil.move(str(fpath), str(self._data_dir / "train" / fpath.name))

    def load_wav(self, full_path):
        data, sr = torchaudio.load(full_path)
        data = data[0:1, :]
        return data

    def _get_or_load_index(self, part):
        index_path = self._data_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part):
        index = []
        split_dir = self._data_dir / part
        if not split_dir.exists():
            self._load_dataset()

        wav_dirs = set()
        for dirpath, dirnames, filenames in os.walk(str(split_dir)):
            if any([f.endswith(".wav") for f in filenames]):
                wav_dirs.add(dirpath)
        for wav_dir in tqdm(
                list(wav_dirs), desc=f"Preparing ljspeech folders: {part}"
        ):
            wav_dir = Path(wav_dir)
            trans_path = list(self._data_dir.glob("*.csv"))[0]
            with trans_path.open() as f:
                for line in f:
                    w_id = line.split('|')[0]
                    w_text = " ".join(line.split('|')[1:]).strip()
                    wav_path = wav_dir / f"{w_id}.wav"
                    t_info = torchaudio.info(str(wav_path))
                    length = t_info.num_frames / t_info.sample_rate

                    index.append(
                        {
                            "path": str(wav_path.absolute().resolve()),
                            "text": w_text.lower(),
                            "audio_len": length,
                        }
                    )
        return index
