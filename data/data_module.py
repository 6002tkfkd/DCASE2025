import librosa
import numpy as np
import torch
import lightning as L
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from util import unique_labels


class AudioDataset(Dataset):
    def __init__(self, meta_dir: str, audio_dir: str, subset: str, sampling_rate: int = 16000):
        self.meta_dir = meta_dir
        self.audio_dir = audio_dir
        self.subset = subset
        self.sr = sampling_rate

        # csv 파일을 공백으로 구분해서 읽기
        self.meta_subset = pd.read_csv(f"{self.meta_dir}/{self.subset}.csv", delim_whitespace=True)

        # scene_label이 존재할 때만 strip 처리
        if 'scene_label' in self.meta_subset.columns:
            self.meta_subset['scene_label'] = self.meta_subset['scene_label'].str.strip()

    def __len__(self):
        return len(self.meta_subset)

    def __getitem__(self, i):
        row_i = self.meta_subset.iloc[i]
        filename = row_i["filename"]
        wav, _ = librosa.load(f"{self.audio_dir}/{filename}", sr=self.sr)
        wav = torch.from_numpy(wav)
        return wav, filename



class AudioLabelsDataset(AudioDataset):
    def __init__(self, meta_dir: str, audio_dir: str, subset: str, sampling_rate: int = 16000):
        super().__init__(meta_dir, audio_dir, subset, sampling_rate)

    def __getitem__(self, i):
        wav, filename = super().__getitem__(i)
        scene_label = filename.split('/')[-1].split('-')[0]
        device_label = filename.split('-')[-1].split('.')[0]
        city_label = filename.split('-')[1]
        scene_label = unique_labels['scene'].index(scene_label)
        scene_label = torch.from_numpy(np.array(scene_label, dtype=np.int64))
        device_label = unique_labels['device'].index(device_label)
        device_label = torch.from_numpy(np.array(device_label, dtype=np.int64))
        city_label = unique_labels['city'].index(city_label)
        city_label = torch.from_numpy(np.array(city_label, dtype=np.int64))
        return wav, scene_label, device_label, city_label, filename


class AudioLabelsDatasetWithLogits(AudioLabelsDataset):
    def __init__(self, logits_files: list, **kwargs):
        super().__init__(**kwargs)
        self.teacher_logits_list = [torch.load(path).float() for path in logits_files]
        self.num_teachers = len(self.teacher_logits_list)

    def __getitem__(self, i):
        wav, scene_label, device_label, city_label, filename = super().__getitem__(i)
        logits_list = [logits[i] for logits in self.teacher_logits_list]
        return wav, scene_label, device_label, city_label, logits_list, filename




class DCASEDataModule(L.LightningDataModule):
    def __init__(self, meta_dir: str, audio_dir: str, batch_size: int = 16, num_workers: int = 0, pin_memory: bool=False,
                 logits_files=None, train_subset="train", test_subset="test", predict_subset="test", **kwargs):
        super().__init__()
        self.meta_dir = meta_dir
        self.audio_dir = audio_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_subset = train_subset
        self.test_subset = test_subset
        self.predict_subset = predict_subset
        self.logits_files = logits_files
        self.kwargs = kwargs

    def setup(self, stage: str):
        if stage == "fit":
            if self.logits_files is not None:
                self.train_set = AudioLabelsDatasetWithLogits(logits_files=self.logits_files, meta_dir=self.meta_dir, audio_dir=self.audio_dir, subset=self.train_subset, **self.kwargs)
            else:
                self.train_set = AudioLabelsDataset(self.meta_dir, self.audio_dir, subset=self.train_subset, **self.kwargs)
            self.valid_set = AudioLabelsDataset(self.meta_dir, self.audio_dir, subset="valid", **self.kwargs)

        elif stage == "validate":
            self.valid_set = AudioLabelsDataset(self.meta_dir, self.audio_dir, subset="valid", **self.kwargs)

        elif stage == "test":
            self.test_set = AudioLabelsDataset(self.meta_dir, self.audio_dir, subset=self.test_subset, **self.kwargs)

        elif stage == "predict":
            self.predict_set = AudioDataset(self.meta_dir, self.audio_dir, subset=self.predict_subset, **self.kwargs)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                          pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.valid_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          pin_memory=self.pin_memory)

    def predict_dataloader(self):
        return DataLoader(self.predict_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          pin_memory=self.pin_memory)