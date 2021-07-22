import pickle, bz2
import os, random
import torch
import pandas as pd
import torchvision.transforms as T


class RandomFlip(object):
    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.5:
            sample = torch.flip(sample, dims=(-3, -1))
        return sample


class RandomNoise(object):
    def __call__(self, image: torch.Tensor, factor=0.1) -> torch.Tensor:
        _, C, H, W = image.shape
        scale_factor = 2 * factor * torch.rand(1, C, H, W) + 1.0 - factor
        shift_factor = 2 * factor * torch.rand(1, C, H, W) - factor
        image = image * scale_factor + shift_factor
        return image


def transform(self, sample: torch.Tensor) -> torch.Tensor:
    trans = T.Compose([
        RandomFlip(),
        RandomNoise()
    ])
    return trans(sample)


class BraTS(Dataset):
    def __init__(self, data_dir: str, label_file: str = None, mode: str = 'train'):
        assert os.path.exists(data_dir), 'Data directory does not exist.'
        self.data_dir = data_dir
        self.mode = mode

        if self.mode == 'train':
            assert os.path.exists(label_file) && label_file.endswith('csv'),\
                'Label file does not exist or not have .CSV extension.' 
            self.files = pd.read_csv(label_file).tolist()

        else:
            self.files = self.__get_filenames(data_dir)
    

    def __load_pkl(self, fname: str) -> torch.Tensor:
        with bz2.open(fname, 'rb') as f:
            case = pickle.load(f)
        return torch.from_numpy(case / 255.0).float()


    def __get_filenames(self, path: str) -> list:
        files = glob.glob(os.path.join(path, '*.pkl.bz2'))
        return sorted(files)


    def __getitem__(self, idx):
        if self.mode == 'train':
            data = self.files[idx]
            path = os.path.join(self.data_dir, '{0:05d}.pkl.bz2'.format(data[0]))
            label = torch.tensor([[data[1]]])
            case = self.__load_pkl(path)
            return case, label
        
        else:
            path = self.files[idx]
            case = self.__load_pkl(path)
            return case


    def __len__(self):
        return len(self.files)
