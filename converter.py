import argparse, glob, os
import logging, pickle, bz2

import numpy as np
import skimage.transform as skTrans
import matplotlib.pyplot as plt

from tqdm import tqdm


# Config logging
logging.basicConfig(format="%(levelname)s - %(message)s", level=logging.INFO)


# Add parser
parser = argparse.ArgumentParser()

parser.add_argument('--img_dir', default='/content/drive/MyDrive/Data/train', type=str)

parser.add_argument('--out_dir', default='/content/drive/MyDrive/BrainTumorClassification/train', type=str)

def shape(s):
    try:
        x, y, z = map(int, s.split(','))
        return (x, y, z)
    except:
        raise argparse.ArgumentTypeError("Shape must have 3 values (C, H, W)")

parser.add_argument('--img_shape', default=(1,512,512), type=shape)

parser.add_argument('--out_shape', default=(64,256,256), type=shape)

args = parser.parse_args()


# Codes start here
def convert_to_pkl(img_dir: str, output_dir: str, img_shape: tuple:, output_shape: tuple) -> None:
    assert os.path.exists(img_dir), "{} does not exist.".format(img_dir)

    try:
        assert os.path.exists(output_dir)
    except AssertionError as e:
        output_dir = img_dir
        logging.error("{0} does not exist, save output to {1}".format(output_dir, img_dir))
    
    assert all(i > 0 for i in output_shape), "Output shape must have positive values."

    cases = get_files(img_dir)
    assert len(cases) > 0, "{} is empty.".format(img_dir)

    for case_dir in tqdm(cases):
        images = load_case(case_dir, img_shape, output_shape)
        save_case(images, case_dir, output_dir)

    logging.info('Preprocessing completed.')


def get_files(path: str) -> list:
    files = glob.glob(os.path.join(path, '*'))
    return sorted(files)


def load_case(case_dir: str, img_shape: tuple, output_shape: tuple) -> np.ndarray:
    modals = ['FLAIR', 'T1w', 'T1wCE', 'T2w']
    C, H, W = output_shape

    images = np.array([], dtype=np.uint8).reshape(0, C, H, W)

    for modal in modals:
        modal_path = os.path.join(case_dir, modals)
        modal_img = load_modal(modal_path, img_shape, output_shape)
        np.concatenate((images, modal_img), axis = 0)

    return images


def save_case(images: np.ndarray, case_dir: str, output_dir: str) -> None:
    output_name = os.path.dirname(case_dir) + '.pkl.bz2'
    output_path = os.path.join(output_dir, out_name)

    logging.debug("Saving to {}".format(output_path))
    pickle.dump(images, bz2.open(output_path, 'wb'))


def load_modal(path: str, img_shape: tuple, output_shape: tuple) -> np.ndarray:
    if not os.path.exists(path) or len(get_files(path)) == 0:
        logging.debug('{} does not contain images.'.format(path))
        return np.zeros(output_shape, dtype=np.uint8)

    images = load_images(path, img_shape)
    skTrans.resize(images, output_shape, order=1, preserve_range=True)

    return images[None, :]


def load_images(path: str, img_shape: tuple) -> np.ndarray:
    C, H, W = img_shape

    imgs = np.array([], dtype=np.uint8).reshape(0, H, W)
    img_paths = glob(os.path.join(path, '*.png'))
    img_paths = sorted(img_paths, key = lambda x: get_index(x))

    for img_path in img_paths:
        img = plt.imread(img_path)
    return imgs


if __name__ == "__main__":
    convert_to_pkl(args.img_dir, args.out_dir, args.img_shape, args.out_shape)