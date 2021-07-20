import argparse, glob, os
import logging, pickle, bz2

import numpy as np
import skimage.transform as skTrans
import matplotlib.pyplot as plt


# Config logging
logging.basicConfig(format="%(levelname)s - %(message)s", level=logging.INFO)


# Add parser
parser = argparse.ArgumentParser()

parser.add_argument('--img_dir', default='/content/drive/MyDrive/Data/test', type=str)

parser.add_argument('--out_dir', default='/content/drive/MyDrive/BrainTumorClassification/test', type=str)

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
def convert_to_pkl(img_dir: str, output_dir: str, img_shape: tuple, output_shape: tuple) -> None:
    assert os.path.exists(img_dir), "{} does not exist.".format(img_dir)
    C, H, W = output_shape

    try:
        assert os.path.exists(output_dir)
    except AssertionError as e:
        output_dir = img_dir
        logging.error("{0} does not exist, save output to {1}".format(output_dir, img_dir))
    
    assert all(i > 0 for i in output_shape), "Output shape must have positive values."

    cases = get_filenames(img_dir)
    assert len(cases) > 0, "{} is empty.".format(img_dir)

    n = len(cases)
    for it in range(n):
        case_dir = cases[it]
        images = load_case(case_dir, img_shape, output_shape)
        assert images.shape == (4, C, H, W),\
            "Case {0} has incorrect shape {1}, which is different from {2}"\
            .format(case_dir, images.shape, (4, C, H, W))

        save_case(images, case_dir, output_dir)
        logging.info("[{1}/{2}] Saved {0}".format(output_dir, it + 1, n))

    logging.info('Preprocessing completed.')


def get_filenames(path: str) -> list:
    files = glob.glob(os.path.join(path, '*'))
    return sorted(files)


def load_case(case_dir: str, img_shape: tuple, output_shape: tuple) -> np.ndarray:
    modals = ['FLAIR', 'T1w', 'T1wCE', 'T2w']
    C, H, W = output_shape

    images = np.array([], dtype=np.uint8).reshape(0, C, H, W)

    for modal in modals:
        modal_path = os.path.join(case_dir, modal)
        modal_img = load_modal(modal_path, img_shape, output_shape)
        assert modal_img.shape == (1, C, H, W),\
            "Modal {0} has incorrect shape {1}, which is different from {2}"\
            .format(modal_path, modal_img.shape, (1, C, H, W))
        images = np.concatenate((images, modal_img), axis = 0)

    return images


def save_case(images: np.ndarray, case_dir: str, output_dir: str) -> None:
    output_name = extract_folder_name(case_dir) + '.pkl.bz2'
    output_path = os.path.join(output_dir, output_name)
    pickle.dump(images, bz2.open(output_path, 'wb'))


def load_modal(path: str, img_shape: tuple, output_shape: tuple) -> np.ndarray:
    C, H, W = img_shape

    if not os.path.exists(path) or len(get_filenames(path)) == 0:
        logging.info('{} does not contain images, paddad by zeros'.format(path))
        return np.zeros(output_shape, dtype=np.uint8)[None, :]

    images = load_images(path, img_shape)

    assert len(images.shape) == 3,\
        "Images {} must have shape of length 3".format(path)

    images = skTrans.resize(images, output_shape, order=1, preserve_range=True)
    logging.info('Loaded {}'.format(path))
    return images[None, :]


def load_images(path: str, img_shape: tuple) -> np.ndarray:
    C, H, W = img_shape

    imgs = np.array([], dtype=np.uint8).reshape(0, H, W)
    img_paths = glob.glob(os.path.join(path, '*.png'))
    img_paths = sorted(img_paths, key = lambda x: get_index(x))

    for img_path in img_paths:
        img = plt.imread(img_path)

        assert (W, H) == img.shape, \
            "{0} has size {1}, which is different from {2}".format(img_path, img.shape, (W, H))

        img = transform(img)
        imgs = np.concatenate((imgs, img), axis = 0)
        logging.debug('Loaded {0} - Current shape {1}'.format(img_path, imgs.shape))

    return imgs


def extract_folder_name(path: str)-> str:
    return path[(path.rfind('/') + 1):]


def get_index(path):
    start = path.rfind('-')
    end = path.rfind('.png')
    return int(path[start+1: end])


def transform(img: np.ndarray) -> np.ndarray:
    output = (img * 255).astype(np.uint8)
    return output[None, :]


if __name__ == "__main__":
    convert_to_pkl(args.img_dir, args.out_dir, args.img_shape, args.out_shape)