import os
import glob
import yaml
import torch
import addict
import logging
import argparse


logging.basicConfig(format="%(levelname)s - %(message)s", level=logging.INFO)


class ForceKeyErrorDict(addict.Dict):
    def __missing__(self, name):
        raise KeyError(name)


def parse_args():
    parser = argparse.ArgumentParser()
    
    # must-have configs
    parser.add_argument('--config', type=str, default=None, help='Path to config file.')
    parser.add_argument('--load_exp', type=str, default=None, help='Directory of experiment to load.')

    args, unknown = parser.parse_known_args()
    return args, unknown


def save_config(datadict, path):
    datadict['training']['ckpt_file'] = None
    datadict['training'].pop('exp_dir')
    
    with open(path, 'w', encoding='utf8') as f:
        yaml.dump(datadict, f, default_flow_style=False)


def load_config(args, unknown):
    ''' overwrite seq
    cmd param >>> .yaml param
    '''
    assert (args.config is not None) != (args.load_exp is not None),\
           "you must specify ONLY one in 'config' or 'load_exp' "

    if args.load_exp is not None:
        assert '--expname' not in unknown,\
               "given --expname with --load_exp will lead to unexpected behavior."

        config_path = os.path.join(args.load_exp, 'config.yaml')
        config = load_yaml(config_path, default_path=None)
        
        config = update_config(config, unknown)

        config.training.exp_dir = args.load_exp
        logging.ìnfo("=> Loading previous experiments in: {}".format(config.training.exp_dir))
        
    else:
        config = load_yaml(args.config)

        config = update_config(config, unknown)

        # use the expname and log_root_dir to get the experiement directory
        config.training.exp_dir = os.path.join(config.training.log_root_dir, config.expname)

    config = set_device_ids(config)
    return config


def load_yaml(path, default_path=None):

    with open(path, encoding='utf8') as yaml_file:
        config_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
        config = ForceKeyErrorDict(**config_dict)

    if default_path is not None and path != default_path:
        with open(default_path, encoding='utf8') as default_yaml_file:
            default_config_dict = yaml.load(
                default_yaml_file, Loader=yaml.FullLoader)
            main_config = ForceKeyErrorDict(**default_config_dict)

        main_config.update(config)
        config = main_config

    return config


def update_config(config, unknown):
    # update config given args
    for idx, arg in enumerate(unknown):
        if arg.startswith("--"):
            if (':') in arg:
                k1, k2 = arg.replace("--", "").split(':')
                argtype = type(config[k1][k2])
                if argtype == bool:
                    v = unknown[idx+1].lower() == 'true'
                else:
                    if config[k1][k2] is not None:
                        v = type(config[k1][k2])(unknown[idx+1])
                    else:
                        v = unknown[idx+1]
                print(f'Changing {k1}:{k2} ---- {config[k1][k2]} to {v}')
                config[k1][k2] = v
            else:
                k = arg.replace('--', '')
                v = unknown[idx+1]
                argtype = type(config[k])
                print(f'Changing {k} ---- {config[k]} to {v}')
                config[k] = v

    return config


def set_device_ids(config):
    # # device_ids: -1 will be parsed as using all available cuda device
    # # device_ids: [] will be parsed as using all available cuda device
    if (type(config.device_ids) == int and config.device_ids == -1) \
            or (type(config.device_ids) == list and len(config.device_ids) == 0):
        config.device_ids = list(range(torch.cuda.device_count()))
    # # e.g. device_ids: 0 will be parsed as device_ids [0]
    elif isinstance(config.device_ids, int):
        config.device_ids = [config.device_ids]
    # # e.g. device_ids: 0,1 will be parsed as device_ids [0,1]
    elif isinstance(config.device_ids, str):
        config.device_ids = [int(m) for m in config.device_ids.split(',')]

    return config


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
