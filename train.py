import utils.config
import utils.logger
import utils.checkpoints

from dataset.BraTS import BraTS, transform
from models.TransBTS import TransBTS
from utils.checkpoints import CheckpointIO
from losses import BinaryCrossEntropy
from metrics import get_accuracy

import os
import time
import torch
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


logging.basicConfig(format="%(levelname)s - %(message)s", level=logging.INFO)


def main(args):
    device_ids = args.device_ids
    # for nn.DataParallel, the model and original input must be on device_ids[0] device
    device = "cuda:{}".format(device_ids[0])

    exp_dir = args.training.exp_dir
    logging.info("=> Experiments dir: {}".format(exp_dir))

    # logger
    logger = utils.logger.Logger(
        log_dir=exp_dir,
        monitoring='tensorboard',
        monitoring_dir=os.path.join(exp_dir, 'events'))

    # save configs
    utils.config.save_config(args.to_dict(), os.path.join(exp_dir, 'config.yaml'))
    
    # checkpoints
    checkpoint_io = CheckpointIO(checkpoint_dir=os.path.join(exp_dir, 'ckpts'))

    # split data indices
    dataset = BraTS(args.data.data_dir, args.data.label, mode='train')
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(args.training.val_split * dataset_size))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Create data loaders
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=args.training.batch_size,
                              sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=args.training.batch_size,
                            sampler=valid_sampler)

    # load model
    model = TransBTS(_conv_repr=args.model.conv_repr, _pe_type="learned")
    model.to(device)

    optimizer = torch.optim.Adam(
        params=model.parameters(), lr = args.training.lr_param, 
        weight_decay = 1e-5, amsgrad = True
    )

    # Register modules to checkpoint
    checkpoint_io.register_modules(
        model=model,
        optimizer=optimizer
    )

    # Load checkpoints
    load_dict = checkpoint_io.load_file(
        args.training.ckpt_file,
        ignore_keys=args.training.ckpt_ignore_keys,
        only_use_keys=args.training.ckpt_only_use_keys)
    it = load_dict.get('global_step', -1)
    epoch_idx = load_dict.get('epoch_idx', -1) + 1

    # start training
    num_ep = args.training.num_ep
    with tqdm(range(num_ep)) as pbar:
        pbar.update(epoch_idx)
        
        while epoch_idx < num_ep:
            pbar.update()
            
            # Learning rate scheduler
            adjust_learning_rate(optimizer, epoch_idx, num_ep, args.training.lr_param)

            for case, target in train_loader:
                it += 1
                pbar.set_postfix(it=it, ep=epoch_idx)
                
                x = transform(case).to(device)
                target = target.to(device)
                
                predict = model(x)
                loss = BinaryCrossEntropy(predict, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                logger.add('learning_rates','model', optimizer.param_groups[0]['lr'], it=it)
                logger.add('losses', 'training', loss.data.cpu().numpy().item(), it)

            #Save checkpoint
            if (epoch_idx + 1) % args.training.epoch_save == 0:
                checkpoint_io.save(
                    filename='{0:05d}.pt'.format(epoch_idx),
                    global_step=it, epoch_idx=epoch_idx)

            # Validation
            targets = torch.empty([0, 1]).to(device)
            predicts = torch.empty([0, 1]).to(device)
            
            for case, target in val_loader:
                with torch.no_grad():
                    x = case.to(device)
                    target = target.to(device)
                    
                    targets = torch.cat((targets, target), 0)                               
                    predict = model(x)
                    predicts = torch.cat((predicts, predict), 0)
                
            acc = get_accuracy(predicts, targets)
            logger.add('accuracy', 'val', acc, it)
            logging.info('Epoch {0}: validation accuracy {1}'.format(epoch_idx, acc))

            epoch_idx += 1

    logging.info('Training completed')        


def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1-(epoch) / max_epoch, power), 8)


if __name__ == "__main__":
    args, unknown = utils.config.parse_args()
    config = utils.config.load_config(args, unknown)
    main(config)
