import os
import torch
import pickle
import imageio
import torchvision
import numpy as np


class Logger(object):
    def __init__(self,
                 log_dir,
                 img_dir,
                 monitoring=None,
                 monitoring_dir=None):
        self.stats = dict()
        self.log_dir = log_dir

        utils.cond_mkdir(self.log_dir)

        if not (monitoring is None or monitoring == 'none'):
            self.setup_monitoring(monitoring, monitoring_dir)
        else:
            self.monitoring = None
            self.monitoring_dir = None


    def setup_monitoring(self, monitoring, monitoring_dir):
        self.monitoring = monitoring
        self.monitoring_dir = monitoring_dir
        if monitoring == 'telemetry':
            import telemetry
            self.tm = telemetry.ApplicationTelemetry()
            if self.tm.get_status() == 0:
                print('Telemetry successfully connected.')
        elif monitoring == 'tensorboard':
            import tensorboardX
            self.tb = tensorboardX.SummaryWriter(self.monitoring_dir)
        else:
            raise NotImplementedError('Monitoring tool "%s" not supported!'
                                      % monitoring)


    def add(self, category, k, v, it):
        if category not in self.stats:
            self.stats[category] = {}

        if k not in self.stats[category]:
            self.stats[category][k] = []

        self.stats[category][k].append((it, v))

        k_name = '%s/%s' % (category, k)
        if self.monitoring == 'telemetry':
            self.tm.metric_push_async({
                'metric': k_name, 'value': v, 'it': it
            })
        elif self.monitoring == 'tensorboard':
            self.tb.add_scalar(k_name, v, it)


    def add_vector(self, category, k, vec, it):
        if category not in self.stats:
            self.stats[category] = {}

        if k not in self.stats[category]:
            self.stats[category][k] = []

        if isinstance(vec, torch.Tensor):
            vec = vec.data.clone().cpu().numpy()

        self.stats[category][k].append((it, vec))


    def add_module_param(self, module_name, module, it):
        if self.monitoring == 'tensorboard':
            for name, param in module.named_parameters():
                self.tb.add_histogram("{}/{}".format(module_name, name), param.detach(), it)


    def get_last(self, category, k, default=0.):
        if category not in self.stats:
            return default
        elif k not in self.stats[category]:
            return default
        else:
            return self.stats[category][k][-1][1]


    def save_stats(self, filename):
        filename = os.path.join(self.log_dir, filename)
        with open(filename, 'wb') as f:
            pickle.dump(self.stats, f)


    def load_stats(self, filename):
        filename = os.path.join(self.log_dir, filename)
        if not os.path.exists(filename):
            # print('=> File "%s" does not exist, will create new after calling save_stats()' % filename)
            return

        try:
            with open(filename, 'rb') as f:
                self.stats = pickle.load(f)
                print("=> Load file: {}".format(filename))
        except EOFError:
            print('Warning: log file corrupted!')
