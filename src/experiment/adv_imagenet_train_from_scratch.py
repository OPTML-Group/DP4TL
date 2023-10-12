import argparse
import json
import os
import sys
from datetime import datetime
from time import time

import numpy as np
import torch
import torchmetrics
from fastargs import get_current_config, Param, Section
from fastargs.decorators import param, section
from fastargs.validation import OneOf, And, File, Folder
from torch.autograd import Variable
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from tqdm import tqdm

sys.path.append(".")
from src.tools.training import apply_blurpool, MeanScalarMetric
from src.tools.misc import set_seed
from src.data.imagenet import get_train_loader, get_val_loader

Section('exp').params(
    identifier=Param(str, 'experiment identifier', default=None)
)

Section('network').params(
    architecture=Param(OneOf(['resnet18', 'resnet50', 'resnet101', 'resnet152']), required=True),
    blurpool=Param(And(int, OneOf([0, 1])), 'use blurpool? (0/1)', default=0),
)

Section('dataset').params(
    train_path=Param(File(), required=True),
    val_path=Param(File(), required=True),
    num_workers=Param(int, 'the number of workers', default=12),
    in_memory=Param(And(int, OneOf([0, 1])), 'does the dataset fit in memory? (0/1)', default=0),
    prune=Param(And(int, OneOf([0, 1])), 'is the dataset pruned? (0/1)', default=0),
)

Section('dataset.indices').enable_if(
    lambda cfg: cfg['dataset.prune']
).params(
    training=Param(File(), required=True),
    testing=Param(File(), required=False),
)

Section('adv').params(
    attack_step=Param(int, default=3),
    epsilon=Param(float, default=3.0),
)

Section('train').params(
    seed=Param(int, required=True),
    epoch=Param(int, required=True),
    batch_size=Param(int, required=True),
    label_smoothing=Param(float, 'label smoothing parameter', default=0.0),
    scheduler_type=Param(OneOf(['step', 'cyclic', 'cosine']), required=True),
)

Section('train.optimizer').params(
    lr=Param(float, required=True),
    weight_decay=Param(float, required=True),
    momentum=Param(float, required=True),
)

Section('train.scheduler.step').enable_if(
    lambda cfg: cfg['train.scheduler_type'] == 'step'
).params(
    step_ratio=Param(float, 'learning rate step ratio', default=0.1),
    step_size=Param(int, 'learning rate step size', default=30),
)

Section('train.scheduler.cyclic').enable_if(
    lambda cfg: cfg['train.scheduler_type'] == 'cyclic'
).params(
    lr_peak_epoch=Param(int, 'epoch at which lr peaks', default=2),
)

Section('train.resolution', 'resolution scheduling').params(
    min_res=Param(int, 'the minimum (starting) resolution', required=True),
    max_res=Param(int, 'the maximum (starting) resolution', required=True),
    end_ramp=Param(int, 'when to stop interpolating resolution', required=True),
    start_ramp=Param(int, 'when to start interpolating resolution', required=True),
)

Section('validation', 'Validation parameters stuff').params(
    batch_size=Param(int, 'the batch size for validation', default=512),
    resolution=Param(int, 'final resized validation image size', default=224),
    lr_tta=Param(And(int, OneOf([0, 1])), 'should do lr flipping/avging at test time? (0/1)', default=0),
)

Section('logging', 'how to log stuff').params(
    dry_run=Param(bool, 'use log or not', is_flag=True),
    path=Param(Folder(), 'resume path, if new experiment leave blank', default=None),
    save_intermediate_frequency=Param(int, 'save extra checkpoints for every t epoch'),
)


class Trainer:

    @param('train.seed')
    @param('train.label_smoothing')
    def __init__(self, seed, label_smoothing):
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        set_seed(seed)
        self.loss = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.create_train_loader()
        self.create_val_loader()
        self.create_network_and_scaler()
        self.create_optimizer_and_scheduler()
        self.initialize_metrics()
        self.init_attacker_info()
        self.resume()
        self.run()

    @param('adv.epsilon')
    @param('adv.attack_step')
    def init_attacker_info(self, epsilon, attack_step):
        self.epsilon = epsilon / 255.0
        self.attack_steps = attack_step
        self.attack_alpha = self.epsilon * 2 / self.attack_steps
        self.normalize_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.normalize_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    @staticmethod
    @section('train.resolution')
    @param('min_res')
    @param('max_res')
    @param('end_ramp')
    @param('start_ramp')
    def get_resolution(epoch, min_res, max_res, end_ramp, start_ramp):
        assert min_res <= max_res

        if epoch <= start_ramp:
            return min_res

        if epoch >= end_ramp:
            return max_res

        interp = np.interp([epoch], [start_ramp, end_ramp], [min_res, max_res])
        final_res = int(np.round(interp[0] / 32)) * 32
        return final_res

    @param('network.architecture')
    @param('network.blurpool')
    def create_network_and_scaler(self, architecture, blurpool):
        self.scaler = GradScaler()
        if architecture == "resnet18":
            from torchvision.models import resnet18
            network = resnet18(weights=None)
        elif architecture == "resnet50":
            from torchvision.models import resnet50
            network = resnet50(weights=None)
        elif architecture == "resnet101":
            from torchvision.models import resnet101
            network = resnet101(weights=None)
        elif architecture == "resnet152":
            from torchvision.models import resnet152
            network = resnet152(weights=None)
        else:
            raise NotImplementedError(f"{architecture} is not supported")

        if blurpool:
            apply_blurpool(network)
        self.network = network.to(self.device)

    @param('dataset.train_path', 'path')
    @param('dataset.num_workers')
    @param('dataset.in_memory')
    @param('train.batch_size')
    @param('dataset.indices.training', 'indices')
    def create_train_loader(self, path, num_workers, in_memory, batch_size, indices=None):
        if indices is not None:
            indices = torch.load(indices, map_location="cpu")
        res = self.get_resolution(epoch=0)
        self.train_loader, self.decoder = get_train_loader(path, num_workers, in_memory, batch_size, res, self.device,
                                                           indices)

    @param('dataset.val_path', 'path')
    @param('dataset.num_workers')
    @param('validation.batch_size')
    @param('validation.resolution')
    @param('dataset.indices.testing', 'indices')
    def create_val_loader(self, path, num_workers, batch_size, resolution, indices=None):
        if indices is not None:
            indices = torch.load(indices, map_location="cpu")
        self.val_loader = get_val_loader(path, num_workers, batch_size, resolution, self.device, indices)

    @param('train.epoch')
    def get_cosine_scheduler(self, epoch):
        return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epoch * len(self.train_loader))

    @param('train.scheduler.step.step_ratio')
    @param('train.scheduler.step.step_size')
    def get_step_scheduler(self, step_ratio, step_size):
        return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size * len(self.train_loader),
                                               gamma=step_ratio)

    @param('train.epoch')
    @param('train.scheduler.cyclic.lr_peak_epoch')
    def get_cyclic_scheduler(self, epoch, lr_peak_epoch):
        return torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=1e-4,
                                                 max_lr=self.optimizer.param_groups[0]['lr'],
                                                 step_size_up=lr_peak_epoch * len(self.train_loader),
                                                 step_size_down=(epoch - lr_peak_epoch) * len(self.train_loader))

    @param('train.optimizer.lr')
    @param('train.optimizer.weight_decay')
    @param('train.optimizer.momentum')
    @param('train.scheduler_type')
    def create_optimizer_and_scheduler(self, lr, weight_decay, momentum, scheduler_type):
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
        self.scheduler = eval(f'self.get_{scheduler_type}_scheduler()')

    def initialize_metrics(self):
        self.train_meters = {
            'top_1': torchmetrics.Accuracy(task='multiclass', num_classes=1000, compute_on_step=False).to(self.device),
            'loss': MeanScalarMetric(compute_on_step=False).to(self.device)
        }
        self.val_meters = {
            'top_1': torchmetrics.Accuracy(task='multiclass', num_classes=1000, compute_on_step=False).to(self.device),
            'top_5': torchmetrics.Accuracy(task='multiclass', num_classes=1000, compute_on_step=False, top_k=5).to(
                self.device),
            'loss': MeanScalarMetric(compute_on_step=False).to(self.device)
        }
        self.start_time = time()
        self.best_acc = 0.
        self.start_epoch = 0

    @param('logging.path')
    def resume(self, path=None):
        try:
            ckpt = torch.load(os.path.join(path, "checkpoints", "newest.ckpt"), map_location=self.device)
            for key, val in ckpt["state_dicts"].items():
                eval(f"self.{key}.load_state_dict(val)")
            self.best_acc = ckpt["best_acc"]
            self.start_epoch = ckpt["current_epoch"]
            self.start_time -= ckpt["relative_time"]
        except FileNotFoundError:
            os.makedirs(os.path.join(path, "checkpoints"), exist_ok=False)
        except TypeError:
            pass

    @param('logging.path')
    def log(self, content, path):
        print(f'=> Log: {content}')
        cur_time = time()
        path = os.path.join(path, 'log.json')
        stats = {
            'timestamp': cur_time,
            'relative_time': cur_time - self.start_time,
            **content
        }
        if os.path.isfile(path):
            with open(path, 'r') as fd:
                old_data = json.load(fd)
            with open(path, 'w') as fd:
                fd.write(json.dumps(old_data + [stats]))
                fd.flush()
        else:
            with open(path, 'w') as fd:
                fd.write(json.dumps([stats]))
                fd.flush()

    @param('train.epoch')
    @param('logging.dry_run')
    @param('logging.path')
    @param('logging.save_intermediate_frequency')
    def run(self, epoch, dry_run, path=None, save_intermediate_frequency=None):
        for e in range(self.start_epoch, epoch):
            res = self.get_resolution(e)
            self.decoder.output_size = (res, res)
            train_stats = self.train_loop(e)
            val_stats = self.val_loop()

            if not dry_run:
                ckpt = {
                    "state_dicts": {
                        "network": self.network.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.state_dict(),
                    },
                    "current_epoch": e + 1,
                    "best_acc": self.best_acc,
                    "relative_time": time() - self.start_time,
                }
                if val_stats['top_1'] > self.best_acc:
                    self.best_acc = val_stats['top_1']
                    ckpt['best_acc'] = self.best_acc
                    torch.save(ckpt, os.path.join(path, "checkpoints", "best.ckpt"))
                torch.save(ckpt, os.path.join(path, "checkpoints", "newest.ckpt"))
                if save_intermediate_frequency is not None:
                    if (e + 1) % save_intermediate_frequency == 0:
                        torch.save(ckpt, os.path.join(path, "checkpoints", f"epoch{e}.ckpt"))

                self.log(content={
                    'epoch': e,
                    'train': train_stats,
                    'val': val_stats,
                    'best_val_top1': self.best_acc,
                })

    def train_loop(self, epoch):
        self.network.train()

        iterator = tqdm(self.train_loader, ncols=160)
        for images, target, _ in iterator:

            ### Generate adversarial examples
            x_pgd = Variable(images.data, requires_grad=True)
            random_noise = (
                torch.FloatTensor(x_pgd.shape).uniform_(-self.epsilon, self.epsilon).to(self.device)
            )
            x_pgd = Variable(x_pgd.data + random_noise, requires_grad=True)

            for _ in range(self.attack_steps):
                with torch.enable_grad(), autocast():
                    result = self.network(x_pgd)
                    loss = self.loss(result, target)
                self.scaler.scale(loss).backward()
                eta = self.attack_alpha * x_pgd.grad.data.sign()
                x_pgd = Variable(x_pgd.data + eta, requires_grad=True)
                with torch.no_grad():
                    x_pgd_original = x_pgd * self.normalize_std + self.normalize_mean
                    x_original = images * self.normalize_std + self.normalize_mean
                    eta = torch.clamp(x_pgd_original.data - x_original.data, -self.epsilon, self.epsilon)
                    x_pgd_original = Variable(x_original.data + eta)
                    x_pgd_original = Variable(torch.clamp(x_pgd_original, 0.0, 1.0))
                x_pgd = Variable((x_pgd_original - self.normalize_mean) / self.normalize_std, requires_grad=True)

            x_pgd.detach().requires_grad_(False)

            ### Training start
            self.optimizer.zero_grad(set_to_none=True)

            with autocast():
                output = self.network(x_pgd)
                loss_train = self.loss(output, target)

            self.scaler.scale(loss_train).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            ### Training end

            self.train_meters['top_1'](output, target)
            self.train_meters['loss'](loss_train)
            stats = {k: m.compute().item() for k, m in self.train_meters.items()}

            group_lrs = []
            for _, group in enumerate(self.optimizer.param_groups):
                group_lrs.append(f'{group["lr"]:.3f}')

            names = ['ep', 'lrs', 'acc', 'loss']
            values = [epoch, group_lrs, f"{stats['top_1']:.3f}", f"{stats['loss']:.3f}"]

            msg = ', '.join(f'{n}={v}' for n, v in zip(names, values))
            iterator.set_description(msg)

        [meter.reset() for meter in self.train_meters.values()]
        return stats

    @param('validation.lr_tta')
    def val_loop(self, lr_tta):
        self.network.eval()

        iterator = tqdm(self.val_loader, ncols=120)

        for images, target, _ in iterator:
            with torch.no_grad(), autocast():
                output = self.network(images)
                if lr_tta:
                    output += self.network(torch.flip(images, dims=[3]))

            for k in ['top_1', 'top_5']:
                self.val_meters[k](output, target)

            loss_val = self.loss(output, target)
            self.val_meters['loss'](loss_val)
            stats = {k: m.compute().item() for k, m in self.val_meters.items()}

            names = ['acc', 'loss']
            values = [f"{stats['top_1']:.3f}", f"{stats['loss']:.3f}"]

            msg = ', '.join(f'{n}={v}' for n, v in zip(names, values))
            iterator.set_description(msg)

        [meter.reset() for meter in self.val_meters.values()]
        return stats


if __name__ == "__main__":
    config = get_current_config()
    parser = argparse.ArgumentParser("Imagenet Train from scratch")
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)

    if config['logging.path'] is not None:
        assert not config['logging.dry_run'], "dry run can not accept resume path!"
        config.collect_config_file(os.path.join(config['logging.path'], 'config.json'))
        config.validate()
    else:
        config.validate()
        if config['exp.identifier'] is not None:
            file_name = config['exp.identifier']
        else:
            file_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")

        path = os.path.join("file", "experiments", os.path.basename(__file__.split('.')[0]), file_name)
        if not config['logging.dry_run']:
            os.makedirs(path, exist_ok=False)
            config.dump_json(os.path.join(path, 'config.json'),
                             [('logging', 'path'), ('logging', 'dry_run'), ('exp', 'identifier')])
            config.collect({'logging': {'path': path}})

    config.summary()

    Trainer()
