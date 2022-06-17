import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from easydict import EasyDict
from collections import OrderedDict

from utils.attacks import get_attack_vector


def register_hooks(model, save_dict, layers, pool):
    def add_hook(key, layer, input, output):
        if pool and (len(output.shape) == 4):
            save_dict[key] = F.adaptive_avg_pool2d(output, (1, 1)).flatten(start_dim=1)
        else:
            save_dict[key] = output.flatten(start_dim=1)

    if layers == "all" or (layers[0] == "all" and len(layers) == 1):
        for name, layer in model.named_modules():
            if isinstance(
                layer,
                (
                    nn.Conv2d,
                    nn.BatchNorm2d,
                    nn.ReLU,
                    nn.AvgPool2d,
                    nn.AdaptiveAvgPool2d,
                    nn.Linear,
                ),
            ):
                layer.register_forward_hook(partial(add_hook, name))
                print(f"Hook registered for layer {name}")
    # Add hook for a fraction of layers, e.g., frac_0.3 will add hooks on first one third layers
    elif layers[0].startswith("frac") and len(layers) == 1:
        all_layers = []
        for name, layer in model.named_modules():
            if isinstance(
                layer,
                (
                    nn.Conv2d,
                    nn.BatchNorm2d,
                    nn.ReLU,
                    nn.AvgPool2d,
                    nn.AdaptiveAvgPool2d,
                    nn.Linear,
                ),
            ):
                all_layers.append([layer, name])

        f = float(layers[0].split("_")[-1])  # assuming input is in "frac_xyz" format
        indices = np.linspace(
            0, len(all_layers) - 1, int(len(all_layers) * f), dtype=int
        )
        print(f"Selecting {len(indices)} layer out of {len(all_layers)} layers.")
        for i in indices:
            selected_layer, name = all_layers[i]
            selected_layer.register_forward_hook(partial(add_hook, name))
            print(f"Hook registered for layer {name}")
    else:
        for name, layer in model.named_modules():
            if name in layers:
                layer.register_forward_hook(partial(add_hook, name))
                print(f"Hook registered for layer {name}")


def copyDictToCpu(d):
    dout = OrderedDict({})
    for (k, v) in d.items():
        dout[k] = v.detach().cpu()
    return dout


def forward_pass_on_batch(model, img, feat_hooks, args):
    agg_feat = OrderedDict({})
    for i, im in enumerate(torch.split(img, args.microbatch)):
        _ = model(im)
        if i == 0:
            agg_feat = copyDictToCpu(feat_hooks)
        else:
            for (k, v) in feat_hooks.items():
                agg_feat[k] = torch.cat([agg_feat[k], v.detach().cpu()])
    return agg_feat


def kernel_on_gpu(feat):
    kernel = []
    for f in feat:
        f = f.cuda()
        kernel.append(f @ f.t())
    return kernel


def calculate_hsic(K, L):
    """
    Computes the unbiased estimate of HSIC metric.
    Reference: https://arxiv.org/pdf/2010.15327.pdf
    """
    K.fill_diagonal_(0.0)
    L.fill_diagonal_(0.0)
    N = K.shape[0]
    ones = torch.ones(N, 1).to(K.device)
    result = torch.trace(K @ L)
    result += ((ones.t() @ K @ ones @ ones.t() @ L @ ones) / ((N - 1) * (N - 2))).item()
    result -= ((ones.t() @ K @ L @ ones) * 2 / (N - 2)).item()
    return (1 / (N * (N - 3)) * result).item()


def generate_adv_examples(model, images, labels, attack, config, batch):
    adv_examples = []
    with open(config, "r") as f:
        config = EasyDict(yaml.load(f))
    vec = get_attack_vector(attack, config.EvalAttack)
    images, labels = images.cuda(), labels.cuda()
    for indices in torch.split(torch.arange(len(images)), batch):
        im, label = images[indices], labels[indices]
        adv_examples.append(vec(model, im, label, None)[0].detach())
    return torch.cat(adv_examples)
