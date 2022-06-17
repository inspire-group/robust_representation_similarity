import os
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict
import torch
import torch.nn as nn

import models
import data
from similarity_metrics.cka_core import cka
from utils.features import (
    register_hooks,
    generate_adv_examples,
    forward_pass_on_batch,
    calculate_hsic,
    kernel_on_gpu,
)


def self_hsic(feat):
    # measure hsic for every pair of layers
    hsic_vec = np.empty(len(feat))
    kernel = kernel_on_gpu(feat)
    for i, k in enumerate(kernel):
        hsic_vec[i] = calculate_hsic(k, k)
    return hsic_vec


def cross_hsic(feat0, feat1):
    """
    Measure cross hsic for every pair layers amont the two set of features
    """
    kernel0 = kernel_on_gpu(feat0)
    kernel1 = kernel_on_gpu(feat1)

    hsic_matrix = np.empty((len(feat0), len(feat1)))
    for i, k0 in enumerate(kernel0):
        for j, k1 in enumerate(kernel1):
            hsic_matrix[i, j] = calculate_hsic(k0, k1)
    return hsic_matrix


def full_cka(feat0, feat1, debiased):
    cka_matrix = np.empty((len(feat0), len(feat1)))
    kernel0 = kernel_on_gpu(feat0)
    kernel1 = kernel_on_gpu(feat1)
    for i, k0 in enumerate(kernel0):
        for j, k1 in enumerate(kernel1):
            cka_matrix[i, j] = cka(
                k0.cpu().numpy(), k1.cpu().numpy(), debiased=debiased
            )

    return cka_matrix


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=("cifar10", "imagenette", "imagewoof"),
    )
    parser.add_argument(
        "--datadir",
        type=str,
        default="./datasets/",
        help="Directory where dataset is stored",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=10000,
        help="Number of images to use in CKA analysis",
    )
    parser.add_argument(
        "--dataset_passes",
        type=int,
        default=3,
        help="Number of passes over datasets to address stochasticity in online CKA.",
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=["wrn_28_1"],
        help="Network arch for each model. Just one input implies that both models will have the same arch",
    )
    parser.add_argument(
        "--num_classes", type=int, default=10, help="Number of classes in the dataset"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        nargs="+",
        default=["./trained_models/cifar10_wrn_28_10_robust/checkpoint.pth.tar"],
        help="Checkpoint of the trained network. Just one input implies that both models will share it",
    )
    parser.add_argument(
        "--adv",
        type=str,
        nargs="+",
        default=["0"],
        help="where adversarial examples are used in feature extraction, i.e., adversarial features."
        + "1 refers to adv. features and zero to benign features. Thus ['0', '1'] implies that benign features"
        + " for first network and adversarial features for the second are used.",
    )
    parser.add_argument(
        "--attack_configs",
        type=str,
        nargs="+",
        default=["./configs/configs_cifar.yml"],
        help="attach configuration for adversarial examples. Just one input implies that both models will share it",
    )
    parser.add_argument(
        "--attack",
        type=str,
        nargs="+",
        default=["linf"],
        help="Type of adversarial attacks (linf, l2, snow, gabor, jpeg). Make sure to use the correct attack config",
    )
    parser.add_argument(
        "--reuse_adv",
        action="store_true",
        default=False,
        help="Reuse first model adv. examples for second model.",
    )
    parser.add_argument(
        "--model0_layers",
        type=str,
        nargs="+",
        default="all",
        help="list of first model layers (precise names from model.named_modules) in cross-layer CKA",
    )
    parser.add_argument(
        "--model1_layers",
        type=str,
        nargs="+",
        default="all",
        help="list of second model layers (precise names from model.named_modules) in cross-layer CKA",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch-size for online CKA. For full-cka it should be size of dataset.",
    )
    parser.add_argument(
        "--microbatch",
        type=int,
        default=128,
        help="Splitting each batch in small microbatches for forward pass",
    )
    parser.add_argument(
        "--full_cka",
        action="store_true",
        default=False,
        help="Calculate CKA while features of whole dataset (thus no online cka)",
    )
    parser.add_argument(
        "--debiased",
        action="store_true",
        default=False,
        help="Use debiased full cka",
    )
    parser.add_argument(
        "--pool_features",
        action="store_true",
        default=False,
        help="Spatial pooling (avgpool) of features to reduce memory overhead (used for large resolution images)",
    )
    parser.add_argument(
        "--savedir",
        type=str,
        default="/shadowdata/vvikash/spring22/representation_similarity/results/cka/cifar10",
        help="dir where cka results will be sotred",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="suffix to add in the filename for cka at saving time",
    )
    args = parser.parse_args()
    return args


def main():
    """
    Calculate Online or Full CKA.

    All common similarity metrics take the form of d(f_a, f_b), where they measure distance between the two
    sets of features. These features will differ base on many design choices: such as the network arch,
    robust vs non-robust model, adversarial vs benign feature, choice of threat model adversarial feature,
    configuration for adversarial attacks, etc. This script is a one-fits-all approach to conduct most
    of these aforementioned analysis.

    Since CKA expect two setup of features, all common args are configured to have two inputs, e.g.,
    model architecture, pre-trained checkpoint, attack configs etc. To avoid verbosity, one can only
    pass one value, which will be shared among the two networks.

    By default, we conduct the most-exhasutive analysis, i.e, masure cross-layer cka between all pairs of layers
    in the two network. Consider using only a few layers if it throws an oom error.
    """

    # Args and check key flags
    args = parse_args()
    for (k, v) in dict(vars(args)).items():
        if isinstance(v, list) and k not in ["model0_layers", "model1_layers"]:
            if len(v) == 1:
                setattr(args, k, v + v)

    str_to_bool_list = lambda d: [bool(int(s)) for s in d]
    args.adv = str_to_bool_list(args.adv)
    assert args.batch_size % args.microbatch == 0.0, "for simplicty"
    args.identical_models = (args.model[0] == args.model[1]) and (
        args.model_path[0] == args.model_path[1]
    )
    args.identical_attacks = (
        args.identical_models
        and (args.adv[0] == args.adv[1])
        and (args.attack[0] == args.attack[1])
        and (args.attack_configs[0] == args.attack_configs[1])
    )
    if args.full_cka:
        assert args.dataset_passes == 1
        assert args.batch_size == args.num_images, "Use whole dataset for cka"
    # We use hooks to extract internal representations, which doesn't play nice with DataParallel
    # The other solution is to use DistributedDataParallel, which we avoid in favor simplicity
    assert torch.cuda.device_count() == 1, "only single gpu evaluation is supported"

    # hooks will add features in these dicts in the forward pass
    model0_features, model1_features = OrderedDict({}), OrderedDict({})

    # Create models and load checkpoints
    model0 = (
        nn.DataParallel(getattr(models, args.model[0])(num_classes=args.num_classes))
        .eval()
        .cuda()
    )
    model0.load_state_dict(
        torch.load(Path(args.model_path[0]), map_location="cpu")["state_dict"]
    )
    register_hooks(model0, model0_features, args.model0_layers, args.pool_features)
    model1 = (
        nn.DataParallel(getattr(models, args.model[1])(num_classes=args.num_classes))
        .eval()
        .cuda()
    )
    model1.load_state_dict(
        torch.load(Path(args.model_path[1]), map_location="cpu")["state_dict"]
    )
    register_hooks(model1, model1_features, args.model1_layers, args.pool_features)

    # Cross HSIC matrices (only used in online CKA)
    cross_hsic_matrix, self_hsic_model0, self_hsic_model1 = [], [], []

    # iterate over dataset, extract embeddeing, measure per-batch cka
    for cycle in range(args.dataset_passes):
        count = 0

        # create dataloader
        _, val_loader = getattr(data, args.dataset)(
            args.datadir,
            batch_size=args.batch_size,
        )

        for _, (img, label) in tqdm(enumerate(val_loader)):
            assert (
                img.max().item() <= 1.0 and img.min().item() >= 0.0
            )  # expected [0, 1] pixel range for images

            # extract features
            if args.adv[0]:
                img0 = generate_adv_examples(
                    model0,
                    img,
                    label,
                    args.attack[0],
                    args.attack_configs[0],
                    args.microbatch,
                )
            else:
                img0 = img.cuda()
            feat0 = forward_pass_on_batch(model0, img0, model0_features, args)

            if args.adv[1]:
                # resuse adversarial examples when feasible to avoid computational cost of re-generating them
                if args.reuse_adv:
                    assert args.adv[
                        0
                    ], "reusing adv. examples requires them to be generated for model-0"
                    img1 = img0
                    print("Reusing adv. examples")
                else:
                    img1 = generate_adv_examples(
                        model1,
                        img,
                        label,
                        args.attack[1],
                        args.attack_configs[1],
                        args.microbatch,
                    )
            else:
                img1 = img.cuda()
            feat1 = forward_pass_on_batch(model1, img1, model1_features, args)

            # TODO: See if it works with models from robsutbench as they may not have fc layer
            # measure accuracy of each model (a quick sanity check of feature quality)
            if cycle == 0 and count == 0:
                try:
                    print(
                        f"Model-0 accuracy: {(feat0['module.fc'].max(dim=-1)[1] == label).float().mean()}",
                    )
                    print(
                        f"Model-1 accuracy: {(feat1['module.fc'].max(dim=-1)[1] == label).float().mean()}",
                    )
                except KeyError:
                    pass

            if not args.full_cka:
                # log HSIC values for all cross-layers between feat0 and feat1 and self-layers.
                cross_hsic_matrix.append(cross_hsic(feat0.values(), feat1.values()))
                self_hsic_model0.append(self_hsic(feat0.values()))
                self_hsic_model1.append(self_hsic(feat1.values()))

            count += len(img)
            if count >= args.num_images:
                break

    # Calculate CKA
    if args.full_cka:
        cka = full_cka(feat0.values(), feat1.values(), args.debiased)
    else:
        hsic_matrix = np.mean(np.stack(cross_hsic_matrix), axis=0)
        self_hsic_model0 = np.mean(np.stack(self_hsic_model0), axis=0).reshape(-1, 1)
        self_hsic_model1 = np.mean(np.stack(self_hsic_model1), axis=0).reshape(-1, 1)
        denominator = np.sqrt(self_hsic_model0 @ self_hsic_model1.T)
        cka = hsic_matrix / denominator
    save_dict = {
        "cka": cka,
        "model0_keys": list(feat0.keys()),
        "model1_keys": list(feat1.keys()),
        "args": dict(vars(args)),
    }
    # Saving CKA: We add all common experimental flags in the filename to easily distinguish between
    # different runs. In addition, one can also add a custom suffix to the filename.
    getkey = lambda x: x[0] if isinstance(x, list) else x
    layer_keyword = (
        lambda layers: getkey(layers)
        if (layers == "all" or len(layers) == 1)
        else str(len(layers))
    )
    os.makedirs(args.savedir, exist_ok=True)
    torch.save(
        save_dict,
        os.path.join(
            args.savedir,
            (
                f"Model_{args.model[0]}_{args.model[1]}-Adv_{args.adv[0]}_{args.adv[1]}-layers_{layer_keyword(args.model0_layers)}"
                + f"_{layer_keyword(args.model1_layers)}-reuse_adv_{args.reuse_adv}-identical_models_{args.identical_models}"
                + f"-identical_attacks_{args.identical_attacks}-batchsize_{args.batch_size}"
                + f"{'_pooled_features' if args.pool_features else ''}{'_'+args.suffix if args.suffix else ''}.pt"
            ),
        ),
    )


if __name__ == "__main__":
    main()
