import torch
import torch.nn.functional as F

from attacks import linf, l2, SnowAttack, GaborAttack, JPEGAttack


def get_attack_vector(name, allparams):
    if name == "none":
        attack_vector = lambda model, x, ytrue, ytarget: (x, ytrue)
    elif name == "linf":
        attack_vector = lambda model, x, ytrue, ytarget: linf(
            model, x, ytrue, allparams
        )
    elif name == "l2":
        attack_vector = lambda model, x, ytrue, ytarget: l2(model, x, ytrue, allparams)
    elif name == "snow":
        params = getattr(allparams, "snow")
        attack = SnowAttack(
            nb_its=params.nb_its,
            eps_max=params.eps_max,
            step_size=params.step_size,
            resol=params.resol,
            rand_init=params.rand_init,
            scale_each=params.scale_each,
            budget=params.budget,
        )
        attack_vector = lambda model, x, ytrue, ytarget: (
            attack._forward(
                lambda x: model(x / 255.0),
                x,
                ytarget,
                avoid_target=True,
                scale_eps=False,
            ),
            ytrue,
        )
    elif name == "gabor":
        params = getattr(allparams, "gabor")
        attack = GaborAttack(
            nb_its=params.nb_its,
            eps_max=params.eps_max,
            step_size=params.step_size,
            resol=params.resol,
            rand_init=params.rand_init,
            scale_each=params.scale_each,
        )
        attack_vector = lambda model, x, ytrue, ytarget: (
            attack._forward(
                lambda x: model(x / 255.0),
                x,
                ytarget,
                avoid_target=True,
                scale_eps=False,
            ),
            ytrue,
        )
    elif name == "jpeg":
        params = getattr(allparams, "jpeg")
        attack = JPEGAttack(
            nb_its=params.nb_its,
            eps_max=params.eps_max,
            step_size=params.step_size,
            resol=params.resol,
            rand_init=params.rand_init,
            scale_each=params.scale_each,
            opt=params.opt,
        )
        attack_vector = lambda model, x, ytrue, ytarget: (
            attack._forward(
                lambda x: model(x / 255.0),
                x,
                ytarget,
                avoid_target=True,
                scale_eps=False,
            ),
            ytrue,
        )
    else:
        raise ValueError(f"{name} attack vector not supported")

    return attack_vector
