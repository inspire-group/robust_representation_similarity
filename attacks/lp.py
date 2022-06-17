import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


def linf(model, x, y, allparams):
    params = getattr(allparams, "linf")
    device = x.device

    random_noise = (
        torch.FloatTensor(x.shape)
        .uniform_(-params.epsilon, params.epsilon)
        .to(device)
        .detach()
    )
    xadv = Variable(x.detach().data + random_noise, requires_grad=True)

    for _ in range(params.steps):
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(xadv), y)
        loss.backward()
        xadv.data = xadv.data + params.step_size * xadv.grad.data.sign()
        eta = torch.clamp(xadv.data - x.data, -params.epsilon, params.epsilon)
        xadv.data = torch.clamp(x.data + eta, params.clip_min, params.clip_max)
        xadv.grad.data = torch.zeros_like(xadv.grad.data)
    return xadv, y


def l2(model, x, y, allparams):
    params = getattr(allparams, "l2")
    device = x.device

    random_noise = torch.FloatTensor(x.shape).uniform_(-1, 1).to(device).detach()
    random_noise.renorm_(p=2, dim=0, maxnorm=params.epsilon)

    xadv = Variable(x.detach().data + random_noise, requires_grad=True)

    for _ in range(params.steps):
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(xadv), y)
        loss.backward()
        grad_norms = xadv.grad.view(len(x), -1).norm(p=2, dim=1)
        xadv.grad.div_(grad_norms.view(-1, 1, 1, 1))
        if (grad_norms == 0).any():
            xadv.grad[grad_norms == 0] = torch.randn_like(xadv.grad[grad_norms == 0])

        xadv.data += params.step_size * xadv.grad.data
        eta = xadv.data - x.data
        eta.renorm_(p=2, dim=0, maxnorm=params.epsilon)
        xadv.data = torch.clamp(x.data + eta, params.clip_min, params.clip_max)
        xadv.grad.data = torch.zeros_like(xadv.grad.data)
    return xadv, y
