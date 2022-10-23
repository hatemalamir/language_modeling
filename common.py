import time
import numpy as np
import torch


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))


def fit(epochs, model, loss_func, opt, train_dl, valid_dl=None):
    for epoch in range(epochs):
        msg = f'>>> epoch: {epoch}'
        model.train()
        start = time.time()
        train_loss = calc_epoch_loss(model, loss_func, opt, train_dl)
        msg += f', train loss: {train_loss}, train time: {round(time.time() - start, 3)} s'
        if valid_dl:
            model.eval()
            start = time.time()
            with torch.no_grad():
                eval_loss = calc_epoch_loss(model, loss_func, None, valid_dl)
            msg += f', eval loss: {eval_loss}, eval time: {round(time.time() - start, 3)} s'
        print(msg)


def calc_epoch_loss(model, loss_func, opt, dl):
    losses, nums = zip(*[loss_batch(model, loss_func, xb, yb, opt) for xb, yb in dl])

    return np.sum(np.multiply(losses, nums)) / np.sum(nums)


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)
