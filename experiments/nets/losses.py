import torch
from torch import nn


class RMSELoss(nn.Module):
    """ Simple torch implementation of the RMSE loss. """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = torch.tensor(eps)

    def forward(self, y_pred, y):
        loss = torch.sqrt(self.mse(y_pred, y) + self.eps.to(y.device))
        return loss


class PvarLossWrapper(nn.Module):
    """Wraps a loss function and computes the additional p-var component.

    This assumes that the predictions made in `y_pred` is of shape 2 * N where N is the true batch dim. The first half
    is assumed to be the normal data, and the second half is assumed to be the predictions made when the step size is
    twice that of y_pred[:y_pred.size(0)/2]. And thus we can compute the loss as:
        Loss p-var = loss(y_pred[0:N], y[0:N]) + lambda * |y_pred[0:N:2] - y_pred[N:N+N/2]|
    where |o| represents some norm.
    """
    def __init__(self, loss_fn, lmbda=0.):
        super(PvarLossWrapper, self).__init__()
        self.loss_fn = loss_fn
        self.lmbda = lmbda

    def forward(self, y_pred, y):
        total_batch = y_pred.size(0)
        assert total_batch % 2 == 0  # Sanity check
        N = int(total_batch / 2)
        true_loss = self.loss_fn(y_pred[:N], y[:N])
        pvar_loss = self.lmbda * (y_pred[slice(0, N, 2)] - y_pred[slice(N, N + int(N/2))]).abs().mean()
        return true_loss + pvar_loss


def kld_loss(mu, logvar):
    """ Kullback-Lieber divergence loss for VAEs. """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


