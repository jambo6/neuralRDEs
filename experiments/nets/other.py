import torch
from torch import nn


class MLP(nn.Module):
    """ Simple MLP network. """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Additional layers are just hidden to hidden with relu activation
        additional_layers = [nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)] * (num_layers - 1) if num_layers > 1 else []

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            *additional_layers,
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class StaticDataNetwork(nn.Module):
    """A network that wraps a sequential network and can operate with static data.

    This net applies a nonlinear network to the final hidden layer of some network concatenated with the static data.
    """
    def __init__(self, model, static_dim, hidden_dim):
        super().__init__()
        assert hasattr(model, 'final_linear'), "Model must have a layer named 'final_linear' to be overwritten'"
        self.model = model
        self.static_dim = static_dim
        self.hidden_dim = hidden_dim

        # Replace the final linear layer with a network
        in_features = self.model.final_linear.in_features
        out_features = self.model.final_linear.out_features
        self.model.final_linear = FinalNonlinear(in_features + static_dim, hidden_dim, out_features)

    def forward(self, inputs):
        initial, controls, static = inputs
        self.model.final_linear.static = static
        return self.model((initial, controls))


class FinalNonlinear(nn.Module):
    """ Final nonlinear layer applied to the hidden state and static features. """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(FinalNonlinear, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        additional_layers = [nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)] * (num_layers - 1) if num_layers > 1 else []

        self.final_linear = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            *additional_layers,
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, inputs):
        return self.final_linear(torch.cat([inputs, self.static], 1))
