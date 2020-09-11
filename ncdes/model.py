"""
model.py
===========================
This contains a model class for NeuralRDEs that wraps `rdeint` as a `nn.Module` that will act similarly to an RNN.
"""
from torch import nn
from ncdes import rdeint


class NeuralRDE(nn.Module):
    """The generic module for learning with Neural RDEs.

    This class wraps the `NeuralRDECell` that acts like an RNN-Cell. This method simply initialises the hidden dynamics
    and computes the updated hidden dynamics through a call to `ode_int` using the `NeuralRDECell` as the function that
    computes the update.

    Here we model the dynamics of some abstract hidden state H via a CDE, and the response as a linear functional of the
    hidden state, that is:
        dH = f(H)dX;    Y = L(H).
    """
    def __init__(self,
                 initial_dim,
                 logsig_dim,
                 hidden_dim,
                 output_dim,
                 hidden_hidden_dim=15,
                 num_layers=3,
                 apply_final_linear=True,
                 solver='midpoint',
                 adjoint=False,
                 return_sequences=False):
        """
        Args:
            initial_dim (int): We use the initial value (t_0 x_0) as an initial condition else we have translation
                invariance.
            logsig_dim (int): The dimension of the log-signature.
            hidden_dim (int): The dimension of the hidden state.
            output_dim (int): The dimension of the output.
            hidden_hidden_dim (int): The dimension of the hidden layer in the RNN-like block.
            num_layers (int): The number of hidden layers in the vector field. Set to 0 for a linear vector field.
            apply_final_linear (bool): Set False to ignore the final linear output.
            solver (str): ODE solver, must be implemented in torchdiffeq.
            adjoint (bool): Set True to use odeint_adjoint.
            return_sequences (bool): If True will return the linear function on the final layer, else linear function on
                all layers.
        """
        super().__init__()
        self.initial_dim = initial_dim
        self.logsig_dim = logsig_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hidden_hidden_dim = hidden_hidden_dim
        self.num_layers = num_layers
        self.apply_final_linear = apply_final_linear
        self.solver = solver
        self.adjoint = adjoint
        self.return_sequences = return_sequences

        # Initial to hidden
        self.initial_linear = nn.Linear(initial_dim, hidden_dim)

        # The net applied to h_prev
        self.func = _NRDEFunc(hidden_dim, logsig_dim, hidden_dim=hidden_hidden_dim, num_layers=num_layers)

        # Linear classifier to apply to final layer
        self.final_linear = nn.Linear(self.hidden_dim, self.output_dim) if apply_final_linear else lambda x: x

    def forward(self, inputs):
        # Setup the inital hidden layer
        assert len(inputs) == 2, "`inputs` must be a 2-tuple containing `(inital_values, logsig)`."
        initial, logsig = inputs
        h0 = self.initial_linear(initial)

        # Perform the adjoint operation
        out = rdeint(
            logsig, h0, self.func, method=self.solver, adjoint=self.adjoint, return_sequences=self.return_sequences
        )

        # Outputs
        outputs = self.final_linear(out[:, -1, :]) if not self.return_sequences else self.final_linear(out)

        return outputs


class _NRDEFunc(nn.Module):
    """The function applied to the hidden state in the log-ode method.

    This creates a simple RNN-like block to be used as the computation function f in:
        dh/dt = f(h) o logsig(X_{[t_i, t_{i+1}]})

    To build a custom version, simply use any NN architecture such that `input_dim` is the size of the hidden state,
    and the output dim must be of size `input_dim * logsig_dim`. Simply reshape the output onto a tensor of size
    `[batch, input_dim, logsig_dim]`.
    """
    def __init__(self, input_dim, logsig_dim, num_layers=1, hidden_dim=15):
        super().__init__()
        self.input_dim = input_dim
        self.logsig_dim = logsig_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Additional layers are just hidden to hidden with relu activation
        additional_layers = [nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)] * (num_layers - 1) if num_layers > 1 else []

        # The net applied to h_prev
        self.net = nn.Sequential(*[
            nn.Linear(input_dim, hidden_dim),
            *additional_layers,
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim * logsig_dim),
        ]) if num_layers > 0 else nn.Linear(input_dim, input_dim * logsig_dim)

    def forward(self, h):
        return self.net(h).view(-1, self.input_dim, self.logsig_dim)


if __name__ == '__main__':
    NeuralRDE(10, 20, 15, 5, hidden_hidden_dim=90, num_layers=3)

