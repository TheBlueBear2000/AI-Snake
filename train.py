import torch.nn as nn


class SnakeNet(nn.Module):
    """
    Input  : 19 features
    Output : direction (0-3 = up, down, left, right)
    """

    def __init__(self, n_inputs=N_INPUTS, n_outputs=N_OUTPUTS, hidden_sizes=(256, 128)):
        super().__init__()
        layers, prev = [], n_inputs
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(0.1)]
            prev = h
        layers += [nn.Linear(prev, n_outputs), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
