from my_utils import *


class PositionalEmbedding(nn.Module):
    __doc__ = r"""Computes a positional embedding of timesteps.
    Input:
        x: tensor of shape (N)
    Output:
        tensor of shape (N, dim)
    Args:
        dim (int): embedding dimension
        scale (float): linear scale to be applied to timesteps. Default: 1.0
    """

    def __init__(self, dim, scale=1.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.outer(x * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Diff(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=1024):
        super().__init__()
        self.activation = F.relu
        self.norm = nn.BatchNorm1d(1024)

        self.demo_dim = state_dim + action_dim
        self.hidden_dim = 1024
        self.linear_1 = torch.nn.Linear(self.demo_dim, self.hidden_dim, bias=True)
        self.linear_2 = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.linear_3 = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.linear_4 = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.linear_5 = torch.nn.Linear(self.hidden_dim, self.demo_dim, bias=True)
        self.dropout = nn.Dropout(0.2)

        # self.linear_5.weight.data.mul_(0.1)
        # self.linear_5.bias.data.mul_(0.0)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.linear_2(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.linear_3(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.linear_4(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.linear_5(x)

        return x

class Discriminator_dwbc(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Discriminator_dwbc, self).__init__()

        self.fc1_1 = nn.Linear(state_dim + action_dim, 128)
        self.fc1_2 = nn.Linear(action_dim, 128)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action, log_pi):
        sa = torch.cat([state, action], 1)
        d1 = F.relu(self.fc1_1(sa))
        d2 = F.relu(self.fc1_2(log_pi))
        d = torch.cat([d1, d2], 1)
        d = F.relu(self.fc2(d))
        d = F.sigmoid(self.fc3(d))
        d = torch.clip(d, 0.1, 0.9)
        return d