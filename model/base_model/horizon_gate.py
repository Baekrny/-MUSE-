import torch


class ScalarHorizonGate(torch.nn.Module):
    def __init__(self, interest_dim, hidden_dim):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(interest_dim * 3, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Sigmoid(),
        )

    def reset_parameters(self):
        for module in self.network:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def forward(self, target, short_interest, long_interest):
        weight = self.network(
            torch.cat([target, short_interest, long_interest], dim=-1)
        )
        return short_interest * weight, long_interest * (1 - weight), weight
