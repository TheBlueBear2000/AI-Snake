import torch
import torch.nn as nn
import torch.optim as optim
import os

import snakeGame

N_INPUTS = 20
N_OUTPUTS = 4

GAMMA = 0.99  # discount factor
GAE_LAMBDA = 0.95  # generalised advantage estimation (λ)
LEARNING_RATE_ACTOR = 3e-4
TRAINING_EPS = 60
CHECKPOINT_EVERY = 5
STEPS_PER_EP = 200
VER = 0


class SnakeNet(nn.Module):
    """
    Input  : 20 features
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


def compute_gae(rewards, values, dones, next_value, gamma=GAMMA, lam=GAE_LAMBDA):
    advantages = []
    gae = 0.0
    next_val = next_value
    for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
        delta = r + gamma * next_val * (1 - d) - v
        gae = delta + gamma * lam * (1 - d) * gae
        advantages.insert(0, gae)
        next_val = v
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


def ppo_update():
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE_ACTOR)


def train_rl(model, episodes):
    for ep in range(episodes):
        steps = 0
        rewards_buffer = []
        values_buffer = []
        dones_buffer = []

        while steps < STEPS_PER_EP:
            iterations, new_values, new_rewards, next_val = snakeGame.gameLoop(
                steps=STEPS_PER_EP, startStep=steps
            )
            steps += iterations
            values_buffer += new_values
            rewards_buffer += new_rewards
            dones_buffer += ([0] * (iterations - 1)) + [1]

        advantages, returns = compute_gae(
            rewards_buffer, values_buffer, dones_buffer, next_val
        )
        advs_t = torch.tensor(advantages, dtype=torch.float32).to(device)
        rets_t = torch.tensor(returns, dtype=torch.float32).to(device)

        # Save checkpoints
        if ep % CHECKPOINT_EVERY == 0 and ep != episodes:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "episode": ep,
                    "n_inputs": N_INPUTS,
                    "n_outputs": N_OUTPUTS,
                },
                f"models/{VER}/ep_{ep}.pth",
            )

    # Final save
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "episode": ep,
            "n_inputs": N_INPUTS,
            "n_outputs": N_OUTPUTS,
        },
        f"models/{VER}/final_ep_{ep}.pth",
    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not (os.path.exists(f"models/{VER}/model_base.pth")):  # Check for file existence
        print("No model_base.pth found, creating random")
        model = SnakeNet().to(device)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "n_inputs": N_INPUTS,
                "n_outputs": N_OUTPUTS,
            },
            f"models/{VER}/model_base.pth",
        )
    else:
        ckpt = torch.load(f"models/{VER}/model_base.pth", map_location=device)
        model = SnakeNet().to(device)
        model.load_state_dict(ckpt["model_state_dict"])
    train_rl(model, TRAINING_EPS)
