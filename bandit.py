from dataclasses import dataclass
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# -----------------------------
# Configs
# -----------------------------
@dataclass
class ModelConfig:
    name: str
    cost: float
    true_weights: np.ndarray  # true linear weights for reward


@dataclass
class BanditConfig:
    epsilon: float = 0.1
    steps: int = 3000
    seed: int = 42
    cost_weight: float = 0.3
    lr: float = 0.05  # learning rate for weight updates


# -----------------------------
# Context generator (real-ish features)
# -----------------------------
def sample_context(rng: np.random.Generator) -> np.ndarray:
    # user_segment: 0..1
    user_segment = rng.random()
    # device_type: mobile=1, desktop=0
    device_type = rng.integers(2)
    # traffic_source: paid=1, organic=0
    traffic_source = rng.integers(2)
    # bias term
    bias = 1.0
    return np.array([bias, user_segment, device_type, traffic_source], dtype=float)


# -----------------------------
# Environment: contextual reward
# -----------------------------
class ContextualModelEnv:
    def __init__(self, models: List[ModelConfig], cfg: BanditConfig):
        self.models = models
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    def get_reward(self, model: ModelConfig, context: np.ndarray) -> float:
        # True reward: linear model + noise - cost penalty
        base = float(context @ model.true_weights)
        base = np.clip(base, 0.0, 1.0)
        noise = self.rng.normal(0, 0.05)
        reward = base + noise - self.cfg.cost_weight * model.cost
        return reward


# -----------------------------
# Contextual epsilon-greedy bandit
# -----------------------------
class ContextualBanditAgent:
    def __init__(self, models: List[ModelConfig], cfg: BanditConfig, rng: np.random.Generator):
        self.models = models
        self.cfg = cfg
        self.rng = rng
        self.n_arms = len(models)
        self.n_features = len(models[0].true_weights)
        # learned weights per arm
        self.weights = np.zeros((self.n_arms, self.n_features))
        self.pulls = np.zeros(self.n_arms, dtype=int)

    def predict_reward(self, arm: int, context: np.ndarray) -> float:
        return float(context @ self.weights[arm])

    def select_arm(self, context: np.ndarray) -> int:
        if self.rng.random() < self.cfg.epsilon:
            return self.rng.integers(self.n_arms)
        preds = np.array([self.predict_reward(a, context) for a in range(self.n_arms)])
        best_arms = np.flatnonzero(preds == preds.max())
        return int(self.rng.choice(best_arms))

    def update(self, arm: int, context: np.ndarray, reward: float):
        self.pulls[arm] += 1
        # simple gradient step towards reward
        pred = self.predict_reward(arm, context)
        error = reward - pred
        self.weights[arm] += self.cfg.lr * error * context


# -----------------------------
# Simulation
# -----------------------------
def run_simulation():
    cfg = BanditConfig()

    # true weights: [bias, user_segment, device_type, traffic_source]
    models = [
        ModelConfig("Model A (fast, cheap)", cost=0.2,
                    true_weights=np.array([0.55, 0.10, 0.05, 0.02])),
        ModelConfig("Model B (accurate, expensive)", cost=0.6,
                    true_weights=np.array([0.70, 0.20, 0.10, 0.05])),
        ModelConfig("Model C (context-specialized)", cost=0.4,
                    true_weights=np.array([0.60, 0.35, 0.05, 0.15])),
    ]

    env = ContextualModelEnv(models, cfg)
    agent = ContextualBanditAgent(models, cfg, env.rng)

    chosen_arms = []
    rewards = []
    cumulative_rewards = []

    for step in range(cfg.steps):
        context = sample_context(env.rng)
        arm = agent.select_arm(context)
        model = models[arm]

        reward = env.get_reward(model, context)
        agent.update(arm, context, reward)

        chosen_arms.append(arm)
        rewards.append(reward)
        cumulative_rewards.append(np.mean(rewards))

    return {
        "models": models,
        "chosen_arms": np.array(chosen_arms),
        "cumulative_rewards": np.array(cumulative_rewards),
        "weights": agent.weights,
        "pulls": agent.pulls,
    }


# -----------------------------
# Plot results
# -----------------------------
def plot_results(results):
    models = results["models"]
    cumulative_rewards = results["cumulative_rewards"]
    pulls = results["pulls"]

    steps = len(cumulative_rewards)
    x = np.arange(steps)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 1) Cumulative reward
    axes[0].plot(x, cumulative_rewards, color="#2a78d6")
    axes[0].set_title("Cumulative Reward (Contextual Bandit Routing)")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Avg Reward")
    axes[0].grid(True, alpha=0.3)

    # 2) Model selection frequency
    freqs = pulls / pulls.sum()
    axes[1].bar(
        [m.name for m in models],
        freqs,
        color=["#2a78d6", "#eb6834", "#4caf50"],
    )
    axes[1].set_title("Model Selection Frequency")
    axes[1].set_ylabel("Traffic Share")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].set_xticklabels([m.name for m in models], rotation=20, ha="right")

    plt.tight_layout()
    plt.show()


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    results = run_simulation()
    print("\nFinal learned weights per model:")
    for model, w, pulls in zip(results["models"], results["weights"], results["pulls"]):
        print(f"{model.name}: pulls={pulls}, weights={np.round(w, 3)}")

    plot_results(results)
