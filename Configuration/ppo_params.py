# Configuration/ppo_params.py

import math


class PPOparameters:
    """
    Notes:
      - MAPPO-only toggle: feature_pruning for Partial Observability
    """

    def __init__(self):
        # Experiment control
        self.num_trials = 1
        self.training_episodes = 100000
        self.batch_size = 256
        # Optimizer / GAE
        self.alpha = 0.0004
        self.beta = 0.0006
        self.lam = 0.95
        self.gamma = 0.99
        # Network sizes
        self.actor_hidden_dim = 128
        self.critic_hidden_dim = 128
        self.value_dim = 1
        # PPO hyperparams
        self.entropy_coef = 0.001
        self.eps_clip = 0.2
        self.num_mini_batches = 4
        self.epochs = 10
        # Derived scheduling
        self.test_interval = self.training_episodes / 100
        self.num_training_iteration = math.ceil(self.training_episodes / self.batch_size)
        self.num_test_episodes = 9

        # -------------------------
        # Function toggles
        # -------------------------
        self.popart = True
        self.critic_rescale = True  # only meaningful if popart=True
        self.action_masking = True

        # -------------------------
        # Partial observability toggles / misc
        # -------------------------
        self.prev_action_input = False
        self.individual_rewards = False
        # MAPPO-only toggles
        self.feature_pruning = True