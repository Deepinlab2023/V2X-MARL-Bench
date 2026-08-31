# Configuration/ppo_params.py

import math


class PPOparameters:
    """
    Notes:
      - MAPPO-only toggle: feature_pruning for Partial Observability
      - DERIVED_FIELDS are recomputed by _derive() and cannot be set via JSON
    """

    DERIVED_FIELDS = ("test_interval", "num_training_iteration")

    def __init__(self):
        # Experiment control
        self.num_trials = 1
        self.training_episodes = 20000
        self.batch_size = 128
        # Optimizer / GAE
        self.alpha = 0.0006
        self.beta = 0.0009
        self.lam = 0.95
        self.gamma = 0.9
        # Network sizes
        self.actor_hidden_dim = 128
        self.critic_hidden_dim = 128
        self.value_dim = 1
        # PPO hyperparams
        self.entropy_coef = 0.01
        self.eps_clip = 0.2
        self.num_mini_batches = 4
        self.epochs = 10
        # Derived scheduling
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

        self._derive()

    def _derive(self):
        self.test_interval = self.training_episodes / 100
        self.num_training_iteration = math.ceil(self.training_episodes / self.batch_size)