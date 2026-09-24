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
        # Update stabilization (IPPO and MAPPO). Defaults keep the original behaviour; see
        # Configuration/experimental/ for a validated setting.
        # KL early stopping: before each actor step, if the minibatch's approx KL to the
        # rollout policy exceeds 1.5 * target_kl, skip that step and end the update. None = off.
        self.target_kl = None
        # Actor learning-rate schedule: "constant", or "adaptive" (KL-based, as in rsl_rl /
        # Rudin et al. 2021): after each update, lr /= 1.5 if its max minibatch KL > 2*target_kl,
        # lr *= 1.5 if < target_kl/2, clamped to [alpha/100, alpha]. "adaptive" needs target_kl.
        self.lr_schedule = "constant"
        # Standard PPO details (OpenAI Baselines ppo2 / CleanRL use eps=1e-5, max_grad_norm=0.5).
        # Defaults: PyTorch Adam eps and no gradient clipping.
        self.adam_eps = 1e-8
        self.max_grad_norm = None
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