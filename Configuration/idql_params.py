class IDQLparameters:
    def __init__(self):
        # Training
        self.num_trials = 1
        self.training_episodes = 30000
        self.batch_size = 64
        self.gamma = 0.9
        self.tau = 0.01
        self.lr = 1e-3

        # Epsilon-greedy schedule (matches idql_trainer.py:75-77)
        self.epsi_start = 1.0
        self.epsi_final = 0.01
        self.epsi_anneal_frac = 0.8   # anneal over first 80% of training_episodes

        # Testing
        self.test_interval = self.training_episodes / 100
        self.num_test_episodes = 9

        # Network
        self.hidden_dim = 128

        # Replay buffer
        self.memory_capacity = 1000

        # Hysteretic Q-learning
        self.hysteretic_high_lr = 1.0
        self.hysteretic_low_lr = 0.2

        # Environment constraints
        self.force_nt_when_empty = False

