class IDQLparameters:
    DERIVED_FIELDS = ("test_interval",)

    def __init__(self):
        # Training
        self.num_trials = 1
        self.training_episodes = 30000
        self.batch_size = 64
        self.gamma = 0.9
        self.tau = 0.005
        self.lr = 3e-5

        # Testing
        self.num_test_episodes = 9

        # Network
        self.hidden_dim = 128

        # Replay buffer
        self.memory_capacity = 10000

        # Hysteretic Q-learning
        self.hysteretic_high_lr = 1.0
        self.hysteretic_low_lr = 0.2

        # Environment constraints
        self.force_nt_when_empty = False

        self._derive()

    def _derive(self):
        self.test_interval = self.training_episodes / 100

