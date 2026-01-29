class QMIXparameters:
    def __init__(self):
        # Training
        self.num_trials = 1
        self.training_episodes = 30000
        self.batch_size = 64
        self.gamma = 0.9
        self.tau = 0.005

        # Testing
        self.test_interval = self.training_episodes / 100
        self.num_test_episodes = 9

        # Agent network
        self.hidden_dim = 128

        # Mixer network
        self.hyper_hidden_dim = 128
        self.qmix_hidden_dim = 32
        self.two_hyper_layers = True

        # Learning rates
        self.agent_lr = 1e-5
        self.mixer_lr = 1e-6

        # Replay buffer
        self.memory_capacity = 10000

        # Environment constraints
        self.force_nt_when_empty = True

