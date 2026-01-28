import math

class IDQLparameters:
    def __init__(self):

        self.num_trials = 1
        self.training_episodes = 30000
        self.batch_size = 32
        self.hidden_dim = 128
        self.gamma = 0.99
        self.value_dim = 1
        self.tau = 0.03
        self.test_interval = 300
        self.num_test_episodes = 9
        self.num_training_iterations = self.training_episodes
        self.no_sharing = True
