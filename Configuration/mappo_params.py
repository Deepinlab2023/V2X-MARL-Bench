import math

class MAPPOparameters:
    def __init__(self):

        self.num_trials = 1
        self.training_episodes = 100000
        self.batch_size = 256
        self.alpha = 0.0004
        self.beta = 0.0006
        self.lam = 0.95
        
        self.actor_hidden_dim = 128
        self.critic_hidden_dim = 128
        
        self.gamma = 0.99            # Usually 0.99
        self.value_dim = 1
        self.entropy_coef = 0.001
        self.eps_clip = 0.2
        self.num_mini_batches = 4
        self.epochs = 10
        self.test_interval = self.training_episodes / 100
        self.num_training_iteration = math.ceil(self.training_episodes / self.batch_size)
        self.num_test_episodes = 9

        # Function Toggles
        self.popart = True
        self.critic_rescale = True # Only Works with PopArt
        self.action_masking = True

        # Partial Observability Toggles
        self.feature_pruning = True
        self.rnn = False
        self.prev_action_input = False
        self.individual_rewards = False
