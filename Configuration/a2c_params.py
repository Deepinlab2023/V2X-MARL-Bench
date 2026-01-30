import math

class A2Cparameters:
    def __init__(self):

        self.num_trials = 1
        self.training_episodes = 100000
        self.batch_size = 8
        self.alpha = 0.0005
        self.beta = 0.0005
        self.actor_hidden_dim = 128
        self.critic_hidden_dim = 128
        self.gamma = 0.99
        self.value_dim = 1
        self.tau = 0.01
        self.test_interval = self.training_episodes / 100
        self.num_test_episodes = 9
        self.num_training_iterations = math.ceil(self.training_episodes / self.batch_size)
        
        # Function Toggles
        # no_sharing:
        #   - True  → independent actors/critics per agent
        #   - False → parameter sharing
        # You *must* keep this False for POSIG
        self.no_sharing = False      # default: FO setting, safe for NFIG/SIG
        self.action_masking = True
        self.adv_normalization = True

        # Partial Observability Toggles
        self.rnn = False
        self.prev_action_input = False # Only run this when self.rnn = True