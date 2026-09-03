import math

class A2Cparameters:
    DERIVED_FIELDS = ("test_interval", "num_training_iterations")

    def __init__(self):

        self.num_trials = 1
        self.training_episodes = 30000
        self.batch_size = 8
        self.alpha = 0.0002
        self.beta = 0.0002
        self.actor_hidden_dim = 128
        self.critic_hidden_dim = 128
        self.gamma = 0.9
        self.value_dim = 1
        self.tau = 0.01
        self.num_test_episodes = 9
        
        # Function Toggles
        # no_sharing:
        #   - True  → independent actors/critics per agent
        #   - False → parameter sharing
        # You *must* keep this False for POSIG
        self.no_sharing = True      # default: FO setting, safe for NFIG/SIG
        self.action_masking = True
        self.adv_normalization = False

        self._derive()

    def _derive(self):
        self.test_interval = self.training_episodes / 100
        self.num_training_iterations = math.ceil(self.training_episodes / self.batch_size)