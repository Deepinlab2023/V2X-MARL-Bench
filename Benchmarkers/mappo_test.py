import numpy as np
import torch as th
import torch.nn.functional as F

from Environment.environment import Environ

device = th.device("cuda" if th.cuda.is_available() else "cpu")

class MAPPOtester:
    @staticmethod
    def test_MAPPO(policy, params):
        """
        Note: MAPPO feature_pruning affects the critic during training, not the actor policy.
        Testing here is greedy action selection from the actor, consistent with prior tester.
        """
        tester = MAPPOtester(policy, params)
        return tester.test()

    # ==========================
    #   INIT / SETUP
    # ==========================
    def __init__(self, policy, params):
        self.policy = policy
        self.params = params
        self.env = Environ(params.env_params)

    # ==========================
    #   MAIN TEST LOOP
    # ==========================
    def test(self):
        p = self.params
        policy = self.policy

        policy.eval()

        test_rewards = np.zeros(p.num_test_episodes, dtype=np.float32)

        n_steps = getattr(self.env, "n_step_per_episode", p.n_step_per_episode)

        for i in range(p.num_test_episodes):
            test_rewards[i] = float(self._run_single_episode(i, n_steps))

        policy.train()
        return float(np.mean(test_rewards))

    # ==========================
    #   SINGLE EPISODE ROLLOUT
    # ==========================
    @th.no_grad()
    def _run_single_episode(self, idx: int, n_steps: int) -> float:
        p = self.params
        env = self.env
        actor = self.policy

        total_reward = 0.0

        test_data = p.test_data_list[idx % len(p.test_data_list)]
        env.train_data = test_data
        env.new_random_game()

        for t in range(n_steps):
            if getattr(p, "fast_fading_enabled", False):
                env._renew_fast_fading()

            rra = np.zeros((p.n_agent, 1, 2), dtype=np.int32)

            # FO state (NFIG/SIG) shared for all agents; POSIG uses per-agent obs
            if p.task_type != "POSIG":
                state_np = env.get_state(0, t)
                state = th.tensor(state_np, dtype=th.float32, device=device).squeeze()
            else:
                state = None

            for a in range(p.n_agent):
                agent_id = F.one_hot(
                    th.tensor(a, device=device),
                    num_classes=p.n_agent,
                ).float()

                if p.task_type == "POSIG":
                    obs_np = env.get_state(a, t)
                    obs = th.tensor(obs_np, dtype=th.float32, device=device).squeeze()
                    logits = actor(obs, agent_id)
                else:
                    logits = actor(state, agent_id)

                # # Greedy action (before)
                # action_id = int(th.argmax(logits, dim=-1).item())

                # # Stochastic policy with action masking (now)
                if p.action_masking:
                    action, _, _ = actor.action_sampler(logits, env.queue.flatten()[a])
                else:
                    action, _, _ = actor.action_sampler(logits)
                action_id = int(action.item())

                sc, pw = env.map_action_to_rra(action_id, a)
                rra[a, 0, 0] = sc
                rra[a, 0, 1] = pw

            global_reward, done = env.step(rra, t)
            total_reward += float(global_reward[0, 0])

            if done:
                break

        return total_reward