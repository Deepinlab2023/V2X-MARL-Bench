import numpy as np
import torch as th
import torch.nn.functional as F

device = th.device("cuda" if th.cuda.is_available() else "cpu")


class MAPPOtester:

    @staticmethod
    def test_MAPPO(actor, params):
        """
        Fully aligned with new environment API.
        Uses runner-owned env.
        """

        env = params.env
        task_type = params.task_type
        n_agent = params.n_agent

        actor.eval()

        test_rewards = np.zeros(params.num_test_episodes, dtype=np.float32)

        # Prefer env-defined episode length
        n_steps = getattr(env, "n_step_per_episode",
                          params.n_step_per_episode)

        for i in range(params.num_test_episodes):

            total_reward = 0.0

            test_data = params.test_data_list[i % len(params.test_data_list)]

            # -------- NEW ENV INIT --------
            env.train_data = test_data
            env.new_random_game()

            for t in range(n_steps):

                # New required shape
                rra = np.zeros((n_agent, 1, 2), dtype=np.int32)

                # Global state for NFIG/SIG
                if task_type != "POSIG":
                    state_np = env.get_state(0, t)
                    state = th.tensor(state_np, dtype=th.float32).squeeze().to(device)

                for a in range(n_agent):

                    agent_id = F.one_hot(
                        th.tensor(a),
                        num_classes=n_agent
                    ).float().to(device)

                    if task_type == "POSIG":
                        obs_np = env.get_state(a, t)
                        obs = th.tensor(obs_np, dtype=th.float32).squeeze().to(device)
                        logits = actor(obs, agent_id)
                    else:
                        logits = actor(state, agent_id)

                    # Greedy action
                    action_id = int(th.argmax(logits, dim=-1).item())

                    sc, pw = env.map_action_to_rra(action_id, a)

                    rra[a, 0, 0] = sc
                    rra[a, 0, 1] = pw

                global_reward, done = env.step(rra, t)

                total_reward += float(global_reward[0, 0])

                if done:
                    break

            test_rewards[i] = total_reward

        actor.train()

        return float(np.mean(test_rewards))