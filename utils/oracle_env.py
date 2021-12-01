import numpy as np

class OracleEnv:
    def __init__(self, training_environment, test_model_length, test_padding):
        self.env = training_environment
        self.test_model_length = test_model_length
        self.test_padding = test_padding

    def step(self, obs, act, deterministic=False):
        assert len(obs.shape) == len(act.shape)
        self.env.reset()

        batchsize = obs.shape[0]
        next_obs_list = []
        rew_list = []
        tem_list = []
        # print('self.env.model.nq: ', self.env.model.nq)
        for i in range(batchsize):
            qpos = obs[i][:self.test_model_length]
            qvel = obs[i][self.test_model_length:]
            if self.test_padding > 0:
                qpos = np.concatenate([[0,], qpos], axis=0)
            self.env.set_state(qpos, qvel)
            next_observation, reward, terminal, _ = self.env.step(act[i])
            next_obs_list.append(next_observation)
            rew_list.append([reward])
            tem_list.append([terminal])
            ### next qpos
            next_qpos = next_observation[:2]
        next_obs = np.array(next_obs_list)
        rewards = np.array(rew_list)
        terminals = np.array(tem_list)

        return next_obs, rewards, terminals, {}
