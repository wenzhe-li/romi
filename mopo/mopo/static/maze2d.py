import numpy as np

class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        bs = obs.shape[0]
        done = np.zeros((bs, 1)).astype(np.bool)

        return done
