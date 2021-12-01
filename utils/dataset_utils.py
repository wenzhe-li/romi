import d4rl
import numpy as np
from numpy.linalg import norm


def antmaze_timeout(dataset):
    threshold = np.mean(norm(dataset['observations'][1:, :2] - dataset['observations'][:-1, :2], axis=1))
    print('threshold', threshold)
    for i in range(dataset['observations'].shape[0]):
        dataset['timeouts'][i] = False
    for i in range(dataset['observations'].shape[0] - 1):
        gap = norm(dataset['observations'][i + 1, :2] - dataset['observations'][i, :2])
        if gap > threshold * 10:
            dataset['timeouts'][i] = True
    return dataset


def processed_qlearning_dataset(env, env_name, timeout_frame=False, done_frame=False):
    dataset = env.get_dataset()
    if 'antmaze' in env_name: # handle wrong timeout
        dataset = antmaze_timeout(dataset)

    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    for i in range(N-1):
        obs = dataset['observations'][i].astype(np.float32)
        new_obs = dataset['observations'][i+1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])

        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)
        if (not timeout_frame) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue
        if (not done_frame) and done_bool:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        episode_step += 1

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
    }


def processed_sequence_dataset(env, env_name, timeout_frame=False, done_frame=False):
    dataset = env.get_dataset()
    if 'antmaze' in env_name:  # handle wrong timeout
        dataset = antmaze_timeout(dataset)

    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    traj_list = []


    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    for i in range(N - 1):
        obs = dataset['observations'][i].astype(np.float32)
        new_obs = dataset['observations'][i + 1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])

        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)
        if (not timeout_frame) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            if episode_step > 0:
                traj = {
                    'observations': np.array(obs_),
                    'actions': np.array(action_),
                    'next_observations': np.array(next_obs_),
                    'rewards': np.array(reward_),
                    'terminals': np.array(done_),
                }
                traj_list.append(traj)
                obs_ = []
                next_obs_ = []
                action_ = []
                reward_ = []
                done_ = []

            episode_step = 0
            continue
        if (not done_frame) and done_bool:
            # Skip this transition and don't apply terminals on the last step of an episode
            if episode_step > 0:
                traj = {
                    'observations': np.array(obs_),
                    'actions': np.array(action_),
                    'next_observations': np.array(next_obs_),
                    'rewards': np.array(reward_),
                    'terminals': np.array(done_),
                }
                traj_list.append(traj)
                obs_ = []
                next_obs_ = []
                action_ = []
                reward_ = []
                done_ = []

            episode_step = 0
            continue
        if done_bool or final_timestep:
            obs_.append(obs)
            next_obs_.append(new_obs)
            action_.append(action)
            reward_.append(reward)
            done_.append(done_bool)
            episode_step += 1
            if episode_step > 0:
                traj = {
                    'observations': np.array(obs_),
                    'actions': np.array(action_),
                    'next_observations': np.array(next_obs_),
                    'rewards': np.array(reward_),
                    'terminals': np.array(done_),
                }
                traj_list.append(traj)
                obs_ = []
                next_obs_ = []
                action_ = []
                reward_ = []
                done_ = []

            episode_step = 0
            continue

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        episode_step += 1

    if episode_step > 0:
        traj = {
            'observations': np.array(obs_),
            'actions': np.array(action_),
            'next_observations': np.array(next_obs_),
            'rewards': np.array(reward_),
            'terminals': np.array(done_),
        }
        traj_list.append(traj)

    return traj_list


