import numpy as np
import tqdm


# Calcul le gain d'une trajectoire à partir d'une itération et de la liste des récompenses
def compute_G(rewards, step, gamma):
    gammas = np.array([gamma ** i for i in range(len(rewards) - step)])
    res = np.array(rewards[step:]) * gammas
    res = np.sum(res)

    return res


# On initialise la table de Q avec des valeurs nulles pour tous les états
def initialize_q_table(state_space, action_space):
    Qtable = np.zeros((state_space, action_space))  # Q = (n_états, n_actions)
    return Qtable


# Algorithme pour évaluer une table Q sur un environnement donné
def evaluate_agent(env, max_steps, n_eval_episodes, Q):

    episode_rewards = []

    for episode in tqdm.tqdm(range(n_eval_episodes)):

        state = env.reset()
        done = False
        total_rewards_ep = 0

        for _ in range(max_steps):

            # Plus de epsilon
            action = np.argmax(Q[state])
            new_state, reward, done, info = env.step(action)
            total_rewards_ep += reward

            if done:
                break
            state = new_state

        episode_rewards.append(total_rewards_ep)
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)

    return mean_reward, std_reward
