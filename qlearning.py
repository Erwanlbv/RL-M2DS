import sys
import numpy as np
import tqdm

from tool_functions import compute_G


# On initialise la table de Q avec des valeurs nulles pour tous les états
def initialize_q_table(state_space, action_space):
    Qtable = np.zeros((state_space, action_space))  # Q = (n_états, n_actions)
    return Qtable


# Algorithme de Q-learning, version TD(0)
def q_learning(n_training_episodes, max_steps, min_epsilon, max_epsilon, decay_rate, gamma, learning_rate, env, Qtable):

    all_traj_rewards = []
    # Boucle sur les épisodes
    for episode in tqdm.tqdm(range(n_training_episodes)):

        state = env.reset()
        done = False
        rewards = []

        # On réduit epsilon pour qu'il y ait de moins en moins d'exploration
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(- decay_rate * episode)
        if (epsilon % 500 == 0):
            print('Epsilon :', epsilon)
            sys.stdout.flush()

        # Trajectoire de maximum max_steps itérations
        for _ in range(max_steps):

            # L'action At est choisie avec une politique eps-greedy
            expl_threshold = np.random.rand()

            if epsilon > expl_threshold:
                action = env.action_space.sample()
            else:
                action = np.argmax(Qtable[state])

            # On prend l'action At et observe R_t+1 et S_t+1
            new_state, reward, done, info = env.step(action)
            rewards.append(reward)

            # Mise à jour de Q version Q-LEARNING  Q(s,a) = Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            Qtable[state, action] = Qtable[state, action] + learning_rate * (reward + gamma * np.max(Qtable[new_state]) - Qtable[state, action])

            if done:
                break

            # On met à jour l'état
            state = new_state

        # On stocke le gain de la trajectoire
        all_traj_rewards.append(compute_G(rewards, step=0, gamma=gamma))

    return Qtable, all_traj_rewards
