import sys
import numpy as np
import tqdm

from tool_functions import compute_G


# Algorithme de Sarsa, version TD(0)
def sarsa(n_training_episodes, max_steps, min_epsilon, max_epsilon, decay_rate, gamma, learning_rate, env, Qtable):

    all_traj_rewards = []
    # Boucle sur les épisodes
    for episode in tqdm.tqdm(range(n_training_episodes)):

        state = env.reset()
        done = False
        rewards = []

        # On réduit epsilon pour qu'il y ait de moins en moins d'exploration
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(- decay_rate * episode)
        if (epsilon % 500 == 0):
            sys.stdout.flush()
            print('Epsilon :', epsilon)

        # Trajectoire de maximum max_steps itérations
        for _ in range(max_steps):

            # Politique epsillon greedy pour le choix de At
            expl_threshold = np.random.rand()
            if epsilon > expl_threshold:
                action = env.action_space.sample()
            else:
                action = np.argmax(Qtable[state])

            # On prend l'action At et observe R_t+1 et S_t+1
            new_state, reward, done, info = env.step(action)
            rewards.append(reward)

            # Politique epsillon-greedy pour le choix de A_t+1 également 
            expl_threshold = np.random.rand()
            if epsilon > expl_threshold:
                new_action = env.action_space.sample()
            else:
                new_action = np.argmax(Qtable[new_state])

            # Mise à jour de Q version SARSA  Q(s,a) = Q(s,a) + lr [R(s,a) + gamma *  Q(s',a') - Q(s,a)]
            Qtable[state, action] = Qtable[state, action] + learning_rate * (reward + gamma * Qtable[new_state, new_action] - Qtable[state, action])

            if done:
                break

            # On met à jour l'état
            state = new_state

        # On stocke le gain de la trajectoire
        all_traj_rewards.append(compute_G(rewards, step=0, gamma=gamma))

    return Qtable, all_traj_rewards
