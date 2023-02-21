import numpy as np
import tqdm


# Algorithme de Monte Carlo Incrémental
def every_visit_mc(n_training_episodes, max_steps, min_epsilon, max_epsilon, decay_rate, gamma, learning_rate, env, Qtable):

    all_traj_rewards = []

    for episode in tqdm.tqdm(range(n_training_episodes)):

        state = env.reset()
        visited_spaces = []
        rewards = []

        done = False

        # On réduit epsilon pour qu'il y ait de moins en moins d'exploration
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        lr = 0.01 + (learning_rate - 0.01) * np.exp(- 0.01 * episode)

        # On génère un épisode
        for step in range(max_steps):

            # Politique epsillon greedy pour le choix de At
            expl_threshold = np.random.rand()
            if epsilon > expl_threshold:
                action = env.action_space.sample()
            else:
                action = np.argmax(Qtable[state])

            # On prend l'action At et observe R_t+1 et S_t+1
            new_state, reward, done, info = env.step(action)

            # On sauvegarde le passage dans St, l'action prise At et la récompense obtenue
            visited_spaces.append([state, action, reward])

            if done:
                break

            # On met à jour les états
            state = new_state

        # On met à jour la table avec l'épisode
        G = 0
        for i, [state, action, reward] in enumerate(visited_spaces[::-1]):
            G = gamma * G + reward

            # On met à jour la table
            # Mise à jour de Q version Incremental Every-visit MC  Q(S,a) = Q(S,a) + lr * [G - Q(S,A)] à chaque fois que (s, a) a été effectué durant la trajectoire
            Qtable[state, action] = Qtable[state, action] + lr * (G - Qtable[state, action])

        all_traj_rewards.append(G)

    return Qtable, all_traj_rewards
