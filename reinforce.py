from collections import deque
import os
import numpy as np

import torch
import torch.nn as nn

from torch.distributions import Categorical


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Policy(nn.Module):
    def __init__(self, game_name, s_size, a_size, h_size):
        super(Policy, self).__init__()

        self.game_name = game_name
        self.fc_1 = nn.Linear(s_size, h_size)
        self.fc_2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = torch.relu(self.fc_1(x))
        x = self.fc_2(x)

        out = torch.softmax(x, dim=1)
        return out

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()

        return action.item(), m.log_prob(action)


def reinforce(policy, optimizer, n_training_episodes, max_steps, gamma, env, check_every):

    # Passer par une liste deque permet d'accélérer le temps de calcul lors de l'entraînement
    scores_deque = deque(maxlen=100)
    scores = []
    avg_scores = []
    std_scores = []
    metric_scores = [0]

    for episode in range(1, n_training_episodes+1):
        saved_log_probs = []
        rewards = []
        state = env.reset()

        # On se déplace dans l'environnement en suivant notre politique (une trajectoire)
        for t in range(max_steps):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)

            if done:
                break

        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        # Returns contiendra les gains de chaque itération
        returns = deque(maxlen=max_steps)
        # Nombre d'itérations effectuées durant la trajectoire qui vient d'avoir lieu
        n_steps = len(rewards)

        # Calcul du gain pour chaque itérations de la trajectoire
        for t in range(n_steps)[::-1]:
            G = (returns[0] if len(returns) > 0 else 0)
            returns.appendleft(gamma * G + rewards[t])

        # Standardiser les gains permet de rendre l'entraînement plus stable
        eps = np.finfo(np.float32).eps.item()
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)  # Le eps évite la division par 0

        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * disc_return)
        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if episode % check_every == 0:
            # Calcul des performances du modèle      
            epi_mean_score, epi_std_score = np.mean(scores_deque), np.std(scores_deque)
            avg_scores.append(epi_mean_score)
            std_scores.append(epi_std_score)

            metric_score = epi_mean_score - epi_std_score
            if metric_score > np.max(metric_scores):
                metric_scores.append(epi_mean_score - epi_std_score)
                print('Episode {}\tAverage Score: {:.2f}, Standard Deviation Score: {:.2f}'.format(episode, epi_mean_score, epi_std_score))
                print('Métrique :', (epi_mean_score - epi_std_score).round(2))

                model_name = 'reinforce_MC_' + policy.game_name + '_' + str(metric_scores[-1].round(2))
                save_path = os.path.join('/Users/erwan/Programmes/M2DS RL/Basic Algs/best_models/', model_name)                
                print('\t Nouveau meilleur score! Sauvegarde des poids du modèle à ' + save_path)
                torch.save(policy.state_dict(), save_path)

    print('Terminé !')
    return policy, scores
