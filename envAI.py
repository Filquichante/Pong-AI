import pygame
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class EnvAI:
    def __init__(self, game):
        self.game = game
        self.state = torch.Tensor([self.game.left_paddle.rect.centery / self.game.HEIGHT,
                       self.game.right_paddle.rect.centery / self.game.HEIGHT,
                       self.game.ball.rect.centery / self.game.HEIGHT,
                       self.game.ball.rect.centerx / self.game.WIDTH,
                       self.game.ball.speed_x / self.game.BALL_SPEED,
                       self.game.ball.speed_y / self.game.BALL_SPEED])
        
        self.gamma = 0.99  # Discount factor
        self.log_probs = []
        self.rewards = []
        self.loss = torch.tensor(0)

    def prepare_training(self):
        self.optimizer = optim.Adam(self.game.agent.parameters(), lr=0.001)




    def update(self):
        self.state = torch.Tensor([self.game.left_paddle.rect.centery / self.game.HEIGHT,
                       self.game.right_paddle.rect.centery / self.game.HEIGHT,
                       self.game.ball.rect.centery / self.game.HEIGHT,
                       self.game.ball.rect.centerx / self.game.WIDTH,
                       self.game.ball.speed_x / self.game.BALL_SPEED,
                       self.game.ball.speed_y / self.game.BALL_SPEED])
        
        # Obtenir les probabilités des actions
        preds = self.game.agent(self.state)
        
        # Choisir une self.action en fonction des probabilités
        self.distrib = torch.distributions.Categorical(preds)
        self.action = self.distrib.sample()


        if self.action == 0:
            self.game.right_paddle.move("up", self.game.HEIGHT)
        elif self.action == 1:
            self.game.right_paddle.move("down", self.game.HEIGHT)


        if self.game.trainingAI:
            self.log_informations()


    ## Fonction pour enregistrer les informations de chaque frame
    def log_informations(self):
        self.log_probs.append(self.distrib.log_prob(self.action))
        self.rewards.append(self.game.passive_reward) # Récompense négative pour chaque frame
    

    ## Fonction pour enregistrer la récompense finale de chaque frame de l'épisode, puis la fonction d'erreur
    def end_of_episode(self, final_reward):

        


        self.returns = []
        for r in reversed(self.rewards):
            final_reward = r + self.gamma * final_reward
            self.returns.append(final_reward)
        self.returns.reverse()

        self.returns = torch.tensor(self.returns)

        # Compute weighted log probabilities
        weighted_log_probs = torch.stack(self.log_probs) * self.returns / len(self.returns)
        
        # Compute the loss as the negative sum of weighted log probabilities
        self.loss = -torch.sum(weighted_log_probs)


        self.optimizer.zero_grad()
        self.loss.backward()
        torch.nn.utils.clip_grad_norm_(self.game.agent.parameters(), max_norm=1.0)  # Clip des gradients
        self.optimizer.step()

        # Réinitialiser les listes pour le prochain épisode
        self.log_probs = []
        self.rewards = []
