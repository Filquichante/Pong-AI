import pygame
import sys
import random
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
        
        self.gamma = 0.998  # Discount factor
        self.log_probs = []
        self.rewards = []
        self.loss = torch.tensor(0)

    def prepare_training(self):
        self.optimizer = optim.Adam(self.game.agent.parameters(), lr=0.1)




    def update(self):
        self.state = torch.Tensor([self.game.left_paddle.rect.centery / self.game.HEIGHT,
                       self.game.right_paddle.rect.centery / self.game.HEIGHT,
                       self.game.ball.rect.centery / self.game.HEIGHT,
                       self.game.ball.rect.centerx / self.game.WIDTH,
                       self.game.ball.speed_x / self.game.BALL_SPEED,
                       self.game.ball.speed_y / self.game.BALL_SPEED])
        

        # Obtenir les probabilités des actions
        if self.game.epsilon_AI < random.random():
            preds = self.game.agent(self.state)
        else:
            preds = torch.tensor([0.5, 0.5])
        
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
    def end_of_episode(self, final_reward, last_touching_frame):

        last_touching_frame = self.game.frame_number if last_touching_frame == 0 else last_touching_frame
        self.rewards[last_touching_frame] += final_reward  # Récompense finale de l'épisode
        self.returns = []
        self.final_return = 0
        for reward in reversed(self.rewards):
            self.final_return = reward + self.gamma * self.final_return
            self.returns.append(self.final_return)
        self.returns.reverse()
        self.returns = torch.tensor(self.returns)

        # Compute weighted log probabilities
        weighted_log_probs = torch.stack(self.log_probs) * self.returns
        
        # Compute the loss as the negative sum of weighted log probabilities
        self.loss = -torch.sum(weighted_log_probs)


        self.optimizer.zero_grad()
        self.loss.backward()
        torch.nn.utils.clip_grad_norm_(self.game.agent.parameters(), max_norm=1.0)  # Clip des gradients
        self.optimizer.step()

        # Réinitialiser les listes pour le prochain épisode
        self.log_probs = []
        self.rewards = []
