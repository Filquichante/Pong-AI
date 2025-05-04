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
                       self.game.left_paddle.rect.centery / self.game.HEIGHT,
                       self.game.ball.rect.centery / self.game.HEIGHT,
                       self.game.ball.rect.centerx / self.game.WIDTH,
                       self.game.ball.speed_x / self.game.BALL_SPEED,
                       self.game.ball.speed_y / self.game.BALL_SPEED])





    def update(self):
        self.state = torch.Tensor([self.game.left_paddle.rect.centery / self.game.HEIGHT,
                       self.game.left_paddle.rect.centery / self.game.HEIGHT,
                       self.game.ball.rect.centery / self.game.HEIGHT,
                       self.game.ball.rect.centerx / self.game.WIDTH,
                       self.game.ball.speed_x / self.game.BALL_SPEED,
                       self.game.ball.speed_y / self.game.BALL_SPEED])
        
        # Obtenir les probabilités des actions
        preds = self.game.agent(self.state)
        
        # Choisir une action en fonction des probabilités
        action = torch.multinomial(preds, 1).item()
        print(preds, action)

        if action == 0:
            self.game.left_paddle.move("up", self.game.HEIGHT)
        elif action == 1:
            self.game.left_paddle.move("down", self.game.HEIGHT)
