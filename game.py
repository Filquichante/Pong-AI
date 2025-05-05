import pygame
import sys
import math
import torch
from paddle import Paddle
from ball import Ball
from envAI import EnvAI
from agent import Agent
from BotOp import BotOP

class Game():
    def __init__(self):


        self.display_active = False
        self.trainingAI = True
        self.AI_touched_the_ball = False


        # Constantes
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)

        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 10, 100
        self.BALL_RADIUS = 7

        self.BALL_SPEED = 12
        self.PADDLE_SPEED = self.BALL_SPEED // 1.5

        self.FPS = 10000 if not self.display_active else 60

        self.passive_reward = 0  # Récompense passive pour chaque frame
        self.botOP_error = 70  # Erreur du botOP

        # Initialisation de Pygame
        pygame.init()

        # Fenêtre plein écran
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self.WIDTH, self.HEIGHT = self.screen.get_size()
        self.alpha_angle = math.atan(self.WIDTH/self.HEIGHT)
        pygame.display.set_caption("Pong")

        

        self.clock = pygame.time.Clock()

        # Objets
        self.left_paddle = Paddle(50, (self.HEIGHT - self.PADDLE_HEIGHT) // 2, self.PADDLE_WIDTH, self.PADDLE_HEIGHT, self.PADDLE_SPEED, self)
        self.right_paddle = Paddle(self.WIDTH - 50 - self.PADDLE_WIDTH, (self.HEIGHT - self.PADDLE_HEIGHT) // 2, self.PADDLE_WIDTH, self.PADDLE_HEIGHT, self.PADDLE_SPEED, self)
        self.ball = Ball(self.WIDTH, self.HEIGHT, self.BALL_RADIUS, self.BALL_SPEED, self)
        self.envAI = EnvAI(self)
        self.load_agent(Agent, "agent.pth")
        self.envAI.prepare_training()

        # Création de l'instance du BotOP
        self.bot = BotOP(50,(self.HEIGHT - self.PADDLE_HEIGHT)//2,self.PADDLE_WIDTH,self.PADDLE_HEIGHT,self.PADDLE_SPEED,self)

        # Scores
        self.left_score = 0
        self.right_score = 0
        self.font = pygame.font.Font(None, 74)

    def update(self):
        # Mouvements
        #self.left_paddle.move_towards(self.ball.rect.centery, self.HEIGHT)
        self.envAI.update()
        
        #self.right_paddle.move_manual(pygame.K_UP, pygame.K_DOWN, self.HEIGHT)

        # Mouvements de la balle
        self.ball.move(self.WIDTH, self.HEIGHT)

        # Vérification des collisions
        self.ball.check_collision(self.left_paddle, self.right_paddle, self.BALL_SPEED)

        # Update de la prédiction du botOP
        self.bot.update()

        # Vérification des scores
        if self.ball.rect.left <= 0:
            self.right_score += 1
            self.ball.reset(self.WIDTH, self.HEIGHT)
            if self.trainingAI and self.AI_touched_the_ball:
                self.envAI.end_of_episode(10)
                self.manage_episode_results()
                self.AI_touched_the_ball = False

        elif self.ball.rect.right >= self.WIDTH:
            self.left_score += 1
            self.ball.reset(self.WIDTH, self.HEIGHT)
            
            if self.trainingAI:
                self.envAI.end_of_episode(-1)
                self.manage_episode_results()
                self.AI_touched_the_ball = False

    def draw(self):
        # Affichage
        self.screen.fill(self.BLACK)

        # Raquettes
        self.left_paddle.draw(self.screen, self.WHITE)
        self.right_paddle.draw(self.screen, self.WHITE)

        # Balle
        self.ball.draw(self.screen, self.WHITE)

        # Ligne centrale
        pygame.draw.aaline(self.screen, self.WHITE, (self.WIDTH // 2, 0), (self.WIDTH // 2, self.HEIGHT))

        # Scores
        left_text = self.font.render(str(self.left_score), True, self.WHITE)
        right_text = self.font.render(str(self.right_score), True, self.WHITE)
        loss_text = self.font.render("Loss: " + str(self.envAI.loss.item()), True, self.WHITE)
        self.screen.blit(loss_text, (self.WIDTH // 2 - loss_text.get_width() // 2, 20))
        self.screen.blit(left_text, (self.WIDTH // 4 - left_text.get_width() // 2, 20))
        self.screen.blit(right_text, (3 * self.WIDTH // 4 - right_text.get_width() // 2, 20))

    def save_agent(self, path="agent.pth"):
        torch.save(self.agent.state_dict(), "agent.pth")
        print("Agent enregistré")


    def load_agent(self, agent_class, path="agent.pth"):
        self.agent = agent_class(self.envAI)

        try:
            self.agent.load_state_dict(torch.load(path))
            if self.trainingAI:
                self.agent.train()
            else:
                self.agent.eval()

        except FileNotFoundError:
            print("Le fichier n'existe pas. Un nouvel agent sera créé.")


    def manage_episode_results(self):
        # Affichage des résultats de l'épisode
        print("----------------------------------")
        print("Score gauche :", self.left_score)
        print("Score droite :", self.right_score)
        print("Loss :", self.envAI.loss.item())
        print("----------------------------------")

        if (self.left_score%100 == 0 and self.left_score>0) or (self.right_score%100 == 0 and self.right_score>0):
            self.save_agent("agent.pth")
            print("Agent enregistré tous les 100 scores !")