import pygame
import sys
from game import Game
import torch

game = Game()


# Boucle principale
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game.save_agent(game.agent)
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            game.save_agent(game.agent)
            pygame.quit()
            sys.exit()

    game.update()
    if game.active:
        game.draw()
    game.clock.tick(game.FPS)
    pygame.display.flip()