import pygame
import sys
from game import Game
import torch

game = Game()


# Boucle principale
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            pygame.quit()
            sys.exit()

    game.update()
    if game.display_active:
        game.draw()
    game.clock.tick(game.FPS)
    pygame.display.flip()