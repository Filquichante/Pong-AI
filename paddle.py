import pygame
import sys


class Paddle:
    def __init__(self, x, y, PADDLE_WIDTH, PADDLE_HEIGHT, PADDLE_SPEED, game):
        self.rect = pygame.Rect(x, y, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.speed = PADDLE_SPEED

    def move(self, direction, HEIGHT):
        if direction == "up":
            self.rect.y -= self.speed
        elif direction == "down":
            self.rect.y += self.speed

        # Empêcher de sortir de l'écran
        if self.rect.top < 0:
            self.rect.top = 0
        if self.rect.bottom > HEIGHT:
            self.rect.bottom = HEIGHT

    def move_towards(self, target_y, HEIGHT):
        if self.rect.centery < target_y:
            self.move("down", HEIGHT)
        elif self.rect.centery > target_y:
            self.move("up", HEIGHT)

    def move_manual(self, up_key, down_key, HEIGHT):
        keys = pygame.key.get_pressed()
        if keys[up_key]:
            self.move("up", HEIGHT)
        if keys[down_key]:
            self.move("down", HEIGHT)


    def draw(self, surface, color):
        pygame.draw.rect(surface, color, self.rect)