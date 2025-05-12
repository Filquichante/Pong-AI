import pygame, sys, random, math

class Ball:
    def __init__(self, WIDTH, HEIGHT, BALL_RADIUS, BALL_SPEED, game):
        self.BALL_SPEED = BALL_SPEED
        self.game = game
        self.rect = pygame.Rect(WIDTH // 2 - BALL_RADIUS, HEIGHT // 2 - BALL_RADIUS, BALL_RADIUS * 2, BALL_RADIUS * 2)
        self.angle = 0
        self.update_angle(self.angle_random())

    def move(self, WIDTH, HEIGHT):
        self.rect.x += self.speed_x
        self.rect.y += self.speed_y

        if self.rect.top <= 0:
            self.speed_y *= -1
            self.rect.top = 0
        if self.rect.bottom >= HEIGHT:
            self.speed_y *= -1
            self.rect.bottom = HEIGHT

    def reset(self, WIDTH, HEIGHT):
        self.rect.center = (WIDTH // 2, HEIGHT // 2)
        self.update_angle(self.angle_random())  # Angle de départ aléatoire entre -15° et 15°

    def draw(self, surface, color):
        pygame.draw.ellipse(surface, color, self.rect)

    def update_angle(self, angle):
        self.angle = angle
        self.speed_x = math.cos(self.angle) * self.BALL_SPEED
        self.speed_y = math.sin(self.angle) * self.BALL_SPEED

    def check_collision(self, paddle1, paddle2, BALL_SPEED):
        if self.rect.colliderect(paddle1.rect):
            # Inverser la direction horizontale et ajuster l'angle
            
            self.angle = math.pi - self.angle


            # Calculer l'offset pour ajuster l'angle vertical
            offset = (self.rect.centery - paddle1.rect.centery) / (paddle1.rect.height / 2)
            self.angle = offset * (math.pi / 4)  # Ajustement de l'angle (limité à ±45°)

            # Mettre à jour les composantes de vitesse
            self.update_angle(self.angle)

        if self.rect.colliderect(paddle2.rect):

            #Récompenser l'IA si elle touche la balle
            if self.game.trainingAI:
                self.game.last_touching_frame = self.game.frame_number
                

            # Inverser la direction horizontale et ajuster l'angle
            self.angle = math.pi - self.angle

            # Calculer l'offset pour ajuster l'angle vertical
            offset = (self.rect.centery - paddle2.rect.centery) / (paddle2.rect.height / 2)
            self.angle = math.pi - offset * (math.pi / 4)  # Ajustement de l'angle (limité à ±45°)

            # Mettre à jour les composantes de vitesse
            self.update_angle(self.angle)


    def angle_random(self):
        random_alpha = random.uniform(-self.game.alpha_angle, self.game.alpha_angle)/2
        #random_alpha += random.choice([0, 1]) * math.pi
        return random_alpha