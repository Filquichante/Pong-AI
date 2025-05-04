import random

class BotOP:
  
    def __init__(self, x, y, paddle_width, paddle_height, speed, game):
        self.x = x
        self.y = y
        self.width = paddle_width
        self.height = paddle_height
        self.speed = speed
        self.game = game
        self.ball_radius = game.BALL_RADIUS

    def predict_ball_y(self):

        bx = self.game.ball.rect.centerx
        by = self.game.ball.rect.centery

        vx = self.game.ball.speed_x
        vy = self.game.ball.speed_y

        target_x = self.x + self.width

        t = (target_x - bx)/vx
        if t < 0:
            return self.game.HEIGHT // 2
        
        period = 2 * (self.game.HEIGHT - self.ball_radius*2)
        simul_y = (by - self.ball_radius + vy*t)%period
        if simul_y > (self.game.HEIGHT - self.ball_radius*2):
            simul_y = period - simul_y


        return simul_y + self.ball_radius           # Le y prédit, il faut maintenant s'y rendre.

    def update(self):
        
        

        target_y = self.predict_ball_y()
        paddle_rect = self.game.left_paddle.rect

        # Calcul de l'erreur NB : -50,50 est un niveau humain, ça devient très compliqué vers -30,30. Si on descend encore plus il n'est plus battable par un humain.
        error = random.gauss(-self.game.botOP_error, self.game.botOP_error)  
        target_y = target_y + error

        # Déplacement vers target
        if paddle_rect.centery < target_y:
            paddle_rect.y += self.speed
        elif paddle_rect.centery > target_y:
            paddle_rect.y -= self.speed
            
        # Attention, même si c'est rare après quelques simulations il arrive que le paddle sorte légérement de la fenètre.
        # On le limite donc à la fenètre en dur pour éviter ce genre de choses.
        paddle_rect.top = max(0, paddle_rect.top)
        paddle_rect.bottom = min(self.game.HEIGHT, paddle_rect.bottom)
