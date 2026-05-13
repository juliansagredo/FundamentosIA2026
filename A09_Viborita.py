#==========================================
#  Agente Viborita Inteligente
#==========================================
#  Traductor: Julián Sagredo
#  Fundamentos de Inteligencia Artificial
#  ESFM IPN 2023
#===========================================
import torch
import random
import numpy as np
from collections import deque
from viborita.juego import ViboritaInteligente, Direccion, Punto
from viborita.modelo import Linear_QNet, QTrainer
from viborita.grafica import graficar

MAX_MEMORIA = 100000
TAMAÑO_DE_LA_COLA = 1000
LR = 0.001
JUEGOS_AL_AZAR = 80

#==================
#  Clase Agente 
#==================
class Agente:

    #===========================
    #  Constructor: 
    #    model - red neuronal
    #    trainer - optimizador
    #===========================
    def __init__(self):

        self.n_games = 0
        self.epsilon = JUEGOS_AL_AZAR # juegos al azar 
        self.gamma = 0.9 # tasa de descuento
        self.memory = deque(maxlen=MAX_MEMORIA) # pila finita popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    #======================
    #  Estado del agente 
    #======================
    def obtener_estado(self, juego):

        head = juego.snake[0]

        #================
        #  Pixel 20x20
        #================
        point_l = Punto(head.x - 20, head.y)
        point_r = Punto(head.x + 20, head.y)
        point_u = Punto(head.x, head.y - 20)
        point_d = Punto(head.x, head.y + 20)
        
        dir_l = juego.direction == Direccion.LEFT
        dir_r = juego.direction == Direccion.RIGHT
        dir_u = juego.direction == Direccion.UP
        dir_d = juego.direction == Direccion.DOWN

        state = [
            #===================
            # Peligro enfrente
            #===================
            (dir_r and juego.is_collision(point_r)) or 
            (dir_l and juego.is_collision(point_l)) or 
            (dir_u and juego.is_collision(point_u)) or 
            (dir_d and juego.is_collision(point_d)),

            #======================
            # Peligro a la derecha
            #======================
            (dir_u and juego.is_collision(point_r)) or 
            (dir_d and juego.is_collision(point_l)) or 
            (dir_l and juego.is_collision(point_u)) or 
            (dir_r and juego.is_collision(point_d)),

            #========================
            # Peligro a la izquierda
            #========================
            (dir_d and juego.is_collision(point_r)) or 
            (dir_u and juego.is_collision(point_l)) or 
            (dir_r and juego.is_collision(point_u)) or 
            (dir_l and juego.is_collision(point_d)),
            
            #=========================
            # Dirección de movimiento
            #=========================
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            #=======================
            # Posición de la comida
            #=======================
            juego.food.x < juego.head.x,  # comida a la izquierda
            juego.food.x > juego.head.x,  # comida a la derecha
            juego.food.y < juego.head.y,  # comida arriba
            juego.food.y > juego.head.y   # comida abajo
            ]

        #===============================================
        #  Regresa estado convertido a enteros (0 o 1)
        #===============================================
        return np.array(state, dtype=int)

    #=====================
    #  Añadir en memoria 
    #=====================
    def recordar(self, state, action, reward, next_state, done):

        self.memory.append((state, action, reward, next_state, done)) 
        # popleft si se alcanza la MAX_MEMORIA 

    #===================================
    #  Entrenar memoria de largo plazo
    #===================================
    def entrenar_memoria_larga(self):

        if len(self.memory) > TAMAÑO_DE_LA_COLA:
            mini_sample = random.sample(self.memory, TAMAÑO_DE_LA_COLA) # lista de tuplas
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(np.array(states,dtype=np.float32), actions, rewards, np.array(next_states,dtype=np.float32), dones)

    #===================================
    #  Entrenar memoria de corto plazo 
    #===================================
    def entrenar_memoria_corta(self, state, action, reward, next_state, done):

        self.trainer.train_step(state, action, reward, next_state, done)

    #==================
    #  Decidir acción
    #==================
    def obtener_accion(self, state):

        #=======================================================================
        # movimientos al azar:  balance entre exploración / explotación 
        # JUEGOS_AL_AZAR juegos con posibilidad de hacer un movimiento al azar
        #=======================================================================
        self.epsilon = JUEGOS_AL_AZAR - self.n_games
        final_move = [0,0,0]

        if random.randint(0, 200) < self.epsilon:
            #=====================================
            #  Genera entero al azar entre 0 y 2
            #=====================================
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            #==========================================
            # Dado state (R11) genera prediction (R3)
            #==========================================
            prediction = self.model(state0)
            #=====================================================
            #  move es entero entre 0 y 2
            #  es la entrada con valor máximo en prediction (R3)
            #=====================================================
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        #==========================================
        # Decisión es un vector en R3 de con 0 o 1 
        #==========================================
        return final_move

#====================================
#  FUNCIÓN PRINCIPAL: ENTRENAMIENTO
#====================================
def entrenar():

    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agente = Agente()
    juego = ViboritaInteligente()

    #====================
    #  Ciclo infinito 
    #====================
    while True:

        #=========================
        # Obtener estado anterior
        #=========================
        state_old = agente.obtener_estado(juego)

        #====================
        # Obtener movimiento
        #====================
        final_move = agente.obtener_accion(state_old)

        #===============================
        # Mover y obtener nuevo estado  
        #===============================
        reward, done, score = juego.play_step(final_move)
        state_new = agente.obtener_estado(juego)

        #=========================
        # Entrenar memoria corta
        #=========================
        agente.entrenar_memoria_corta(state_old, final_move, reward, state_new, done)

        #==========
        # Recordar
        #==========
        agente.recordar(state_old, final_move, reward, state_new, done)

        if done:
            #=============================================
            # Entrenar memoria larga, graficar resultado
            #=============================================
            juego.reset()
            agente.n_games += 1
            agente.entrenar_memoria_larga()

            if score > record:
                record = score
                agente.model.save()

            print('Juego', agente.n_games, 'Puntos', score, 'Récord:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agente.n_games
            plot_mean_scores.append(mean_score)
            graficar(plot_scores, plot_mean_scores)

#=======================
#  PROGRAMA PRINCIPAL
#=======================
if __name__ == '__main__':
    entrenar()
