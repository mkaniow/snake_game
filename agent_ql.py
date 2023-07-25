import numpy as np
import random
from game import Point
from game import Direction, SnakeGame
import json
import os.path

LS = 0.2
GAMMA = 0.2

class Agent:

    def __init__(self):
        self.epsilon = 0
        self.gamma = GAMMA
        self.ls = LS
        self.fname = f'file_alfa{self.ls}_gamma{self.gamma}.json'

        if not os.path.isfile(self.fname):
            with open(self.fname, 'a') as f:
                table = np.zeros((4, 9, 16, 3))
                tableBetter = table.tolist()
                sample = {
                'game_number' : 0,
                'memory' : tableBetter,
                'score' : [0],
                'average' : [0],
                'record' : 0
                }
                json_string = json.dumps(sample)
                f.write(json_string)
            self.memory = json.load(open(self.fname))
        elif os.path.isfile(self.fname):
            self.memory = json.load(open(self.fname))

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state1 = [
            dir_l,
            dir_r,
            dir_u,
            dir_d
            ]

        state2 = [
            # Danger straight
            (dir_r and game.colision(point_r)) or 
            (dir_l and game.colision(point_l)) or 
            (dir_u and game.colision(point_u)) or 
            (dir_d and game.colision(point_d)),

            # Danger right
            (dir_u and game.colision(point_r)) or 
            (dir_d and game.colision(point_l)) or 
            (dir_l and game.colision(point_u)) or 
            (dir_r and game.colision(point_d)),

            # Danger left
            (dir_d and game.colision(point_r)) or 
            (dir_u and game.colision(point_l)) or 
            (dir_r and game.colision(point_u)) or 
            (dir_l and game.colision(point_d))]

        state3 = [
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]
        
        return np.array(state1, dtype=int), np.array(state2, dtype=int), np.array(state3, dtype=int)

    def get_state_next(self, game, move):
        head = game.snake[0]
        
        clock_directions = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        index = clock_directions.index(game.direction)

        if np.array_equal(move, [1, 0, 0]): #no change of direction
            new_direction = clock_directions[index]
        elif np.array_equal(move, [0, 1, 0]): #turn right
            next_index = (index + 1) % 4
            new_direction = clock_directions[next_index]
        else:
            next_index = (index - 1) % 4
            new_direction = clock_directions[next_index]

        old_direction = game.direction
        if old_direction == Direction.LEFT and np.array_equal(move, [1, 0, 0]):
            new_head_x = head.x - 20
            new_head_y = head.y
        if old_direction == Direction.LEFT and np.array_equal(move, [0, 1, 0]):
            new_head_x = head.x
            new_head_y = head.y - 20
        if old_direction == Direction.LEFT and np.array_equal(move, [0, 0, 1]):
            new_head_x = head.x
            new_head_y = head.y + 20
        if old_direction == Direction.RIGHT and np.array_equal(move, [1, 0, 0]):
            new_head_x = head.x + 20
            new_head_y = head.y
        if old_direction == Direction.RIGHT and np.array_equal(move, [0, 1, 0]):
            new_head_x = head.x
            new_head_y = head.y + 20
        if old_direction == Direction.RIGHT and np.array_equal(move, [0, 0, 1]):
            new_head_x = head.x
            new_head_y = head.y - 20
        if old_direction == Direction.UP and np.array_equal(move, [1, 0, 0]):
            new_head_x = head.x
            new_head_y = head.y - 20
        if old_direction == Direction.UP and np.array_equal(move, [0, 1, 0]):
            new_head_x = head.x + 20
            new_head_y = head.y
        if old_direction == Direction.UP and np.array_equal(move, [0, 0, 1]):
            new_head_x = head.x - 20
            new_head_y = head.y
        if old_direction == Direction.DOWN and np.array_equal(move, [1, 0, 0]):
            new_head_x = head.x
            new_head_y = head.y + 20
        if old_direction == Direction.DOWN and np.array_equal(move, [0, 1, 0]):
            new_head_x = head.x - 20
            new_head_y = head.y
        if old_direction == Direction.DOWN and np.array_equal(move, [0, 0, 1]):
            new_head_x = head.x + 20
            new_head_y = head.y

        point_l = Point(new_head_x - 20, new_head_y)
        point_r = Point(new_head_x + 20, new_head_y)
        point_u = Point(new_head_x, new_head_y - 20)
        point_d = Point(new_head_x, new_head_y + 20)


        dir_l = new_direction == Direction.LEFT
        dir_r = new_direction == Direction.RIGHT
        dir_u = new_direction == Direction.UP
        dir_d = new_direction == Direction.DOWN

        state1 = [
            dir_l,
            dir_r,
            dir_u,
            dir_d
            ]

        state2 = [
            # Danger straight
            (dir_r and game.colision(point_r)) or 
            (dir_l and game.colision(point_l)) or 
            (dir_u and game.colision(point_u)) or 
            (dir_d and game.colision(point_d)),

            # Danger right
            (dir_u and game.colision(point_r)) or 
            (dir_d and game.colision(point_l)) or 
            (dir_l and game.colision(point_u)) or 
            (dir_r and game.colision(point_d)),

            # Danger left
            (dir_d and game.colision(point_r)) or 
            (dir_u and game.colision(point_l)) or 
            (dir_r and game.colision(point_u)) or 
            (dir_l and game.colision(point_d))]

        state3 = [
            game.food.x < new_head_x,  # food left
            game.food.x > new_head_x,  # food right
            game.food.y < new_head_y,  # food up
            game.food.y > new_head_y  # food down
            ]
        
        return np.array(state1, dtype=int), np.array(state2, dtype=int), np.array(state3, dtype=int)

    def get_action(self, c1, c2, c3): #c stands for coordinate in memory
        self.epsilon = 150 - self.memory['game_number']
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon: #random move
            move = random.randint(0, 2)
            final_move[move] = 1
        else: #move based on memory
            move = np.argmax(self.memory['memory'][c1][c2][c3])
            final_move[move] = 1
        return final_move

def find_index(vector):
    i = 0
    for cell in vector:
        if cell == 1:
            return i
        else:
            i += 1

def binaryToDecimal(val): 
    return int(val, 2) 

def train():
    agent = Agent()
    game = SnakeGame()
    while True:
        #take state informations
        state_old = agent.get_state(game)
        vect_directions = state_old[0]
        vect_dangers = state_old[1]
        vect_point = state_old[2]

        #calculate cords for memory
        binary2 = ''.join([str(elem) for elem in vect_dangers])
        binary3 = ''.join([str(elem) for elem in vect_point])
        coordinate1 = find_index(vect_directions)
        coordinate2 = binaryToDecimal(binary2)
        coordinate3 = binaryToDecimal(binary3)

        #find best action and make decision 
        index_q = np.argmax(agent.memory['memory'][coordinate1][coordinate2][coordinate3])
        q = agent.memory['memory'][coordinate1][coordinate2][coordinate3][index_q]
        finale_move = agent.get_action(coordinate1, coordinate2, coordinate3)

        #make move
        game_over, score, reward = game.play_step(finale_move)

        #take next state informations
        state_new = agent.get_state_next(game, finale_move)
        vect_directions_new = state_new[0]
        vect_dangers_new = state_new[1]
        vect_point_new = state_new[2]

        #calculate cords for memory for next state
        binary2_new = ''.join([str(elem) for elem in vect_dangers_new])
        binary3_new = ''.join([str(elem) for elem in vect_point_new])
        coordinate1_new = find_index(vect_directions_new)
        coordinate2_new = binaryToDecimal(binary2_new)
        coordinate3_new = binaryToDecimal(binary3_new)

        #find best action for next state
        index_q_prim = np.argmax(agent.memory['memory'][coordinate1_new][coordinate2_new][coordinate3_new])
        q_prim = agent.memory['memory'][coordinate1_new][coordinate2_new][coordinate3_new][index_q_prim]

        #new_Q = q + agent.ls * (reward + agent.gamma * q_prim - q)
        #calculate newQ value
        agent.memory['memory'][coordinate1][coordinate2][coordinate3][index_q] = q + agent.ls * (reward + agent.gamma * q_prim - q)
        with open(f'file_alfa{agent.ls}_gamma{agent.gamma}.json', 'w') as f:
            json.dump(agent.memory, f)

        if game_over == True:
            game.reset()
            agent.memory['game_number'] = agent.memory['game_number'] + 1
            agent.memory['score'].append(score)

            #loop to stop agent form playing
            #if agent.memory['game_number'] > 300:
            #    break
    
if __name__ == '__main__':
    train()