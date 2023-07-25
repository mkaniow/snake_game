from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import numpy as np
from game import Point
from game import Direction, SnakeGame
import json
import os.path
import random

#some variables to create model
MEMORY_SIZE = 100000
INPUT_SHAPE = 11
LR = 0.001
GAMMA = 0.8
BATCH_SIZE = 128
LAYER_SIZE = 256
FILE_NAME = f'dqn_model_score_memory_{LAYER_SIZE}.json'

#length of training stage
GAME_EPSILON = 150

class ReplayBuffer(object):
    """
    ReplayBuffer handles storage of the state, action, reward, state transition
    """
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size)

    def store_transition(self, state, action, reward, state_):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]

        return states, actions, rewards, states_

#below are 6 functions to build a model

def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):
    model = Sequential([
                Dense(fc1_dims, input_shape=(input_dims,)),
                Activation('relu'),
                Dense(fc2_dims),
                Activation('relu'),
                Dense(n_actions)])

    model.compile(optimizer=Adam(lr=lr), loss='mse')

    return model

#def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):
#    model = Sequential([
#                Dense(fc1_dims, input_shape=(input_dims,)),
#                Activation('relu'),
#                Dense(fc2_dims),
#                Activation('relu'),
#                Dense(fc2_dims),
#                Activation('relu'),
#                Dense(n_actions)])
#
#    model.compile(optimizer=Adam(lr=lr), loss='mse')
#
#    return model

#def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):
#    model = Sequential([
#                Dense(fc1_dims, input_shape=(input_dims,)),
#                BatchNormalization(),
#                Activation('relu'),
#                Dense(fc2_dims),
#                BatchNormalization(),
#                Activation('relu'),
#                Dense(n_actions)])
#
#    model.compile(optimizer=Adam(lr=lr), loss='mse')
#
#    return model

#def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):
#    model = Sequential([
#                Dense(fc1_dims, input_shape=(input_dims,)),
#                Dropout(0.2),
#                Activation('relu'),
#                Dense(fc2_dims),
#                Dropout(0.2),
#                Activation('relu'),
#                Dense(n_actions)])
#
#    model.compile(optimizer=Adam(lr=lr), loss='mse')
#
#    return model

#def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):
#    model = Sequential([
#                Dense(fc1_dims, input_shape=(input_dims,)),
#                BatchNormalization(),
#                Activation('relu'),
#                Dense(fc2_dims),
#                BatchNormalization(),
#                Activation('relu'),
#                Dense(fc2_dims),
#                BatchNormalization(),
#                Activation('relu'),
#                Dense(n_actions)])
#
#    model.compile(optimizer=Adam(lr=lr), loss='mse')
#
#    return model

#def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):
#    model = Sequential([
#                Dense(fc1_dims, input_shape=(input_dims,)),
#                Dropout(0.2),
#                Activation('relu'),
#                Dense(fc2_dims),
#                Dropout(0.2),
#                Activation('relu'),
#                Dense(fc2_dims),
#                Dropout(0.2),
#                Activation('relu'),
#                Dense(n_actions)])
#
#    model.compile(optimizer=Adam(lr=lr), loss='mse')
#
#    return model

class Agent:
    """
    This class creating a deep Q-learning agent
    """
    def __init__(self, alpha, gamma, n_actions, batch_size,
                 input_dims,mem_size=MEMORY_SIZE, fname=f'dqn_model.h5'):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = 0
        self.batch_size = batch_size
        self.model_file = fname
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)
        self.q_eval = build_dqn(alpha, n_actions, input_dims, LAYER_SIZE, LAYER_SIZE)

        if not os.path.isfile(FILE_NAME):
            with open(FILE_NAME, 'a') as f:
                sample = {
                'game_number' : 0,
                'score' : []
                }
                json_string = json.dumps(sample)
                f.write(json_string)
            self.score_memory = json.load(open(FILE_NAME))
        elif os.path.isfile(FILE_NAME):
            self.score_memory = json.load(open(FILE_NAME))


    def remember(self, state, action, reward, new_state):
        self.memory.store_transition(state, action, reward, new_state)
               
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

        arr = np.concatenate((state1, state2, state3))
        
        return arr

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
        
        arr_next = np.concatenate((state1, state2, state3))
        
        return arr_next

    def get_action(self, state):
        state = state[np.newaxis, :]
        epsilon = GAME_EPSILON - self.score_memory['game_number']
        final_move = [0, 0, 0]
        if random.randint(0, GAME_EPSILON) < epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)
            final_move[action] = 1
        return final_move

    def learn(self):
        if self.memory.mem_cntr > self.batch_size:
            state, action, reward, new_state = self.memory.sample_buffer(self.batch_size)
            action_values = np.array(self.action_space, dtype=np.int8)
            action_indices_ = np.dot(action, action_values)
            action_indices = action_indices_.astype(int)

            q_eval = self.q_eval.predict(state)

            q_next = self.q_eval.predict(new_state)

            q_target = q_eval.copy()

            batch_index = np.arange(self.batch_size, dtype=np.int32)

            q_target[batch_index, action_indices] = reward + self.gamma*np.max(q_next, axis=1)

            _ = self.q_eval.fit(state, q_target)

    #def save_model(self):
    #    save_path = 'C:\\Users\\Michał\\Desktop\\repo\\snake\\dql_models'
    #    completeName = os.path.join(save_path, self.model_file)
    #    self.q_eval.save(completeName)

def train():
    agent = Agent(LR, GAMMA, 3, BATCH_SIZE, INPUT_SHAPE)
    game = SnakeGame()
    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        state_new = agent.get_state_next(game, final_move)
        game_over, score, reward = game.play_step(final_move)
        agent.remember(state_old, final_move, reward, state_new)

        if game_over == True:
            agent.learn()
            with open(FILE_NAME, 'w') as f:
                json.dump(agent.score_memory, f)
            #agent.save_model()
            agent.score_memory['game_number'] += 1
            agent.score_memory['score'].append(score)
            game.reset()
            #if agent.score_memory['game_number'] > 300:
            #    break

if __name__ == '__main__':

    train()

#loop to train diffrent models
#    for LAYER_SIZE in np.arange(65, 155, 15):
#        train()
#    