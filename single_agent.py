from environment.v1 import FishPondEnv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
import keras.backend as K


NUM_STEPS = 1

params = {
    'num_agents': 1, #dont change
    'health_max': 100,
    'grid_size': 10,
    'pond_size_ratio': 0.25,
    'fishing_zone_size': 4, #Auto-Symmetry
    'fish_count_initial': 10,
    'fish_count_max': 50,
    'fish_regeneration_rate': 0.05,
    'hunger_per_step': 5,
    'nutrition_per_fish': 10,
    'reward_per_step': 0,
    'episode_length': 3
}


env = FishPondEnv(params)
state = env.reset()

action_dict = np.array(['Eat Fish', 'Up', 'Down', 'Left', 'Right'])




for t in range(NUM_STEPS):
    print('\n T =', t, '\n')
    print(state[:env.grid_area].reshape(env.grid_size, env.grid_size))
    actions = int(input())#np.random.randint(0, 5, env.num_agents)
    
    state,rewards,done = env.step([actions])
    health = state[-1]
    num_fishes = state[-2]
    
    print('\nActions: ', action_dict[actions],', Health: ',health,', Num Fishes: ',num_fishes, ', Running Fish Count',env.running_fish_count)