import numpy as np
import time
import gym
from gym import spaces, logger
from IPython.display import clear_output


class FishPondEnv(gym.Env):
    def __init__(self, params):
        super(FishPondEnv, self).__init__()
        self.params = params
        self.num_agents = params['num_agents']
        self.grid_size = params['grid_size']
        self.pond_size = np.int(self.grid_size * self.params['pond_size_ratio'])
        self.grid_area = self.grid_size * self.grid_size
        self.pond_area = self.grid_size * self.pond_size
        self.action_space = spaces.Discrete(5)
        state_space_low = np.array(self.grid_area * [0] + [0] + self.num_agents * [0])
        state_space_high = np.array(self.grid_area * [2] + [params['fish_count_max']] + self.num_agents * [params['health_max']])
        self.state_space = spaces.Box(state_space_low, state_space_high, dtype=np.float32) #Check dtype here
        self.state = None
        self.steps = None
        self.loc_agents = None
    
    def get_initial_state(self): #Missing Agent Issue Fixed
        grid = np.array(self.pond_area * [1] + (self.grid_area - self.pond_area) * [0])
        zone_correction = np.int((self.params['fishing_zone_size'] % 2) != (self.grid_size % 2))
        zone_size = self.params['fishing_zone_size'] + zone_correction
        zone_index = np.int((self.pond_size - 1) * self.grid_size + (self.grid_size - zone_size) / 2)
        grid[range(zone_index, zone_index + zone_size)] = 2
        agent_location = np.random.choice(range(self.pond_area, self.grid_area), self.num_agents)
        self.loc_agents = agent_location
        grid[agent_location] = 3
        self.fish_count = float(self.params['fish_count_initial'])
        agent_health = self.num_agents * [self.params['health_max']]
        state = np.append(grid, [[self.fish_count] + agent_health])
        return state
        
    def reset(self):
        self.state = self.get_initial_state()
        self.steps = 0
        return self.state
    
    def to_grid_index(self, index, inverse=False):
        if not inverse:
            row = np.int(index / self.grid_size)
            col = index % self.grid_size
            return [row, col]
        return (index[0] * self.grid_size + index[1])
    
    def step_agent(self, agent_index, action): # Agent Collision Issue
        agent_health = self.state[(self.grid_area + 1 + agent_index)]
        dead = True if (agent_health == 0) else False
        if(dead):
            return 0
        reward = self.params['reward_per_step']
        grid = self.state[:self.grid_area].reshape(self.grid_size, self.grid_size)
        fish_count = self.fish_count
        loc_vt, loc_hz = self.to_grid_index(self.loc_agents[agent_index]) #Update self.loc_agents
        grid[loc_vt, loc_hz] = 0
        
        if (action == 0):
            if (grid[loc_vt-1][loc_hz] == 2 and fish_count >= 1):
                fish_count = np.clip(fish_count - 1, 0, self.params['fish_count_max'])
                agent_health = np.clip(agent_health + self.params['nutrition_per_fish'], 0, self.params['health_max'])
                
                
        elif (action == 1):
            loc_vt = np.clip(loc_vt - 1, self.pond_size, self.grid_size - 1)
        elif (action == 2):
            loc_vt = np.clip(loc_vt + 1, self.pond_size, self.grid_size - 1)
        elif (action == 3):
            loc_hz = np.clip(loc_hz - 1, 0, self.grid_size - 1)
        elif (action == 4):
            loc_hz = np.clip(loc_hz + 1, 0, self.grid_size - 1)
        else:
            logger.warn("Undefined Action")
        
        grid[loc_vt, loc_hz] = 3
        self.loc_agents[agent_index] = self.to_grid_index([loc_vt, loc_hz], inverse=True)
        agent_health = np.clip(agent_health - self.params['hunger_per_step'], 0, self.params['health_max'])
        fish_count = np.clip(fish_count + self.params['fish_regeneration_rate'] * fish_count, 0, self.params['fish_count_max'])
        self.fish_count = fish_count
        self.state[:self.grid_area] = grid.flatten()
        self.state[self.grid_area] = int(fish_count)
        self.state[(self.grid_area + 1 + agent_index)] = agent_health
        return reward
    
    def check_termination(self, rewards): # Negative Rewards for Losing
        fish_count = self.state[self.grid_area]
        agents_health = self.state[self.grid_area + 1 : self.grid_area + 1 + self.num_agents]
        condition1 = False if (np.sum(agents_health)) else True
        condition2 = False if (fish_count) else True
        condition3 = False if (self.steps <= self.params['episode_length']) else True
        done = condition1 or condition2 or condition3
        return [rewards, done]
    
    def step(self, actions): #Discuss about synchronization in real time before each agent takes action
        self.steps += 1
        rewards = np.array([])
        if (self.num_agents == 1):
            actions = [actions]
        for agent_index in range(self.num_agents):
            reward = self.step_agent(agent_index, actions[agent_index])
            rewards = np.append(rewards, reward)
        rewards, done = self.check_termination(rewards) #Check for steps beyond done
        if (self.num_agents == 1):
            rewards = rewards[0]
        info = {}
        return [self.state, rewards, done, info]

    def render(self, mode, delay=1):
        clear_output(wait=True)
        grid = self.state[:self.grid_area].reshape(self.grid_size, self.grid_size).astype(int)
        fish_count = self.state[self.grid_area]
        agent_health = self.state[(self.grid_area + 1):]
        print('Fish Count:', fish_count)
        print('Fish Count (Running):', self.fish_count)
        print('Agent Health:', agent_health)
        print('\n')
        for row in grid:
            print(' '.join(map(str, row)))
        print('\n')
        time.sleep(delay)
        return