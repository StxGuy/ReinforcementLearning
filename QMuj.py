import mujoco as mj
import numpy as np
import matplotlib.pyplot as pl

class DiscreteMuJoCoEnv:
    def __init__(self, filename, target_pos=1.5):
        self.model = mj.MjModel.from_xml_path(filename)
        self.data = mj.MjData(self.model)
        self.target_pos = target_pos
        
        # Discretize position space: -2 to 2 in 20 bins
        self.pos_bins = np.linspace(-2,2,21) # 20 intervals, 21 edges
        self.n_pos_states = len(self.pos_bins) - 1
        
        # Discretize velocity space: -2 to 2 in 10 bins
        self.vel_bins = np.linspace(-2,2,11) # 10 intervals, 11 edges
        self.n_vel_states = len(self.vel_bins) - 1
        
        # Total discrete states
        self.n_states = self.n_pos_states*self.n_vel_states
        
        # Discrete actions: 0=left, 1=stay, 2=right
        self.actions = [-1.0, 0.0, 1.0]
        self.n_actions = len(self.actions)
        
        print(f"State space: {self.n_states} states")
        print(f"Action space: {self.n_actions} actions")
        
    def get_state(self):
        pos = self.data.qpos[0]
        vel = self.data.qvel[0]
        
        # Discretize position and velocity
        pos_idx = np.digitize(pos, self.pos_bins) - 1
        vel_idx = np.digitize(vel, self.vel_bins) - 1
        
        # Clamp to valid ranges
        pos_idx = np.clip(pos_idx, 0, self.n_pos_states - 1)
        vel_idx = np.clip(vel_idx, 0, self.n_vel_states - 1)
        
        # Convert to single state index
        state = pos_idx*self.n_vel_states + vel_idx
        return state
    
    def step(self, action):
        # Apply action
        self.data.ctrl[0] = self.actions[action]
        
        # Step simulation
        mj.mj_step(self.model, self.data)
        
        # Get new state
        new_state = self.get_state()
        pos = self.data.qpos[0]
        
        # Calculate reward
        distance_to_target = abs(pos - self.target_pos)
        
        if distance_to_target < 0.1:    # Close to target
            reward = 10
            done = True
        elif abs(pos) > 1.9:    # Hit boundaries
            reward = -5
            done = True
        else:
            reward = -distance_to_target
            done = False
            
        return new_state, reward, done
    
    def reset(self):
        self.data.qpos[0] = np.random.uniform(-1,1)
        self.data.qvel[0] = 0
        mj.mj_forward(self.model, self.data)
        return self.get_state()
    
class QLearning:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_states = n_states
        self.n_actions = n_actions
        
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q[state])
        
    def update(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            target = reward + self.gamma*np.max(self.Q[next_state])
            
        self.Q[state,action] += self.alpha*(target - self.Q[state,action])
        
def train():
    env = DiscreteMuJoCoEnv("slider.xml")
    agent = QLearning(env.n_states, env.n_actions)
    
    rewards = []
    
    for episode in range(5000):
        state = env.reset()
        total_reward = 0
        steps = 0
        max_steps = 200
        
        while steps < max_steps:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            
            agent.update(state,action,reward,next_state,done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
            
        rewards.append(total_reward)
        
        if episode % 100 == 0:
            avg_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")
            
    return agent, rewards

if __name__ == "__main__":
    agent, rewards = train()
    
    window_size = 100
    moving_avg = []
    for i in range(len(rewards)):
        start_idx = max(0,i-window_size+1)
        moving_avg.append(np.mean(rewards[start_idx:i+1]))
    
    pl.figure(figsize=(10,6))
    pl.plot(rewards, alpha=0.4)
    pl.plot(moving_avg)
    pl.title('Q-Learning on MuJoCo Slider')
    pl.xlabel('Episode')
    pl.ylabel('Total Reward')
    pl.grid(True, alpha=0.3)
    pl.axis([0, 5000, -600, 35])
    pl.show()
        
