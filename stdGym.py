import gymnasium as gym
import numpy as np
   

class game:
    def __init__(
        self,
        env: gym.Env,
        alpha: float = 0.1,
        gamma: float = 0.95):
        
        self.env = env
        self.Q = np.random.normal(size=(env.observation_space.n,
                                        env.action_space.n))*0.01
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.max_steps = 100
        self.max_episodes = 1000
        
    def eGreedy(self,s,epsilon):
        nA = self.Q.shape[1]
        if np.random.random() < epsilon:
            return np.random.randint(0,nA)
        else:
            return np.argmax(self.Q[s,:])
    
    def updateQ(self,s,a,r,sn,t):
        if t:
            self.Q[s,a] += self.alpha*(r - self.Q[s,a])
        else:
            self.Q[s,a] += self.alpha*(r + self.gamma*max(self.Q[sn,:]) - self.Q[s,a])
            
    def run_episode(self):
        s, info = self.env.reset()
        
        episode_reward = 0.0
        step_count = 0
        terminated = False
        
        while not terminated:
            a = self.eGreedy(s,self.epsilon)
            x = env.step(a)
            sn = x[0]
            r  = x[1]
            terminated = x[2] or step_count > self.max_steps
       
            self.updateQ(s,a,r,sn,terminated)
            
            s = sn
            episode_reward += r
            step_count += 1
       
        return episode_reward, step_count
        
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)
        
    def train(self):
        episode_rewards = []
        success_count = 0
        
        for episode in range(self.max_episodes):
            episode_reward, steps = self.run_episode()
            episode_rewards.append(episode_reward)
            
            if episode_reward > 0:
                success_count += 1
                
            self.decay_epsilon()
            
            if (episode + 1)%1000 == 0:
                recent_success_rate = np.mean([r > 0 for r in episode_rewards[-100:]])
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode+1:5d}: "
                      f"Success Rate = {recent_success_rate:.3f}, "
                      f"Avg Reward   = {avg_reward:.3f}, "
                      f"Epsilon      = {self.epsilon:.3f}")
                
        final_success_rate = success_count / self.max_episodes
        print(f"\nTraining complete!")
        print(f"Total successful episodes: {success_count}/{self.max_episodes}")
        print(f"Final success rate.......: {final_success_rate:.3f}")
        
        return episode_rewards
    
    def test_policy(self, num_episodes=100):
        print(f"\nTesting learned policy over {num_episodes} episodes...")
        
        test_rewards = []
        for _ in range(num_episodes):
            s, info = self.env.reset()
            episode_reward = 0.0
            step_count = 0
            terminated = False
            
            while not terminated and step_count < self.max_steps:
                a = np.argmax(self.Q[s,:])
                x = self.env.step(a)
                s = x[0]
                episode_reward += x[1]
                step_count += 1
                
            test_rewards.append(episode_reward)
            
        success_rate = np.mean([r > 0 for r in test_rewards])
        avg_reward = np.mean(test_rewards)
        
        print(f"Test Results:")
        print(f"  Success rate..: {success_rate:.3f}")
        print(f"  Average reward: {avg_reward:.3f}")
        
        return test_rewards
            
if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=False)

    print(f"Environment......: {env.spec.id}")
    print(f"State space size.: {env.observation_space.n}")
    print(f"Action space size: {env.action_space.n}")

    g = game(env)
    episode_rewards = g.train()
    test_rewards = g.test_policy()
    
    env.close()
    
    try:
        import matplotlib.pyplot as pl
        
        window_size = 100
        moving_avg = []
        for i in range(len(episode_rewards)):
            start_idx = max(0,i-window_size+1)
            moving_avg.append(np.mean(episode_rewards[start_idx:i+1]))
            
        pl.figure(figsize=(10,6))
        pl.plot(episode_rewards, alpha=0.4, label='Episode Rewards')
        pl.plot(moving_avg, label=f'Moving Average ({window_size} episodes)')
        pl.xlabel('Episode')
        pl.ylabel('Reward')
        pl.title('Q-Learning Training Process')
        pl.legend()
        pl.grid(True, alpha=0.3)
        pl.show()
    
    except ImportError:
        print("Matplotlib not available. Skipping plot.")

