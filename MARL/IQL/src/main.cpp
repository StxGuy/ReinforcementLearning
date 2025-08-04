#include <iostream>
#include <fstream>

#include "Agent.hpp"
#include "Environment.hpp"


// Runs one episode
float run_episode(Environment &env, 
                  std::vector<Agent> &agents,
                  const std::vector<Vector2D> &initial_positions,
                  bool do_learning, 
                  bool accumulate_rewards,
                  std::ostream &out = std::cout,
                  bool print_steps = false) 
{
    env.reset(initial_positions);
    
    // Last observations and actions
    std::vector<Observation> last_obs(agents.size());
    std::vector<Action>      last_actions(agents.size());
    
    for(size_t i = 0; i < agents.size(); ++i) {
        last_obs[i] = env.get_observation(agents[i].get_id());
    }
    
    // Total reward per episode
    float total_rewards = 0.0f;
    
    // Run until done
    while(!env.is_done()) {
        std::vector<Action> actions;
        
        // Take actions according to exploration-exploitation algorithm
        for(size_t i = 0; i < agents.size(); ++i) {
            Action action = agents[i].decide_action(last_obs[i]);
            actions.push_back(action);
            last_actions[i] = action;
        }            
        
        // Step environment
        Environment::StepResult result = env.step(actions);
        
        // Optional: Print results
        if (print_steps) {
            out << "Step: " << result.state.size() << " | Reward: " 
            << result.reward << " | Done: " << result.done << std::endl;
            
            for(size_t i = 0; i < result.state.size(); ++i) {
                out << "Agent " << i << " at (" << result.state[i].x
                << "," << result.state[i].y << ")" << std::endl;
            }
        }
        
        // Get new observations for next state
        std::vector<Observation> new_obs(agents.size());
        for(size_t i = 0; i < agents.size(); ++i) {
            new_obs[i] = env.get_observation(agents[i].get_id());
        }
        
        // Optional: Learn
        if (do_learning) {
            // Agents learn from experience (last_obs, last_actions, reward, new_obs)
            for(size_t i = 0; i < agents.size(); ++i) {
                agents[i].learn(last_obs[i], last_actions[i], result.reward, new_obs[i]);
            }
        }
        
        // Optional: Accumulate rewards
        if (accumulate_rewards) {
            total_rewards += result.reward;
        }
        
        // Update last observation
        last_obs = new_obs;
    }
    
    return total_rewards;
}


// Main function
int main() {
    const int   NUM_EPISODES = 3000;
    
    try {
        std::vector<Vector2D> initial_positions = {{2,2}, {1,5}};
        Environment env(10, 10, initial_positions, {4,4}, 100);
        std::vector<Agent> agents = {Agent(0,10), Agent(1,10)};
        
        std::ofstream rewards_file("rewards.txt");
        
       // Cycle through episodes
        for(size_t episode = 0; episode < NUM_EPISODES; ++episode) {
            std::cout << "Episode: " << episode + 1 << " begins. ";
            
            float total_rewards = run_episode(env, agents, initial_positions, true, true);
            
            std::cout << "** Total reward: " << total_rewards << std::endl;
            
            if (rewards_file.is_open()) {
                rewards_file << total_rewards << std::endl;
            }
        }
        
        rewards_file.close();
        
        // Show results
        std::cout << "-----------------" << std::endl;
        std::cout << " Results" << std::endl;
        std::cout << "-----------------" << std::endl;
        run_episode(env, agents, initial_positions, false, false, std::cout, true);
        
    } catch(const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
       
    return 0;
}
