#ifndef AGENT_H
#define AGENT_H

#include <unordered_map>
#include <random>
#include "types.hpp"

class Agent {
private:
    int id;
    std::unordered_map<int, std::vector<float>> Q_table; 
    // first argument is a key for the state
    // second argument is a set of Q values for each action
    // using an unordered_map helps save memory. 
    // There's no need to store the whole matrix.
    
    // Learning parameters
    float eta = 0.1f;     // Learning rate
    float gamma = 0.99f;    // Discount factor
    float epsilon = 0.1f;   // epsilon-greedy exploration
    
    // Information about the environment
    int width;
    
    std::mt19937 rng{std::random_device{}()};
    
public:
    // Constructor
    explicit Agent(int id, int env_width);
    Action decide_action(const Observation &obs) const;    
    int get_id() const;
    std::vector<float> &getQ(const Vector2D &state);
    void learn(const Observation &state, Action action, float reward, const Observation &next_stte);
    Action decide_action(const Observation &obs);
};

#endif /* !AGENT_H */
