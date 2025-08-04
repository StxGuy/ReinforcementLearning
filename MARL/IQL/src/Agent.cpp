#include <algorithm>
#include "Agent.hpp"

/*-------------
 * Constructor
 *-------------*/
Agent::Agent(int id, int env_width) : id(id), width(env_width) {}

/*--------
 * Policy 
 *--------*/
Action Agent::decide_action(const Observation &obs) const {
    if (obs.self_position == obs.goal)    return Action::doNothing;
    if (obs.self_position.x < obs.goal.x) return Action::goEast;
    if (obs.self_position.x > obs.goal.x) return Action::goWest;
    if (obs.self_position.y < obs.goal.y) return Action::goNorth;
    if (obs.self_position.y > obs.goal.y) return Action::goSouth;
        
    return Action::doNothing;
}

/*-----------------------
 * Get agent's ID number
 *-----------------------*/
int Agent::get_id() const {
    return id;
}

/*---------------------------------------------
 * Get Q values for all actions given a state
 *---------------------------------------------*/
std::vector<float> &Agent::getQ(const Vector2D &state) {
    int state_id = state.y*width + state.x;
    
    auto it = Q_table.find(state_id);    
    if (it == Q_table.end()) {
        Q_table[state_id] = std::vector<float>(5, 0.0f);
        return Q_table[state_id];
    }
    
    return it->second;
}

/*--------------
 * IQL Learning
 *--------------*/
void Agent::learn(const  Observation &state, 
                  Action action, 
                  float  reward, 
                  const  Observation &next_state) {
    
    const int action_index = static_cast<int>(action);
    auto &Q = getQ(state.self_position);
    const auto &Qn = getQ(next_state.self_position);
    
            
    // Q(s,a) <- Q(s,a) + eta.[r + g.max(a', Q(s',a')) - Q(s,a)]
    
    // First, compute max_a' Q(s',a')
    const float maxQ = *std::max_element(Qn.begin(), Qn.end());
        
    // Q-learning update
    Q[action_index] += eta*(reward + gamma*maxQ - Q[action_index]);
}

/*---------------------------------------------
 * Decide what action to take: Epsilon-greedy
 *---------------------------------------------*/
Action Agent::decide_action(const Observation &obs) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> &Q_values = getQ(obs.self_position);
    
    if (dist(rng) < epsilon) {     // Exploration
        int random_action = rand() % Q_values.size();
        return static_cast<Action>(random_action);
    }
    else {                   
        int best_action = std::distance(Q_values.begin(),
                                        std::max_element(Q_values.begin(), Q_values.end()));
        return static_cast<Action>(best_action);
    }
}
