#include <stdexcept>
#include <algorithm>
#include "Environment.hpp"

/*-------------
 * Constructor
 *-------------*/    
Environment::Environment(int w, int h, 
                         const std::vector<Vector2D> &initial_positions, 
                         Vector2D goal_pos,
                         int move_limit) : width(w), height(h), 
                            move_count(0),
                            move_limit(move_limit),
                            done(false),
                            goal(goal_pos),
                            positions(initial_positions) {
    if (positions.size() < 2) {
        throw std::invalid_argument("At least two agents required!");
    }
}

/*----------------------------
 * Check if position is valid 
 *----------------------------*/
bool Environment::is_valid_position(const Vector2D &pos) const noexcept {
    return pos.x >= 0 || pos.x < width || pos.y >= 0 || pos.y < height;
}

/*-----------------------
 * Reset the environment
 *-----------------------*/
void Environment::reset(const std::vector<Vector2D> &new_positions) {
    if (new_positions.size() != positions.size()) {
        throw std::invalid_argument("Mismatched reset positions!");
    }
    for (const auto &pos : new_positions){
        if (!is_valid_position(pos)) {
            throw std::invalid_argument("Invalid reset position!");
        }
    }
    
    positions  = new_positions;
    move_count = 0;
    done       = false;
}

/*--------------------------------------
 * Get observation from the environment
 *--------------------------------------*/
Observation Environment::get_observation(int agent_id) const {
    Observation obs;
    obs.self_position = positions[agent_id];
    for(size_t i = 0; i < positions.size(); ++i) {
        if (static_cast<int>(i) != agent_id) {
            obs.other_positions.push_back(positions[i]);
        }
    }
    
    obs.move_count = move_count;
    obs.goal       = goal;
    
    return obs;
}

/*----------------------------------------------------
 * Step into the actions of agents in the environment
 *----------------------------------------------------*/
Environment::StepResult Environment::step(const std::vector<Action> &actions) {
    if (done) {
        throw std::runtime_error("Environment is done!");
    }        
    if (actions.size() != positions.size()) {
        throw std::invalid_argument("Action count mismatch!");
    }
    
    // Check if moves are allowed
    move_count ++;
    bool any_illegal_move = false;
    
    for(size_t i = 0; i < positions.size(); ++i) {
        Vector2D new_pos = positions[i].move(actions[i]);
        if (is_valid_position(new_pos)) {
            positions[i] = new_pos;
        } 
        else {
            any_illegal_move = true;
        }
    }
    
    // Prepare result
    float reward = compute_reward();
    done = any_illegal_move || all_agents_at_goal() || move_count >= move_limit;
    
    // IQL Learning
    for(size_t i = 0; i < positions.size(); ++i) {
        
    }
    
    return {positions, reward, done};
}

/*----------------
 * Compute reward
 *----------------*/
float Environment::compute_reward() const {
    if (done && move_count > 0) {
        if (std::any_of(positions.begin(), positions.end(), 
                        [this](const Vector2D &p) {
                            return !is_valid_position(p);})) {
            return -1.0f;
        }
    }
            
    // Check if all agents are at the same position
    if (positions[0] == positions[1]) {
        return (positions[0] == goal) ? 10.0f : -1.0f; // is it the goal?
    }
    return -0.1f;
}

/*-----------------------------
 * Are all agents at the goal?
 *-----------------------------*/
bool Environment::all_agents_at_goal() const {
    return std::all_of(positions.begin(), positions.end(),
                       [this](const Vector2D &p) { 
                            return p == goal;});
}
    
/*---------------------------
 * Get all agent's positions 
 *---------------------------*/    
const::std::vector<Vector2D> &Environment::get_positions() const {
    return positions;
}
    
/*--------------------------
 * Is the environment done? 
 *--------------------------*/   
bool Environment::is_done() const noexcept{
    return done;
}
   
