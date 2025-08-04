#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include <vector>
#include "types.hpp"


class Environment {
private:
    int      width, height;
    int      move_count;
    int      move_limit;
    bool     done;
    Vector2D goal;
    std::vector<Vector2D> positions;
    
public:
    // Store the results of an episode
    struct StepResult {
        std::vector<Vector2D> state;
        float reward;
        bool  done;
    };
    
    // Constructor
    Environment(int w, int h, 
                const std::vector<Vector2D> &initial_positions, 
                Vector2D goal_pos = {4,4},
                int move_limit = 100);
    
    void        reset(const std::vector<Vector2D> &new_positions);
    Observation get_observation(int agent_id) const;
    StepResult  step(const std::vector<Action> &actions);
    float       compute_reward() const;
    bool        is_valid_position(const Vector2D &pos) const noexcept;
    bool        all_agents_at_goal() const;    
    bool        is_done() const noexcept;
    const::std::vector<Vector2D> &get_positions() const;    
};

#endif /* !ENVIRONMENT_H */
