using ReinforcementLearning
using Random

# Define the FrozenLake environment
mutable struct FrozenLakeEnv <: AbstractEnv
    size::Tuple{Int, Int}        # Grid dimensions (rows, cols)
    grid::Matrix{Char}           # The grid: 'S'=start, 'F'=frozen, 'H'=hole, 'G'=goal
    agent_pos::CartesianIndex{2} # Current agent position
    start_pos::CartesianIndex{2} # Starting position
    goal_pos::CartesianIndex{2}  # Goal position
    holes::Set{CartesianIndex{2}} # Hole positions
    slip_prob::Float64           # Probability of slipping (not going intended direction)
    terminated::Bool             # Whether episode is terminated
    rng::AbstractRNG             # Random number generator
    max_steps::Int               # Maximum steps per episode
    current_steps::Int           # Current step count
    
    # Actions: 1=Left, 2=Down, 3=Right, 4=Up
    action_meanings::Vector{String}
    
    function FrozenLakeEnv(;
        size::Tuple{Int, Int} = (4, 4),
        slip_prob::Float64 = 0.8,  # Probability of NOT slipping
        is_slippery::Bool = true,
        map_name::String = "4x4",
        max_steps::Int = 100,
        rng::AbstractRNG = Random.GLOBAL_RNG
    )
        
        # Create the grid based on map_name
        if map_name == "4x4"
            grid_layout = [
                "SFFF";
                "FHFH";
                "FFFH";
                "HFFG"
            ]
        elseif map_name == "8x8"
            grid_layout = [
                "SFFFFFFF";
                "FFFFFFFF";
                "FFFHFFFF";
                "FFFFFHFF";
                "FFFHFFFF";
                "FHHFFFHF";
                "FHFFHFHF";
                "FFFHFFFG"
            ]
        else
            # Custom size with random holes
            grid_layout = generate_random_map(size, rng)
        end
        
        rows, cols = length(grid_layout), length(grid_layout[1])
        grid = Matrix{Char}(undef, rows, cols)
        
        # Parse the grid and find special positions
        start_pos = CartesianIndex(1, 1)
        goal_pos = CartesianIndex(1, 1)
        holes = Set{CartesianIndex{2}}()
        
        for i in 1:rows, j in 1:cols
            char = grid_layout[i][j]
            grid[i, j] = char
            
            if char == 'S'
                start_pos = CartesianIndex(i, j)
            elseif char == 'G'
                goal_pos = CartesianIndex(i, j)
            elseif char == 'H'
                push!(holes, CartesianIndex(i, j))
            end
        end
        
        # Adjust slip probability if not slippery
        actual_slip_prob = is_slippery ? slip_prob : 1.0
        
        env = new(
            (rows, cols),
            grid,
            start_pos,
            start_pos,
            goal_pos,
            holes,
            actual_slip_prob,
            false,
            rng,
            max_steps,
            0,
            ["Left", "Down", "Right", "Up"]
        )
        
        return env
    end
end

# Generate a random map for custom sizes
function generate_random_map(size::Tuple{Int, Int}, rng::AbstractRNG)
    rows, cols = size
    grid = fill('F', rows, cols)
    
    # Set start and goal
    grid[1, 1] = 'S'
    grid[rows, cols] = 'G'
    
    # Add some random holes (about 10% of cells)
    num_holes = max(1, div(rows * cols, 10))
    for _ in 1:num_holes
        r, c = rand(rng, 2:rows-1), rand(rng, 2:cols-1)
        if grid[r, c] == 'F'  # Don't overwrite start, goal, or existing holes
            grid[r, c] = 'H'
        end
    end
    
    return [String(grid[i, :]) for i in 1:rows]
end

# ReinforcementLearning.jl interface implementations
RLBase.action_space(env::FrozenLakeEnv) = Base.OneTo(4)  # 4 actions: Left, Down, Right, Up
RLBase.state_space(env::FrozenLakeEnv) = Base.OneTo(prod(env.size))
RLBase.state(env::FrozenLakeEnv) = pos_to_state(env, env.agent_pos)
RLBase.is_terminated(env::FrozenLakeEnv) = env.terminated
RLBase.reward(env::FrozenLakeEnv) = env.agent_pos == env.goal_pos ? 1.0 : 0.0

function RLBase.reset!(env::FrozenLakeEnv)
    env.agent_pos = env.start_pos
    env.terminated = false
    env.current_steps = 0
    return nothing
end

# Convert 2D position to 1D state
function pos_to_state(env::FrozenLakeEnv, pos::CartesianIndex{2})
    rows, cols = env.size
    return (pos[1] - 1) * cols + pos[2]
end

# Convert 1D state to 2D position
function state_to_pos(env::FrozenLakeEnv, state::Int)
    rows, cols = env.size
    row = div(state - 1, cols) + 1
    col = mod(state - 1, cols) + 1
    return CartesianIndex(row, col)
end

# Get the intended direction vector for each action
function get_direction(action::Int)
    directions = [
        CartesianIndex(0, -1),  # Left
        CartesianIndex(1, 0),   # Down
        CartesianIndex(0, 1),   # Right
        CartesianIndex(-1, 0)   # Up
    ]
    return directions[action]
end

# Get perpendicular actions for slipping
function get_perpendicular_actions(action::Int)
    perpendiculars = [
        [2, 4],  # Left -> Down, Up
        [1, 3],  # Down -> Left, Right
        [2, 4],  # Right -> Down, Up
        [1, 3]   # Up -> Left, Right
    ]
    return perpendiculars[action]
end

# Check if position is valid (within bounds)
function is_valid_position(env::FrozenLakeEnv, pos::CartesianIndex{2})
    rows, cols = env.size
    return 1 <= pos[1] <= rows && 1 <= pos[2] <= cols
end

# Main step function
function (env::FrozenLakeEnv)(action::Int)
    if env.terminated
        @warn "Episode is already terminated. Call reset! first."
        return 0.0, true
    end
    
    env.current_steps += 1
    
    # Determine actual action (with slipping)
    actual_action = action
    
    if rand(env.rng) > env.slip_prob
        # Agent slips - choose a perpendicular direction
        perpendicular_actions = get_perpendicular_actions(action)
        actual_action = rand(env.rng, perpendicular_actions)
    end
    
    # Calculate new position
    direction = get_direction(actual_action)
    new_pos = env.agent_pos + direction
    
    # Check bounds - if out of bounds, stay in place
    if is_valid_position(env, new_pos)
        env.agent_pos = new_pos
    end
    
    # Check termination conditions
    reward = 0.0
    if env.agent_pos == env.goal_pos
        reward = 1.0
        env.terminated = true
    elseif env.agent_pos in env.holes
        reward = 0.0
        env.terminated = true
    elseif env.current_steps >= env.max_steps
        reward = 0.0
        env.terminated = true
    end
    
    return reward, env.terminated
end

# Rendering functions
function render(env::FrozenLakeEnv; mode::String = "human")
    if mode == "human"
        render_human(env)
    elseif mode == "ansi"
        return render_ansi(env)
    else
        error("Unsupported render mode: $mode")
    end
end

function render_human(env::FrozenLakeEnv)
    println(render_ansi(env))
end

function render_ansi(env::FrozenLakeEnv)
    rows, cols = env.size
    output = ""
    
    # Top border
    output *= "+" * repeat("-", cols * 2 + 1) * "+\n"
    
    for i in 1:rows
        output *= "|"
        for j in 1:cols
            pos = CartesianIndex(i, j)
            
            if pos == env.agent_pos
                if pos == env.goal_pos
                    output *= " A"  # Agent at goal
                else
                    output *= " A"  # Agent
                end
            else
                cell = env.grid[i, j]
                if cell == 'S'
                    output *= " S"  # Start
                elseif cell == 'F'
                    output *= "  "  # Frozen (empty)
                elseif cell == 'H'
                    output *= " H"  # Hole
                elseif cell == 'G'
                    output *= " G"  # Goal
                end
            end
        end
        output *= " |\n"
    end
    
    # Bottom border
    output *= "+" * repeat("-", cols * 2 + 1) * "+"
    
    return output
end

# Utility function to get action name
function get_action_name(env::FrozenLakeEnv, action::Int)
    return env.action_meanings[action]
end

# Example usage and testing
function demo_frozen_lake()
    println("Creating FrozenLake Environment...")
    
    # Create environment
    env = FrozenLakeEnv(slip_prob = 0.8, is_slippery = true, map_name = "4x4")
    
    println("Environment created!")
    println("Grid size: ", env.size)
    println("Action space: ", action_space(env))
    println("State space: ", state_space(env))
    
    # Reset and show initial state
    reset!(env)
    println("\nInitial state:")
    render(env)
    
    println("\nRunning a random episode...")
    reset!(env)
    step = 0
    total_reward = 0.0
    
    while !is_terminated(env) && step < 20
        step += 1
        action = rand(action_space(env))
        reward, terminated = env(action)
        total_reward += reward
        
        println("\nStep $step:")
        println("Action: $(get_action_name(env, action)) ($action)")
        println("Reward: $reward")
        println("Terminated: $terminated")
        render(env)
        
        if terminated
            if reward > 0
                println("🎉 Reached the goal!")
            else
                println("💀 Fell into a hole or ran out of time!")
            end
            break
        end
    end
    
    println("\nTotal reward: $total_reward")
    return env
end

# Run the demo
demo_frozen_lake()
