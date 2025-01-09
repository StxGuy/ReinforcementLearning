using ReinforcementLearning
using Statistics

#=========================================#
# MAZE ENVIRONMENT

# Define the maze elements
@enum MazeElement EMPTY=0 WALL=1 TRAP=2 TREASURE=3 AGENT=4

mutable struct MazeEnv <: AbstractEnv
    grid        ::Matrix{MazeElement}
    agent_pos   ::Tuple{Int,Int}
    initial_pos ::Tuple{Int,Int}
    done        ::Bool
    reward      ::Float64

    function MazeEnv()
        # Create a 5x5 maze with walls, traps, and treasure
        grid = fill(EMPTY, (5,5))

        # Add walls
        grid[1,3] = WALL
        grid[2,1] = WALL
        grid[3,3] = WALL
        grid[3,4] = WALL
        grid[4,2] = WALL

        # Add traps
        grid[2,4] = TRAP
        grid[4,3] = TRAP

        # Add treasure
        grid[5,5] = TREASURE

        # Set initial position
        initial_pos = (1,1)

        new(grid, initial_pos, initial_pos, false, 0.0)
    end
end

# Required interface implementations
RLBase.state_space(env::MazeEnv) = Base.OneTo(25)  # Flattened 5x5 grid
RLBase.action_space(env::MazeEnv) = Base.OneTo(4)  # Up, Right, Down, Left
RLBase.reward(env::MazeEnv) = env.reward
RLBase.is_terminated(env::MazeEnv) = env.done
RLBase.state(env::MazeEnv) = (env.agent_pos[1]-1) * 5 + env.agent_pos[2]

function RLBase.reset!(env::MazeEnv)
    env.agent_pos = env.initial_pos
    env.done = false
    env.reward = 0.0
    return state(env)
end

function RLBase.act!(env::MazeEnv, action::Int)
    if env.done
        return state(env)
    end

    # Calculate new position based on action
    # 1: Up, 2: Right, 3: Down, 4: Left
    old_pos = env.agent_pos
    new_pos = if action == 1
        (old_pos[1] - 1, old_pos[2])
    elseif action == 2
        (old_pos[1], old_pos[2] + 1)
    elseif action == 3
        (old_pos[1] + 1, old_pos[2])
    else  # action == 4
        (old_pos[1], old_pos[2] - 1)
    end

    # Check if new position is valid
    if new_pos[1] < 1 || new_pos[1] > 5 || new_pos[2] < 1 || new_pos[2] > 5
        env.reward = -1.0  # Penalty for hitting boundary
        return state(env)
    end

    # Check what's in the new position
    cell = env.grid[new_pos...]
    if cell == WALL
        env.reward = -1.0  # Penalty for hitting wall
    elseif cell == TRAP
        env.agent_pos = new_pos
        env.reward = -10.0  # Big penalty for falling into trap
        env.done = true
    elseif cell == TREASURE
        env.agent_pos = new_pos
        env.reward = 10.0  # Big reward for finding treasure
        env.done = true
    else  # EMPTY
        env.agent_pos = new_pos
        env.reward = -0.1  # Small penalty for each step to encourage efficiency
    end

    return state(env)
end

#=========================================#

mutable struct Agent
    env :: MazeEnv
    α   :: Float64
    γ   :: Float64
    ε   :: Float64
    Q   :: Matrix{Float64}

    function Agent(;α=0.001,γ=0.99,ε=0.01)
        env = MazeEnv()
        S = state_space(env)
        A = action_space(env)
        nS = length(S)
        nA = length(A)
        Q = randn(nS,nA).*0.1

        return new(env,α,γ,ε,Q)
    end
end

#---------------------------
# Epsilon-greedy algorithm
#---------------------------
function ε_greedy(agent::Agent,s::Int)
    if rand() < agent.ε
        a = rand(action_space(agent.env))
    else
        best_value = maximum(agent.Q[s,:])
        best_actions = findall(x -> x == best_value, agent.Q[s,:])
        a = rand(best_actions)
    end

    return a
end

#---------------------------------
# Train agent in the environment
#---------------------------------
function train!(agent::Agent,num_episodes::Int)
    rewards_per_episode = Float64[]

    for episode in 1:num_episodes
        # Initialize
        reset!(agent.env)
        s = state(agent.env)
        a = ε_greedy(agent,s)

        # Episode
        episode_reward = 0.0

        while !is_terminated(agent.env)
            act!(agent.env,a)
            r = reward(agent.env)
            s′ = state(agent.env)
            a′ = argmax(agent.Q[s′,:])

            agent.Q[s,a] += agent.α*(r + agent.γ*agent.Q[s′,a′] - agent.Q[s,a])

            s,a = s′,a′

            episode_reward += r
        end

        push!(rewards_per_episode, episode_reward)

        if episode % 100 == 0
            avg_reward = mean(rewards_per_episode[max(1,episode-99):episode])
            println("Episode $episode, Average Reward (last 100): $avg_reward")
        end
    end

    return rewards_per_episode
end

#---------------------------------
# Evaluate agent
#---------------------------------
function evaluate(agent::Agent)
    reset!(agent.env)
    while !is_terminated(agent.env)
        s = state(agent.env)
        a = argmax(agent.Q[s,:])    # Exploit only
        act!(agent.env,a)

        row = (s-1)÷5 + 1
        col = s - (row-1)*5
        println("$row x $col")
    end
end

#---------------------------------
# MAIN
#---------------------------------
agent = Agent()
train!(agent,10000)

evaluate(agent)




