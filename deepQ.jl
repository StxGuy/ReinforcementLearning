using ReinforcementLearning
using DataStructures
using Statistics
using Plots
using Flux

using LinearAlgebra

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

function NN(num_in::Int, num_out::Int)
    return Chain(
        x -> x ./ 25f0 .- 0.5f0,
        Dense(num_in => 64, relu; init=Flux.glorot_uniform()),
        Dense(64 => 64, relu; init=Flux.glorot_uniform()),
        Dense(64 => num_out, identity; init=Flux.glorot_uniform())
    )
end

mutable struct Agent
    env         :: MazeEnv
    γ           :: Float64
    ε           :: Float64
    Q           :: Flux.Chain
    Qt          :: Flux.Chain
    optimiser   :: Flux.Optimiser
    update_f    :: Int
    batch_size  :: Int

    function Agent(;γ=0.95,ε=1.0)
        env = MazeEnv()
        S = state_space(env)
        A = action_space(env)
        nS = length(state(env))
        nA = length(A)
        Q = NN(nS,nA)
        Qt = deepcopy(Q)
        op = Flux.Optimiser(ClipValue(1),ADAM(1E-4))

        return new(env,γ,ε,Q,Qt,op,10,32)
    end
end

#---------------------------
# Epsilon-greedy algorithm
#---------------------------
function ε_greedy(agent::Agent,s::Int)
    if rand() < agent.ε
        a = rand(action_space(agent.env))
    else
        best_value = maximum(agent.Q([s]))
        best_actions = findall(x -> x == best_value, agent.Q([s]))
        a = rand(best_actions)[1]
    end

    return a
end

#--------------------
# Experience replay
#--------------------
struct Experience
    s       :: Int
    a       :: Int
    r       :: Float32
    s′      :: Int
    done    :: Bool
end

# Sample batch from experience replay
function sample_batch(buffer::CircularBuffer{Experience},batch_size::Int)
    indices = rand(1:length(buffer),batch_size)
    return [buffer[n] for n in indices]
end

#---------------------------------
# Train agent in the environment
#---------------------------------
function update!(agent::Agent,buffer::CircularBuffer{Experience})
    batch = sample_batch(buffer, agent.batch_size)
    batch_loss = 0.0

    for ex in batch
        # Target
        y = ex.r + agent.γ*maximum(agent.Qt([ex.s′]))

        ps = Flux.params(agent.Q)
        loss, ∇ = Flux.withgradient(ps) do
            Flux.huber_loss(y,agent.Q([ex.s])[ex.a])
        end
        Flux.Optimise.update!(agent.optimiser, ps, ∇)

        batch_loss += loss
    end

    return batch_loss/agent.batch_size
end

function train!(agent::Agent,num_episodes::Int)
    rewards_per_episode = Float64[]
    losses = Float64[]
    buffer = CircularBuffer{Experience}(10000)
    T = 1

    for episode in 1:num_episodes
        # Initialize
        ReinforcementLearning.reset!(agent.env)
        s = state(agent.env)

        # Episode
        episode_reward = 0.0
        episode_loss = 0.0

        step = 0
        while !is_terminated(agent.env)
            a = ε_greedy(agent,s)
            act!(agent.env,a)
            r = reward(agent.env)
            s′ = state(agent.env)
            done = is_terminated(agent.env)

            experience = Experience(s, a, r, s′, done)
            push!(buffer, experience)

            if length(buffer) >= agent.batch_size
                loss = update!(agent,buffer)
                episode_loss += loss
            end

            step += 1
            s = s′

            agent.ε = max(0.01,agent.ε*0.99995)

            episode_reward += r
        end

        push!(rewards_per_episode, episode_reward)
        push!(losses,step == 0 ? 0 : episode_loss/step)

        # Update target network
        if episode % agent.update_f == 0
            agent.Qt = deepcopy(agent.Q)
        end

        # Print statistics
        if episode % T == 0
            avg_reward = mean(rewards_per_episode[max(1,episode-(T-1)):episode])
            avg_loss = mean(losses[max(1,episode-(T-1)):episode])
            z = agent.ε
            println("Episode $episode, Reward (last $T): $avg_reward, Loss: $avg_loss, ε: $z")
        end
    end

    return rewards_per_episode,losses
end

#---------------------------------
# Evaluate agent
#---------------------------------
function evaluate(agent::Agent)
    reset!(agent.env)
    while !is_terminated(agent.env)
        s = state(agent.env)
        a = argmax(agent.Q([s]))    # Exploit only
        act!(agent.env,a)

        row = (s-1)÷5 + 1
        col = s - (row-1)*5
        println("$row x $col")
    end
end

#-------------------------------
# Plot results of training
#-------------------------------
function plot_training(rewards, losses)
    p1 = plot(rewards, label="Rewards", title="Training Progress")
    p2 = plot(losses, label="Loss", title="Training Loss")
    p = plot(p1, p2, layout=(2,1))
    display(p)
    readline()
end

#---------------------------------
# MAIN
#---------------------------------
agent = Agent()
rewards,losses = train!(agent,10000)

s = state(agent.env)
Q = agent.Q([s])
println(Q)

plot_training(rewards,losses)
evaluate(agent)
