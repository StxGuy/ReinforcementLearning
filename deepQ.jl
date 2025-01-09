using ReinforcementLearning
using Statistics

model = Chain(
    # Reshape
    x -> reshape(x,210,160,1,1),

    # Normalize pixels
    x -> float.(x) ./ 255.0,

    # Convolutional layers
    Conv((8,8), 1 => 32,  stride=(4,4), relu),  # 51x39x32
    Conv((4,4), 32 => 64, stride=(2,2), relu),  # 24x28x64
    Conv((3,3), 64 => 64, stride=(1,1), relu),  # 22x16x64

    # Flatten
    Flux.flatten,   # 22x16x64 = 22528

    # Fully connected layers
    Dense(22528,512,relu),
    Dense(512,4)
)

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




