using Gym, Flux, Plots, Statistics

# Actions: {0,1}
# State: 
#        Cart position........ [-4.8, 4.8]
#        Cart velocity........ (-inf, inf)
#        Pole angle........... [-24°, 24°]
#        Pole angular velocity (-inf,inf)

#--------------------------------------------------
# Sample an action using policy
#--------------------------------------------------
function sample_action(policy,state)
    logits = policy(state)
    probs = softmax(logits)
    
    return rand() < probs[1] ? 1 : 2
end

#--------------------------------------------------
# Run a single episode applying the policy
#--------------------------------------------------
function run_episode(policy,env)
    states, actions, rewards = [], [], []
        
    s,_ = reset!(env)
        
    while true
        a = sample_action(policy,s)
        
        push!(states,s)
        push!(actions,a)
        
        s,r,terminated,truncated = step!(env, a-1)
        push!(rewards,r)
        
        if terminated || truncated
            break
        end
    end
    
    return states, actions, rewards #./ maximum(abs.(rewards))
end

#--------------------------------------------------
# Compute episode returns and normalize them 
# for stability.
#--------------------------------------------------
function compute_returns(rewards,γ)
    T = length(rewards)
    returns = zeros(T)
    R = 0.0
    @simd for t in T:-1:1
        R = rewards[t] + γ*R
        returns[t] = R
    end
    
    # Return normalized returns for stability
    return (returns .- mean(returns)) ./ (std(returns) + 1f-8)
end

#--------------------------------------------------
# Policy loss
#--------------------------------------------------
function log_prob(policy,state,action)
    logits = policy(state)
    return Flux.logsoftmax(logits)[action]
end

function policy_loss(policy,states,actions,returns)
    loss = sum([-log_prob(policy,s,a)*R for (s,a,R) in zip(states,actions,returns)])
    return loss
end

#--------------------------------------------------
# Update policy
#--------------------------------------------------
function update_policy(policy,states,actions,returns,opt)
    θ = Flux.params(policy)
    
    ∇ = Flux.gradient(θ) do
        policy_loss(policy,states,actions,returns)
    end
    
    Flux.Optimise.update!(opt,θ,∇)
end

#--------------------------------------------------
# Train policy
#--------------------------------------------------
function train!(env,policy,opt,hyperparams)
    G = []
    for episode in 1:hyperparams[:num_episodes]
        states,actions,rewards = run_episode(policy, env)
        returns = compute_returns(rewards, hyperparams[:γ])
        update_policy(policy,states,actions,returns,opt)
        
        if episode % 10 == 0
            println("Episode $episode: Total reward = $(sum(rewards))")
        end
        
        push!(G,sum(rewards))
    end
    
    return G
end

#--------------------------------------------------
# Show one trained episode
#--------------------------------------------------
function showResult(env,policy)
    s,_ = reset!(env)
        
    anim = @animate while true
        a = sample_action(policy,s)
        
        s,r,terminated,truncated = step!(env, a-1)
        
        x = s[1]
        Θ = s[3]
        
        # Plot configurations
        plot(xlims=(-1,1),
             ylims=(-0.1,2.5),
             legend=false,
             border=:none)
        
        # Plot the cart
        plot!([x-0.5,x-0.5,x+0.5,x+0.5],[-0.05,0,0,-0.05];
             seriestype=:shape
             )
        
        # Plot the pole
        plot!([x,x+2sin(Θ)],[0,2cos(Θ)];
              linewidth=3
              )
        
        
        if terminated || truncated
            break
        end
    end
    
    gif(anim,"REINFORCE.gif",fps=10)
end

#--------------------------------------------------
# MAIN
#--------------------------------------------------
# Hyper Parameters
hyperparams = Dict(
    :γ => 0.99,
    :η => 0.001,
    :num_episodes => 1000
)

# Initialize environment
env = GymEnv("CartPole-v1")

# Policy returns logits
policy = Chain(
    Dense(4,32,relu),
    Dense(32,2),
)

# Train policy
opt = Descent(hyperparams[:η])
G = train!(env,policy,opt,hyperparams)

# Plot cumulated rewards
plot(G)
gui()
readline()

# Show one trained episode
println("Generating episode...")
showResult(env,policy)


