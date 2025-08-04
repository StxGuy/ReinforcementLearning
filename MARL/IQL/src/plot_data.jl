using Plots
gr()

y = []

println("Reading file...")
open("rewards.txt","r") do f
    while !eof(f)
        s = readline(f)
        push!(y,parse(Float64,s))
    end
end

#Plotting
println("Plotting...")
p = plot(y, ylabel="Episode Reward", label="", legend=false);
d = plot(p)
display(d)
readline()
