using Flux
using Statistics
using LinearAlgebra

# Source Embedding
#   1st dim: is it an animal?
#   2nd dim: does it fly?
cat  = [0.82;0.15;;]
rock = [0.12;0.08;;]
bird = [0.76;0.93;;]

X = [cat rock bird]

# Target Embedding
#   1st dim: sense of security
#   2nd dim: sense of control
comfort = [0.9;0.1;;]
stability = [0.8;0.8;;]
freedom = [0.3;1.0;;]

Y = [comfort stability freedom]

# Weight matrices
Wq = randn(2,2)
Wk = randn(2,2)
Wv = randn(2,1)
    
function forward(X,Y,Wq,Wk,Wv)
    Q = Wq'*Y
    K = Wk'*X
    V = Wv'*X
    
    S = (Q'*K) ./ sqrt(size(Q,1))
    A = softmax(S,dims=2)
    C = A*V'
        
    return A,C
end


opt = Adam(0.001)
Ax = [1.0 0.0 0.0;0.0 1.0 0.0;0.0 0.0 1.0]

for epoch in 1:10000
    p = Flux.params(Wq,Wk,Wv)
    L,∇ = Flux.withgradient(p) do
        A,C = forward(X,Y,Wq,Wk,Wv)
        mean(abs2,A - Ax)
    end
    
    println(L)
        
    Flux.Optimise.update!(opt,p,∇)
end

A,C = forward(X,Y,Wq,Wk,Wv)

