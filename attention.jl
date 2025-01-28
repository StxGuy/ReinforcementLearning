using Flux
using Statistics
using LinearAlgebra

# Source Embedding
#   1st dim: is it an animal?
#   2nd dim: does it fly?
cat  = [0.82;0.15;;]
rock = [0.12;0.08;;]
bird = [0.76;0.93;;]

X = [cat rock bird]'
n,d = size(X)

# Target Embedding
#   1st dim: sense of security
#   2nd dim: sense of control
comfort   = [0.9;0.1;;]
stability = [0.8;0.8;;]
freedom   = [0.3;1.0;;]

Y = [comfort stability freedom]'

# Weight matrices
dk = 3
dv = 2

Wq = randn(d,dk)
Wk = randn(d,dk)
Wv = randn(d,dv)
    
function forward(X,Y,Wq,Wk,Wv)
    Q = Y*Wq
    K = X*Wk
    V = X*Wv
    
    S = (Q*K') ./ sqrt(dk)
    A = softmax(S,dims=2)
    C = A*V
        
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

