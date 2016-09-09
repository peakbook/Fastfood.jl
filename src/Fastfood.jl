module Fastfood

using Hadamard
using Distributions

export FastfoodParam, FastfoodKernel

type FastfoodParam
    B::AbstractArray  # Binary scaling matrix
    G::AbstractArray  # Gaussian scaling matrix
    PI::AbstractArray # Permutation matrix
    S::AbstractArray  # Scaling matrix
    function FastfoodParam()
        new()
    end
end

function FastfoodParam(n::Integer, d::Integer;
                       rng::AbstractRNG=MersenneTwister())
    param = FastfoodParam()

    d = 2^ceil(Integer, log2(d))
    k = ceil(Integer, n/d)
    n = d*k

    param.B = rand(rng, [-1, 1], d, k)
    param.G = randn(rng, d, k);
    param.PI = zeros(Integer, d, k)
    for i in 1:k
        param.PI[:,i] = randperm(d)
    end
    param.S = rand(Chi(1), d * k)*norm(param.G)^(1/2)

    return param
end

function FastfoodKernel(X::Matrix, param::FastfoodParam;
                        sgm::Float64=1.0)
    d0, m = size(X)
    d = 2^ceil(Integer, log2(d0))

    if d == d0
        XX = X
    else
        # expand the data dimension to a power of 2.
        XX = zeros(d, m)
        XX[1:d0, :] = X
    end

    k = size(param.B, 2)
    n = d * k
    THT = zeros(n, m)

    for i = 1:k
        B = param.B[:, i]
        G = param.G[:, i]
        PI = param.PI[:, i]
        XX = broadcast(*, XX, B)
        T = fwht(XX, 1)
        T = T[PI, :]
        T = broadcast(*, T, G*d)
        THT[((i-1)*d+1):(i*d), :] = fwht(T, 1)
    end

    THT = broadcast(*, THT, param.S*d^(1/2))
    T = THT/sgm
    PHI = [cos(T); sin(T)]*n^(-1/2)

    return PHI
end

end # module
