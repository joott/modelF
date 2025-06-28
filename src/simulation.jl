include("helpers.jl")

"""
  Elementary stochastic step with the transfer of the momentum density (μ-th component) from the cell x1 to x2 
"""
function psi_step(ψ, ϕ, n, m, (i,j,k))
    xyz = ((2i + m)%L+1, j%L+1, k%L+1)
    x1 = (xyz[(3-n)%3+1], xyz[(4-n)%3+1], xyz[(5-n)%3+1])
    x2 = ((x1[1]-(n!=0))%L+1, (x1[2]-(n!=1))%L+1, (x1[3]-(n!=2))%L+1)

    norm = cos(2pi*rand())*sqrt(-2.0*log(rand()))
    q = Rate_psi * norm

    δH = (q * (ψ[x1...] - ψ[x2...]) + q^2)/C0 + γ0 * q * (ϕ[x1..., 1]^2 + ϕ[x1..., 2]^2 - ϕ[x2..., 1]^2 - ϕ[x2..., 2]^2)
    P = exp(-δH)
    r = rand()

    ψ[x1...] += q * (r<P)
    ψ[x2...] -= q * (r<P)
end

"""
  Computing the local change of energy in the cell x 
"""
function ΔH_phi(ϕ, μ, ψ, m², x, q)
    ϕold = ϕ[x..., μ]
    ϕt = ϕold + q
    Δϕ² = ϕt^2 - ϕold^2

    ∑nn = (ϕ[NNp(x[1]), x[2], x[3], μ] + ϕ[x[1], NNp(x[2]), x[3], μ] + ϕ[x[1], x[2], NNp(x[3]), μ]
         + ϕ[NNm(x[1]), x[2], x[3], μ] + ϕ[x[1], NNm(x[2]), x[3], μ] + ϕ[x[1], x[2], NNm(x[3]), μ])

    return 3Δϕ² - q * ∑nn + 0.5m² * Δϕ² + 0.25λ * (ϕt^4 - ϕold^4) + γ0 * ψ[x...] * Δϕ²
end

function phi_step(ϕ, μ, ψ, m², n, (i,j,k))
    x = ((2i + j + k + n)%L+1, j%L+1, k%L+1)

    norm = cos(2pi*rand())*sqrt(-2*log(rand()))
    q = Rate_phi * norm

    δH = ΔH_phi(ϕ, μ, ψ, m², x, q)

    ϕ[x..., μ] += q * (rand() < exp(-δH/T))
end

##
@static if cpu

function psi_sweep(ψ, ϕ, n, m)
    Threads.@threads for l in 0:L^3÷2-1
        i = l ÷ L^2
        j = (l ÷ L) % L
        k = l % L

        psi_step(ψ, ϕ, n, m, (i,j,k))
    end
end

function phi_sweep(ϕ, ψ, m², n)
    Threads.@threads for l in 0:L^3-1
        μ = l ÷ (L^3÷2) + 1
        i = (l ÷ L^2) % (L^3÷2)
        j = (l ÷ L) % L
        k = l % L

        phi_step(ϕ, μ, ψ, m², n, (i,j,k))
    end
end

else

function _psi_sweep(ψ, ϕ, n, m)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1
    stride = gridDim().x * blockDim().x

    for l in index:stride:L^3÷2-1
        i = l ÷ L^2
        j = (l ÷ L) % L
        k = l % L

        psi_step(ψ, ϕ, n, m, (i,j,k))
    end
end

function _phi_sweep(ϕ, ψ, m², n)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1
    stride = gridDim().x * blockDim().x

    for l in index:stride:L^3-1
        μ = l ÷ (L^3÷2) + 1
        i = (l ÷ L^2) % (L^3÷2)
        j = (l ÷ L) % L
        k = l % L

        phi_step(ϕ, μ, ψ, m², n, (i,j,k))
    end
end

_psi_sweep_temp = @cuda launch=false _psi_sweep(CuArray{FloatType}(undef,(L,L,L)), CuArray{FloatType}(undef,(L,L,L,2)), 0, 0)
_phi_sweep_temp = @cuda launch=false _phi_sweep(CuArray{FloatType}(undef,(L,L,L,2)), CuArray{FloatType}(undef,(L,L,L)), zero(FloatType), 0)

const N_psi = L^3÷2
config = launch_configuration(_psi_sweep_temp.fun)
const threads_psi = min(N_psi, config.threads)
const blocks_psi = cld(N_psi, threads_psi)

const N_phi = L^3÷2
config = launch_configuration(_phi_sweep_temp.fun)
const threads_phi = min(N_phi, config.threads)
const blocks_phi = cld(N_phi, threads_phi)

psi_sweep =  (ψ, ϕ, n, m) -> _psi_sweep_temp(ψ, ϕ, n, m; threads=threads_psi, blocks=blocks_psi)
phi_sweep = (ϕ, ψ, m², n) -> _phi_sweep_temp(ϕ, ψ, m², n; threads=threads_phi, blocks=blocks_phi)

end
##

function dissipative(state, m²)
    # psi update
    for n in 0:1, m in 0:1
        psi_sweep(state.ψ, state.ϕ, n, m)
    end

    # phi update
    for n in 0:1
        phi_sweep(state.ϕ, state.ψ, m², n)
    end
end

function thermalize(state, m², N)
    for _ in 1:N
        dissipative(state, m²)
    end
end
