#= 

.88b  d88.  .d88b.  d8888b. d88888b db           db   db 
88'YbdP`88 .8P  Y8. 88  `8D 88'     88           88   88 
88  88  88 88    88 88   88 88ooooo 88           88ooo88 
88  88  88 88    88 88   88 88~~~~~ 88           88~~~88 
88  88  88 `8b  d8' 88  .8D 88.     88booo.      88   88 
YP  YP  YP  `Y88P'  Y8888D' Y88888P Y88888P      YP   YP 

=# 

using Distributions
using Printf
using Random

include("initialize.jl")
include("simulation.jl")

function kinetic_energy(ϕ)
    0.25 * sum(3 * ϕ.^2 - circshift(ϕ, (1,0,0)) .* circshift(ϕ, (-1,0,0))
                        - circshift(ϕ, (0,1,0)) .* circshift(ϕ, (0,-1,0))
                        - circshift(ϕ, (0,0,1)) .* circshift(ϕ, (0,0,-1)))
end

function energy(state)
    K = kinetic_energy(state.ϕ)
    (π1, π2, π3) = view_tuple(state.π)
    K + sum(0.5 * (π1.^2 + π2.^2 + π3.^2 + 1/2 * m² * state.ϕ.^2 + λ/4 * state.ϕ.^2))
end

function make_temp_arrays(state)
    (k1, k2, k3) = (similar(state.u), similar(state.u), similar(state.u))
    rk_temp = State(similar(state.u))
    fft_temp = ArrayType{ComplexType}(undef, (L,L,L,3))
    (k1,k2,k3,rk_temp,fft_temp)
end

function save_state(filename, state, m², i=nothing)
    if isnothing(i)
        jldsave(filename, true; u=Array(state.u), m²=m²)
    else
        jldsave(filename, true; u=Array(state.u), m²=m², i=i)
    end
end
