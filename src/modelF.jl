#=
                      _      _   _____
  _ __ ___   ___   __| | ___| | |  ___|
 | '_ ` _ \ / _ \ / _` |/ _ \ | | |_
 | | | | | | (_) | (_| |  __/ | |  _|
 |_| |_| |_|\___/ \__,_|\___|_| |_|

=#

using Distributions
using Printf
using Random

include("initialize.jl")
include("simulation.jl")

function make_temp_arrays(state)
    (k1, k2, k3) = (similar(state.u), similar(state.u), similar(state.u))
    rk_temp = State(similar(state.u))
    (k1,k2,k3,rk_temp)
end

function save_state(filename, state, m², i=nothing)
    if isnothing(i)
        jldsave(filename, true; u=Array(state.u), m²=m²)
    else
        jldsave(filename, true; u=Array(state.u), m²=m², i=i)
    end
end
