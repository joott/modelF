using ArgParse
using Distributions
using Random
using JLD2
using CodecZlib
using ParallelStencil

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--mass"
            help = "mass as an offset from the critical mass m²c"
            arg_type = Float64
            default = 0.0
        "--dt"
            help = "size of time step"
            arg_type = Float64
            default = 0.04
        "--imG"
            help = "im Gamma"
            arg_type = Float64
            default = 1.0
        "--gamma0"
            help = "coupling between the energy mode and the order parameter"
            arg_type = Float64
            default = 1.0
        "--g0"
            help = "mode coupling"
            arg_type = Float64
            default = 1.0
	"--h1"
	    help = "external field"
	    arg_type = Float64
	    default = 1.0
        "--rng"
            help = "seed for random number generation"
            arg_type = Int
            default = 0
        "--init"
            help = "path of .jld2 file with initial state"
            arg_type = String
        "--fp64"
            help = "flag to use Float64 type rather than Float32"
            action = :store_true
        "--cpu"
            help = "parallelize on CPU rather than GPU"
            action = :store_true
        "size"
            help = "side length of lattice"
            arg_type = Int
            required = true
    end

    return parse_args(s)
end

parsed_args = parse_commandline()

const cpu = parsed_args["cpu"]
!cpu && using CUDA

const FloatType = parsed_args["fp64"] ? Float64 : Float32
const ComplexType = complex(FloatType)
const ArrayType = cpu ? Array : CuArray
const SubArrayType = cpu ? SubArray : CuArray

const h1 = FloatType(parsed_args["h1"])
const h2 = FloatType(0.0) 
const h_psi = FloatType(0.0)
const λ = FloatType(4.0)
const imG = FloatType(parsed_args["imG"] )
const Γ = ComplexType(1.0 + 1.0im*imG)
const κ = FloatType(1.0)
const T = FloatType(1.0)

const L = parsed_args["size"]
const γ0 = FloatType(parsed_args["gamma0"])
const C0 = FloatType(1.0)
const g0 = FloatType(parsed_args["g0"])
const m² = FloatType(parsed_args["mass"])
const Δt = FloatType(parsed_args["dt"]/real(Γ))

const Δtdet = Δt/10
const Rate_phi = FloatType(sqrt(2.0*Δt*real(Γ)))
const Rate_psi  = FloatType(sqrt(2.0*Δt*κ))
const ξ = Normal(FloatType(0.0), FloatType(1.0))

const seed = parsed_args["rng"]
if seed != 0
    Random.seed!(seed)
    !cpu && CUDA.seed!(seed)
end

struct State
    u::ArrayType
    ϕ::SubArrayType
    ψ::SubArrayType
    State(u) = new(u, @view(u[:,:,:,1:2]), @view(u[:,:,:,3]))
end

function hotstart(n, n_components)
	u = rand(ξ, n, n, n, n_components)

    for i in 1:3
        u[:,:,:,i] .-= shuffle(u[:,:,:,i])
    end

    State(ArrayType(u))
end

init_arg = parsed_args["init"]

##
if isnothing(init_arg)

macro init_state() esc(:( state = hotstart(L, 3) )) end

else

macro init_state()
    file = jldopen(init_arg, "r")
    state = State(ArrayType(file["u"]))
    return esc(:( state = $state ))
end

end
##

##
@static if cpu

@init_parallel_stencil(Threads, FloatType, 3);

else

@init_parallel_stencil(CUDA, FloatType, 3);

end
##
