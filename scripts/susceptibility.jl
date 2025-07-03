cd(@__DIR__)

using JLD2
using CodecZlib
using Printf

include("../src/modelF.jl")

function av(X)
    sum(X)/length(X)
end

function main()
    @init_state
    thermalize(state, m², L^3)
    maxt = L^3*4

    open("../data/output_mass_$(m²)_seed_$(seed).dat", "w") do io
        for i in 0:maxt
            ϕ1 = state.ϕ[:,:,:,1]
            ϕ2 = state.ϕ[:,:,:,2]
            Printf.@printf(io, "%i ", i*L)
            Printf.@printf(io, "%f %f %f ", av(ϕ1.^2), av(abs.(ϕ1))^2, av(ϕ1)^2)
            Printf.@printf(io, "%f %f %f ", av(ϕ2.^2), av(abs.(ϕ2))^2, av(ϕ2)^2)
            Printf.@printf(io, "%f %f\n", av(ϕ1.^2 .+ ϕ2.^2), av(sqrt.(ϕ1.^2 .+ ϕ2.^2))^2)

            if i%100==0
                Printf.flush(io)
                @show i
            end

            thermalize(state, m², L)
        end
    end
    
end

main()
