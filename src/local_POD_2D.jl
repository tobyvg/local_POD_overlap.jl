using LinearAlgebra
using FFTW


using Distributions
using JLD

using Flux
using Random

using ProgressBars
using Zygote
using CUDA



export reshape_for_local_SVD, carry_out_local_SVD, local_to_global_modes, compute_overlap_matrix, add_filter_to_modes, gen_projection_operators, true_W, true_R,projection_operators_struct





function reshape_for_local_SVD(input,MP; subtract_average = false)
    J = MP.J
    I = MP.I
    dims = MP.fine_mesh.dims
    dims = length(J)



    offsetter = [J...]
    loop_over = gen_permutations(I)
    data = []
    for i in 1:size(loop_over)[1]

        i = loop_over[i,:]
        first_index = offsetter .* (i .-1 ) .+ 1
        second_index = offsetter .* (i)
        index = [(first_index[i]:second_index[i]) for i in 1:dims]
        index = [index...,(:),(:)]
        to_push = input[index...]
        if subtract_average
            to_push .-= mean(to_push,dims = collect(1:dims))
        end
        push!(data,to_push)
    end

    return cat(data...,dims = dims + 2)
end




function carry_out_local_SVD(input,MP;subtract_average = false)
    UPC = MP.coarse_mesh.UPC
    J = MP.J
    I = MP.I
    dims = MP.fine_mesh.dims
    reshaped_input = reshape_for_local_SVD(input,MP,subtract_average = subtract_average)

    vector_input = reshape(reshaped_input,(prod(size(reshaped_input)[1:end-1]),size(reshaped_input)[end]))

    SVD = svd(vector_input)
    return reshape(SVD.U,(J...,UPC,Int(size(SVD.U)[end]))),SVD.S
end



function local_to_global_modes(modes,MP)

    number_of_modes = size(modes)[end]
    UPC = MP.coarse_mesh.UPC
    J = MP.J
    I = MP.I
    dims = MP.fine_mesh.dims

    if MP.fine_mesh.use_GPU
        some_ones = cu(ones(size(modes)[1:end]...,prod(I)))
    else
        some_ones = ones(size(modes)[1:end]...,prod(I))
    end

    global_modes = modes .* some_ones

    original_dims = collect(1:length(size(global_modes)))
    permuted_dims = copy(original_dims)
    permuted_dims[end] = original_dims[end-1]
    permuted_dims[end-1] = original_dims[end]

    global_modes = permutedims(global_modes,permuted_dims)

    global_modes = reshape(global_modes,(J...,UPC, I...,number_of_modes))

    output = zeros(I..., J...,UPC,number_of_modes)


    loop_over = gen_permutations((J...,UPC))

    for i in 1:size(loop_over)[1]

        i = loop_over[i,:]
        if MP.fine_mesh.use_GPU
            CUDA.@allowscalar begin
                output[[(:) for j in 1:dims]...,i...,:] = global_modes[i...,[(:) for j in 1:dims]...,:]
            end
        else
            output[[(:) for j in 1:dims]...,i...,:] = global_modes[i...,[(:) for j in 1:dims]...,:]
        end
    end

    to_reconstruct = reshape(output,(I..., prod(J)*UPC,number_of_modes))

    return reshape(reconstruct_signal(to_reconstruct,J),(([I...] .* [J...])...,UPC,number_of_modes))
end

function compute_overlap_matrix(modes)
    dims = length(size(modes)) -2
    overlap = zeros(size(modes)[end],size(modes)[end])
    for i in 1:size(modes)[end]
        input_1 = modes[[(:) for k in 1:dims+1]...,i:i]
        #input_1 = reshape(input_1,(size(input_1)...,1))
        for j in 1:size(modes)[end]
            input_2 = modes[[(:) for k in 1:dims+1]...,j:j]
            #input_2 = reshape(input_2,(size(input_2)...,1))
            overlap[i,j] = sum(input_1 .* input_2, dims = collect(1:dims+1))[1]
        end
    end
    return overlap
end






struct projection_operators_struct
    Phi_T
    Phi
end





#POD_modes

function gen_projection_operators(POD_modes,MP)

    dims = MP.fine_mesh.dims
    J = MP.J
    I = MP.I
    r = size(POD_modes)[end]



    weights = POD_modes[[(1:J[i]) for i in 1:dims]...,:,:]

    #@assert dims <= 1 "Uniform Phi is not supported for dims > 1 at this time, set uniform = false"

    for i in 1:dims
        weights = reverse(weights,dims = i)
    end
    if MP.fine_mesh.use_GPU
        Phi_T = Conv(J, size(weights)[dims+1]=>size(weights)[dims+2],stride = J,pad = 0,bias =false) |> gpu  # First convolution, operating upon a 28x28 image
        Phi = ConvTranspose(J, size(weights)[dims+2]=>size(weights)[dims+1],stride = J,pad = 0,bias =false) |> gpu # First c

    else
        Phi_T = Conv(J, size(weights)[dims+1]=>size(weights)[dims+2],stride = J,pad = 0,bias =false)   # First convolution, operating upon a 28x28 image
        Phi = ConvTranspose(J, size(weights)[dims+2]=>size(weights)[dims+1],stride = J,pad = 0,bias =false)  # First

    end

    Phi_T.weight .= weights
    Phi.weight .= weights




    return projection_operators_struct(Phi_T,Phi)
end
