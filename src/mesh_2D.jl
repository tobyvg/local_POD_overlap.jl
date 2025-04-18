

using LinearAlgebra
using Random

using Distributions
using CUDA
using Flux
using Zygote



export generate_spectrum,construct_k,gen_random_field,gen_mesh,gen_mesh_pair,gen_coarse_from_fine_mesh, project_on_basis


struct mesh_struct
    dims # 1D/2D
    N # grid resolution
    x # coordinates
    dx #dx in each dimension
    x_edges # edges
    omega # mass matrix
    eval_function # evaluate function on the grid
    ip # computes inner-product
    integ # integral on the grid
    UPC # unknows per grid cell
    use_GPU # Whether to use GPU for computations
end

struct mesh_pair_struct
    fine_mesh
    coarse_mesh
    J
    I
    one_filter
    one_reconstructor
    omega_tilde
end


function construct_k(N)
    dims = length(N)
    k = [fftfreq(i,i) for i in N]

    some_ones = ones(N)
    k_mats = some_ones .* k[1]

    k_mats = reshape(k_mats,(size(k_mats)...,1))

    for i in 2:dims
        original_dims = collect(1:dims)
        permuted_dims = copy(original_dims)
        permuted_dims[1] = i
        permuted_dims[i] = 1

        k_mat = permutedims(k[i] .* permutedims(some_ones,permuted_dims),permuted_dims)
        k_mats = cat(k_mats,k_mat,dims = dims + 1)
    end
    return k_mats
end


function gen_permutations(N)

    N_grid = [collect(1:n) for n in N]

    sub_grid = ones(Int,(N...))

    dims = length(N)
    sub_grids = []

    for i in 1:dims
        original_dims = collect(1:dims)
        permuted_dims = copy(original_dims)
        permuted_dims[1] = original_dims[i]
        permuted_dims[i] = 1


        push!(sub_grids,permutedims(N_grid[i] .*  permutedims(sub_grid,permuted_dims),permuted_dims))

    end

    return reshape(cat(sub_grids...,dims = dims + 1),(prod(N)...,dims))
end


function project_on_basis(data,basis ; positivity_constraint = false)

    normal_basis = basis ./ (sqrt.(sum(basis .^2,dims = [length(size(basis))-1])))
    weights = sum(data .* normal_basis ,dims = [length(size(basis))-1])
    if positivity_constraint
        weights = max.(weights,0)
    end
    projected_data = weights .* normal_basis

    return projected_data

end
#function generate_spectrum(samples,mesh; a = 1.6)
#    k = construct_k(mesh.N)
#    k2 = sqrt.(sum((k).^2,dims = 3))
#
#    V_hat = fft(samples,[1,2]) ./ prod(mesh.N)
#    e_hat = mean(1/2*sum(V_hat .* conj.(V_hat),dims = [3]),dims =[4])
#
#    k2_flat = reshape(k2,(prod(size(k2)[1:end-1])...,size(k2)[end]))
#    e_hat_flat = reshape(e_hat,(prod(size(e_hat)[1:end-1])...,size(e_hat)[end]))

#    bin_lims = extrema(k2_flat)
#    bins = collect(1:floor(bin_lims[2]/a))

    #Generate dyadic bins
#    lower_bins = bins / a
#    upper_bins = bins * a

#    energies = zeros(size(bins)[1])
#    for i in 1:size(e_hat_flat)[1]
#        for j in 1:size(lower_bins)[1]
#            if k2_flat[i,1] > lower_bins[j] && k2_flat[i,1] <= upper_bins[j]
#                energies[j] += mean(e_hat_flat[i,:])
#            end
#        end
#    end
#    return bins,energies
#end

function generate_spectrum(samples; a = 1.6, return_CI = false)
    N = size(samples)[1:end-2]
    k = construct_k(N)
    k2 = sqrt.(sum((k).^2,dims = 3))

    V_hat = fft(samples,[1,2]) ./ prod(N)
    k2_flat = reshape(k2,(prod(size(k2)[1:end-1])...,size(k2)[end]))

    bin_lims = extrema(k2_flat)
    bins = collect(1:floor(bin_lims[2]/a))

    #Generate dyadic bins
    lower_bins = bins / a
    upper_bins = bins * a
    energies = zeros(size(bins)[1])

    if return_CI

        e_hat = 1/2*sum(V_hat .* conj.(V_hat),dims = [length(size(samples))-1])
        e_hat_flat = reshape(e_hat,(prod(size(e_hat)[1:end-1])...,size(e_hat)[end]))


        for i in 1:size(lower_bins)[1]
            collect_energies = []
            for j in 1:size(e_hat_flat)[1]
                if k2_flat[j,1] > lower_bins[i] && k2_flat[j,1] <= upper_bins[i]
                    #collect_energies = cat(collect_energies,e_hat_flat[j,:],dims = 1)
                    push!(collect_energies,e_hat_flat[j,:])
                end
            end
            collect_energies = cat(collect_energies...,dims = 2)
            collect_energies = sum(collect_energies,dims = [2])[1:end]
            CI = std(collect_energies) /sqrt(size(collect_energies)[1])
            energies[i] += CI
        end


    else

        e_hat = mean(1/2*sum(V_hat .* conj.(V_hat),dims = [length(size(samples))-1]),dims =[length(size(samples))])
        e_hat_flat = reshape(e_hat,(prod(size(e_hat)[1:end-1])...,size(e_hat)[end]))


        for i in 1:size(lower_bins)[1]
            for j in 1:size(e_hat_flat)[1]

                if k2_flat[j,1] > lower_bins[i] && k2_flat[j,1] <= upper_bins[i]
                    energies[i] += e_hat_flat[j,1]
                end
            end
        end

    end

    return bins,energies
end

function construct_spectral_filter(k_mats,max_k)
    filter = ones(size(k_mats)[1:end-1])
    N = size(k_mats)[1:end-1]
    dims = length(N)
    loop_over = gen_permutations(N)
    for i in 1:size(loop_over)[1]
        i = loop_over[i,:]
        k = k_mats[i...,:]
        if sqrt(sum(k.^2)) >= max_k
            filter[i...] = 0
        end
    end
    return filter
end

function gen_random_field(N,max_k;norm = 1,samples = (1,1))
    dims = length(N)
    k = construct_k(N)
    filter = construct_spectral_filter(k,max_k)
    coefs = (rand(Uniform(-1,1),(N...,samples...)) + rand(Uniform(-1,1),(N...,samples...)) * (0+1im))

    result = real.(ifft(filter .* coefs,collect(1:dims)))
    result .-= mean(result,dims = collect(1:dims))
    sqrt_energies = sqrt.(1/prod(N) .* sum(result.^2,dims = collect(1:dims)))
    result ./= sqrt_energies
    result .*= norm


    return result
end





function gen_mesh(x,y = nothing, z = nothing;UPC=1,use_GPU = false)
    if y != nothing
        if z != nothing
            x = [x,y,z]
        else
            x = [x,y]
        end
    else
        if length(size(x[1])) <= 0
            x = [x]
        end
    end
    T = typeof(x)
    #print(x[1])
    mid_x = [ [(i[j] + i[j+1])/2 for j in 1:(size(i)[1]-1)] for i in x]

    dx = [ [(i[j+1] - i[j]) for j in 1:(size(i)[1]-1)] for i in x]


    sub_grid = ones([size(i)[1] for i in mid_x]...)
    sub_dx = ones([size(i)[1] for i in mid_x]...)
    omega = ones([size(i)[1] for i in mid_x]...)


    sub_grids = []
    sub_dxs = []


    dims = size(x)[1]


    for i in 1:dims
        original_dims = collect(1:dims)
        permuted_dims = copy(original_dims)
        permuted_dims[1] = original_dims[i]
        permuted_dims[i] = 1


        omega = permutedims(dx[i] .*  permutedims(omega,permuted_dims),permuted_dims)

        push!(sub_grids,permutedims(mid_x[i] .*  permutedims(sub_grid,permuted_dims),permuted_dims))
        push!(sub_dxs,permutedims(dx[i] .*  permutedims(sub_dx,permuted_dims),permuted_dims))
    end


    x_edges = x
    x = cat(sub_grids...,dims = dims + 1)
    dx = cat(sub_dxs...,dims = dims + 1)
    omega = cat([omega for i in 1:UPC]...,dims = dims+1)

    x = reshape(x,(size(x)...,1))
    dx = reshape(dx,(size(dx)...,1))
    omega = reshape(omega,(size(omega)...,1))

    if use_GPU
        x = cu(x)
        dx = cu(dx)
        omega = cu(omega)
    end

    function eval_function(F,x = x,dims = dims)
        return F([x[[(:) for j in 1:dims]...,i:i,:] for i in 1:dims])

    end

    function ip(a,b;weighted = true,omega = omega,dims = dims,combine_channels = true)
        if weighted
            IP = a .* omega[[(:) for i in 1:dims]...,1,1] .* b
        else
            IP = a .* b
        end
        if combine_channels == true
            IP =  sum(IP,dims = collect(1:(dims+1)))
        else
            IP =  sum(IP,dims = collect(1:(dims)))
        end
        return IP
    end

    function integ(a;weighted = true,omega = omega,dims = dims,ip = ip,use_GPU = use_GPU)
        #channel_a = a[[(:) for i in 1:dims]...,channel:channel,:]
        if use_GPU
            some_ones = stop_gradient() do
                cu(ones(size(a)))
            end
        else
            some_ones = stop_gradient() do
                ones(size(a))
            end
        end
        return ip(some_ones,a,weighted=weighted,omega=omega,dims=dims,combine_channels = false)
    end



    return mesh_struct(dims,size(omega)[1:dims],x,dx,x_edges,omega,eval_function,ip,integ,UPC,use_GPU)
end


# connect to NS code
function gen_one_filter(J,UPC)

    #Jx = Int(grid.nx/grid_bar.nx)
    #Jy = Int(grid.ny/grid_bar.ny)
    dims = length(J)
    #J = (Jy,Jx)
    filter = Conv(J, UPC=>UPC,stride = J,pad = 0,bias =false)  # First convolution, operating upon a 28x28 image
    for i in 1:UPC

        for j in 1:UPC
            if i == j
                filter.weight[[(:) for k in 1:dims]...,i,j] .= 1.
            else
                filter.weight[[(:) for k in 1:dims]...,i,j] .= 0.
            end
        end
    end
    return filter
end



function gen_one_reconstructor(J,UPC)

    #Jx = Int(grid.nx/grid_bar.nx)
    #Jy = Int(grid.ny/grid_bar.ny)
    dims = length(J)
    #J = (Jy,Jx)
    reconstructor = ConvTranspose(J, UPC=>UPC,stride = J,pad = 0,bias =false)  # First convolution, operating upon a 28x28 image

    for i in 1:UPC

        for j in 1:UPC
            if i == j
                reconstructor.weight[[(:) for k in 1:dims]...,i,j] .= 1.
            else
                reconstructor.weight[[(:) for k in 1:dims]...,i,j] .= 0.
            end
        end
    end
    return reconstructor
end


function reconstruct_signal(R_q,J)

    ndims(R_q) > 2 || throw(ArgumentError("expected x with at least 3 dimensions"))
    d = ndims(R_q) - 2
    sizein = size(R_q)[1:d]
    cin, n = size(R_q, d+1), size(R_q, d+2)
    #cin % r^d == 0 || throw(ArgumentError("expected channel dimension to be divisible by r^d = $(
    #    r^d), where d=$d is the number of spatial dimensions. Given r=$r, input size(x) = $(size(x))"))

    cout = cin ÷ prod(J)
    R_q = reshape(R_q, sizein..., J..., cout, n)
    perm = hcat(d+1:2d, 1:d) |> transpose |> vec  # = [d+1, 1, d+2, 2, ..., 2d, d]
    R_q = permutedims(R_q, (perm..., 2d+1, 2d+2))
    R_q = reshape(R_q, J.*sizein..., cout, n)

    return R_q
end




function gen_mesh_pair(fine_mesh,coarse_mesh)
    divide = [fine_mesh.N...] .% [coarse_mesh.N...]
    for i in divide
        @assert i == 0 "Meshes are not compatible. Make sure the dimensions of the fine mesh are
                divisible by the dimensions of the coarse mesh."
    end
    UPC = fine_mesh.UPC
    dims = fine_mesh.dims
    J =Tuple([Int(fine_mesh.N[i]/coarse_mesh.N[i]) for i in 1:dims])
    I = coarse_mesh.N
    use_GPU = fine_mesh.use_GPU




    if use_GPU
        one_filter = gen_one_filter(J,UPC) |> gpu

        one_reconstructor = gen_one_reconstructor(J,UPC) |> gpu



    else
        one_filter = gen_one_filter(J,UPC)

        one_reconstructor = gen_one_reconstructor(J,UPC)


    end



    omega_tilde = fine_mesh.omega

    #print(typeof(fine_mesh.omega))
    #print(typeof(one_reconstructor(coarse_mesh.omega)))
    omega_tilde = fine_mesh.omega ./ one_reconstructor(coarse_mesh.omega)


    return mesh_pair_struct(fine_mesh,coarse_mesh,J,I,one_filter,one_reconstructor,omega_tilde)
end


function gen_coarse_from_fine_mesh(fine_mesh,J)

    divide = [fine_mesh.N...] .% [J...]
    for i in divide
        @assert i == 0 "Meshes are not compatible. Make sure the dimensions of the fine mesh are
                divisible reduction parameter J in each dimension."
    end


    dims = fine_mesh.dims
    N = fine_mesh.N
    x = fine_mesh.x_edges

    I =Tuple([Int(fine_mesh.N[i]/J[i]) for i in 1:dims])

    X  = []
    for i in 1:length(x)
        selector = [1,(1 .+ J[i]*collect(1:I[i]))...]
        push!(X,x[i][selector])
    end
    return gen_mesh(X,UPC = fine_mesh.UPC,use_GPU = fine_mesh.use_GPU)
end
