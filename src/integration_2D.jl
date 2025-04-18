using LinearAlgebra
using CUDA
using JLD
using ProgressMeter


export simulate,simulate_differentiable, gen_time_interpolator


########### Time integration code ####

#function rhs(input,mesh,t;other_arguments = (model,))
    # implement boundary conditions through clever use of the "other_arguments" input
#    return other_arguments[1].eval(input)
#end

function time_step(input,mesh,t,dt,rhs;other_arguments = 0,method = "RK4")

    T = stop_gradient() do
        typeof(CUDA.@allowscalar(mesh.dx[1]))
    end
    if method == "RK4"

        k1 =  rhs(input,mesh,t,other_arguments = other_arguments)

        k2 = rhs(input + T(dt/2)*k1,mesh,t .+ dt/2,other_arguments = other_arguments)


        #k1 = rhs(input,mesh,t,other_arguments = other_arguments)

        k3 = rhs(input + T(dt/2)*k2,mesh,t .+ dt/2,other_arguments = other_arguments)
        k4 = rhs(input + T(dt)*k3,mesh,t .+ dt,other_arguments = other_arguments)



        return T(1/6*dt)*(k1 + 2*k2 + 2*k3 + k4)


    end
end


function simulate(input0,mesh,dt,t_start,t_end,rhs;time_step_function=time_step,save_every = 1,other_arguments = 0,pre_allocate = false)
    use_GPU = mesh.use_GPU

    if project_div
        setup = gen_setup(mesh)
    end

    dims = length(size(input0))-2

    steps = stop_gradient() do
        CUDA.@allowscalar(round(Int,(t_end[1] - t_start[1]) ./ dt))
    end


    if pre_allocate == false
        output = stop_gradient() do
            Array{Float32}(undef, size(input0)[1:end]..., 0)
        end

        output_t = stop_gradient() do
            Array{Float32}(undef, ones(Int,dims + 1)...,size(input0)[end], 0)
        end
    else
        output = stop_gradient() do
            zeros(size(input0)[1:end]..., floor(Int,steps/save_every))
        end

        output_t = stop_gradient() do
            zeros(ones(Int,dims + 1)...,size(input0)[end], floor(Int,steps/save_every))
        end
    end

    if length(size(t_start)) == 0
        t_start = stop_gradient() do
            [t_start]
        end
    end

    t = reshape(t_start,(size(output_t)[1:end-2]...,prod(size(t_start))))
    t = stop_gradient() do
        typeof(t)(ones(size(output_t)[1:end-2]...,size(input0)[end])).* t
    end


    input = input0
    pre_alloc_counter = 0

    if pre_allocate
        iterate = tqdm
    else
        iterate = collect
    end

    if use_GPU
        output = cu(output)
        output_t = cu(output_t)
        t = cu(t)
    end
    crashed = false
    @showprogress for i in 1:steps
        input += time_step_function(input,mesh,t,dt,rhs;other_arguments = other_arguments)



        t  = t .+ dt

        #if pre_allocate
        #    @assert isnan(CUDA.@allowscalar(input[1])) == false "NaN appeared in simulation"
        #end

        if i % save_every == 0
            pre_alloc_counter += 1
            if pre_allocate
                output[[(:) for i in size(input)]...,pre_alloc_counter:pre_alloc_counter] .+= input
                output_t[[(:) for i in size(t)]...,pre_alloc_counter:pre_alloc_counter] .+= t
            else

                output = cat([output,input]...,dims = dims + 3)
                output_t = cat([output_t,t]...,dims = dims + 3)

            end
        end


        #if i % GC_every == 0 && pre_allocate
        #    GC.gc()
        #end
    end

    return output_t,output
end

function simulate_differentiable(input0,mesh,dt,t_start,t_end,rhs;time_step_function=time_step,save_every = 1,other_arguments = 0)
    use_GPU = mesh.use_GPU



    dims = length(size(input0))-2

    steps = stop_gradient() do
        CUDA.@allowscalar(round(Int,(t_end[1] - t_start[1]) ./ dt))
    end

    output = stop_gradient() do
        Array{Float32}(undef, size(input0)[1:end]..., 0)
    end

    output_t = stop_gradient() do
        Array{Float32}(undef, ones(Int,dims + 1)...,size(input0)[end], 0)
    end


    if length(size(t_start)) == 0
        t_start = stop_gradient() do
            [t_start]
        end
    end

    t = reshape(t_start,(size(output_t)[1:end-2]...,prod(size(t_start))))
    t = stop_gradient() do
        typeof(t)(ones(size(output_t)[1:end-2]...,size(input0)[end])).* t
    end


    input = input0


    iterate = collect


    if use_GPU
        output = cu(output)
        output_t = cu(output_t)
        t = cu(t)
    end

    for i in iterate(1:steps)
        input += time_step_function(input,mesh,t,dt,rhs;other_arguments = other_arguments)



        t  = t .+ dt


        if i % save_every == 0


            output = cat([output,input]...,dims = dims + 3)
            output_t = cat([output_t,t]...,dims = dims + 3)

        end


        #if i % GC_every == 0 && pre_allocate
        #    GC.gc()
        #end
    end

    return output_t,output
end

function gen_time_interpolator(t_data,data) # only for uniform timesteps
    # supply t_data as e.g. (1,1,1,number_of_simulations,number_of_time_steps) and
    # data as (N,N,UPC,number_of_simulations,number_of_time_steps) sized array
    function interpolator_function(t;simulation_indexes = (:),data = data,t_data = t_data)
        # supply t as (1,1,1,considered_number_of_simulations,considered_points_in_time) and
        # simulation indexes as a (considered_number_of_simulations) sized array
        dims = length(size(data))-3

        data = data[[(:) for i in 1:dims+1]...,simulation_indexes,:]
        t_data = t_data[[(:) for i in 1:dims+1]...,simulation_indexes,:]

        t_start = t_data[[(:) for i in 1:dims+1]...,:,1:1]
        t_end =  t_data[[(:) for i in 1:dims+1]...,:,end:end]
        number_of_time_steps = size(t_data)[end] .- 1

        indexes = number_of_time_steps .* (((t .* typeof(t)(ones(size(t_start)))) .- t_start) ./ (t_end .- t_start)) .+ 1

        lower_index = floor.(Int,indexes)
        higher_index = ceil.(Int,indexes)

        weight = indexes - lower_index

        lower_data = cat([data[[(:) for i in 1:dims+1]...,j,lower_index[[1 for i in 1:dims+1]...,j:j,:]] for j in 1:size(data)[dims+2]]...,dims = dims +2)
        higher_data = cat([data[[(:) for i in 1:dims+1]...,j,higher_index[[1 for i in 1:dims+1]...,j:j,:]] for j in 1:size(data)[dims+2]]...,dims = dims +2)

        #lower_data = cat([t_data[[(:) for i in 1:dims+1]...,j,lower_index[[1 for i in 1:dims+1]...,j:j,:]] for j in 1:size(t_data)[dims+2]]...,dims = dims +2)
        #higher_data = cat([t_data[[(:) for i in 1:dims+1]...,j,higher_index[[1 for i in 1:dims+1]...,j:j,:]] for j in 1:size(t_data)[dims+2]]...,dims = dims +2)



        interpolated = weight .* higher_data + (1 .- weight) .* lower_data

        return interpolated
    end
end
