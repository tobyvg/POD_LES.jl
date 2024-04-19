export gen_skew_NN, save_skew_model, load_skew_model, gen_channel_mask, padding

using LinearAlgebra


using Distributions
using JLD

using Flux
using Random
using LaTeXStrings
using ProgressBars
using Zygote
using CUDA


using NNlib


stop_gradient(f) = f()
Zygote.@nograd stop_gradient

function cons_mom_B(B_kernel;channels = 1)
    if B_kernel != 0
        for channel in channels
            dims = length(size(B_kernel))-2
            channel_mask = stop_gradient() do 
                cu(gen_channel_mask(B_kernel,channel))
            end
            means = mean(B_kernel,dims = collect(1:dims))

            B_kernel = B_kernel .- means .* channel_mask

        end
        return B_kernel
    else
        return 0
    end
end

function transpose_B(B_kernel)
    if B_kernel != 0
        dims = length(size(B_kernel))-2
        original_dims = stop_gradient() do
           collect(1:dims+2)
        end
        permuted_dims = stop_gradient() do
           copy(original_dims)
        end

        stop_gradient() do
            permuted_dims[dims+1] = original_dims[dims+2]
            permuted_dims[dims+2] = original_dims[dims+1]
        end

        T_B_kernel = permutedims(B_kernel,permuted_dims)

        for i in 1:dims
           T_B_kernel = reverse(T_B_kernel,dims = i)

        end

        return T_B_kernel
    else
        return 0
    end
end


struct skew_model_struct
    eval
    CNN
    r
    B
    B_mats
    UPC
    pad_size
    boundary_padding
    constrain_energy
    conserve_momentum
    dissipation
    kernel_sizes
    channels
    strides
end




function find_padding_size(CNN,test_size = 100)
    dims = length(size(CNN[1].weight)) - 2
    input_channels = size(CNN[1].weight)[dims + 1]
    test_input = cu(zeros(Tuple([[test_size for i in 1:dims]...,input_channels,1])))
    test_output = CNN(test_input)
    required_padding = ([size(test_input)...] .- [size(test_output)...])[1:dims]
    return Tuple(Int.(required_padding ./ 2))
end

function conv_NN(widths,channels,strides = 0,bias = true)
    if strides == 0
        strides = ones(Int,size(widths)[1])
    end
    pad = Tuple(zeros(Int,length(widths[1])))
    storage = []
    for i in 1:size(widths)[1]
        kernel_size = Tuple(2* [widths[i]...] .+ 1)
        if i == size(widths)[1]
            storage = [storage;Conv(kernel_size, channels[i]=>channels[i+1],stride = strides[i],pad = pad,bias = bias)]
        else

            storage = [storage;Conv(kernel_size, channels[i]=>channels[i+1],stride = strides[i],pad = pad,relu,bias = bias)]
        end
    end
    return Chain((i for i in storage)...)
end





function gen_channel_mask(data,channel)
    dims = length(size(data)) - 2
    number_of_channels = size(data)[end-1]
    channel_mask = stop_gradient() do
        zeros(size(data)[1:end-1])
    end
    stop_gradient() do
        channel_mask[[(:) for i in 1:dims]...,channel] .+= 1
    end
    return channel_mask
end

function construct_corner_mask(N,pad_size)
    dims = length(N)
    corner_mask = zeros(N)
    for i in 1:dims
        original_dims = collect(1:length(size(corner_mask)))
        new_dims = copy(original_dims)
        new_dims[1] = original_dims[i]
        new_dims[i] =   1

        corner_mask = permutedims(corner_mask,new_dims)
        pad_start = corner_mask[(end-pad_size[i]+1):end,[(:) for j in 1:(dims-1)]...]
        pad_end = corner_mask[1:pad_size[i],[(:) for j in 1:(dims-1)]...]




        if i == dims
            pad_start = permutedims(pad_start,new_dims)
            pad_end = permutedims(pad_end,new_dims)

            for j in 1:dims-1
                select_start = [(:) for k in 1:(j-1)]
                select_end = [(:) for k in j:(dims-1)]
                pad_start[select_start...,1:pad_size[j],select_end...] .+= 1
                pad_start[select_start...,end-pad_size[j]+1:end,select_end...] .+= 1
                pad_end[select_start...,1:pad_size[j],select_end...] .+= 1
                pad_end[select_start...,end-pad_size[j]+1:end,select_end...] .+= 1

            end

            pad_start = permutedims(pad_start,new_dims)
            pad_end = permutedims(pad_end,new_dims)
            corner_mask = cat([pad_start,corner_mask,pad_end]...,dims = 1)
        else

            corner_mask = cat([pad_start,corner_mask,pad_end]...,dims = 1)

        end
        corner_mask = permutedims(corner_mask,new_dims)

    end
    return corner_mask
end





# figure out how to deal with BCs
# Possibly implement energy conserving auto-encoder
# finish boundary condition indicator (maybe filtered level, maybe ROM level, probably start with filtered)



function padding(data,pad_size;circular = false,UPC = 0,BCs = 0,zero_corners = true,navier_stokes = false)


    dims = length(size(data)) - 2
    if navier_stokes == false
        UPC = 0
    elseif UPC == 0
        UPC = dims
    end


    if navier_stokes
        if length(size(BCs)) == 0
            BCs = stop_gradient() do
                BCs = BCs*ones(2,dims,UPC)
            end
        end
    else
        if length(size(BCs)) == 0
            BCs = stop_gradient() do
                BCs = BCs*ones(2,dims)
            end
        end
    end

    if circular
        padded_data = NNlib.pad_circular(data,Tuple(cat([[i for j in 1:2] for i in pad_size]...,dims = 1)))
    else
        N = size(data)[1:dims]
        if navier_stokes
            split_data = [data[[(:) for i in 1:dims]...,j:j,:] for j in 1:UPC]
            unknown_index = 0
            padded_data = []
            for data in split_data
                unknown_index += 1
                for i in 1:dims
                    original_dims = stop_gradient() do
                        collect(1:length(size(data)))
                    end
                    new_dims = stop_gradient() do
                        copy(original_dims)
                    end
                    stop_gradient() do
                            new_dims[1] = original_dims[i]
                            new_dims[i] = 1
                    end

                    data = permutedims(data,new_dims)

                    pad_start = data[(end-pad_size[i]+1):end,[(:) for j in 1:(dims+1)]...]
                    pad_end = data[1:pad_size[i],[(:) for j in 1:(dims+1)]...]


                    #@assert one_hot[i][1] != 0 && one_hot[i][2] != 0 "A one-hot encoding of 0 is saved for the corners outside the domain. Use a different number."
                    if BCs[1,i,unknown_index] != "c" && BCs[1,i,unknown_index] != "m"
                        pad_start_cache = 2* BCs[1,i,unknown_index] .- reverse(pad_end,dims = 1)

                    elseif BCs[1,i,unknown_index] == "m"
                        pad_start_cache = reverse(pad_end,dims = 1)
                    else
                        pad_start_cache = pad_start
                    end

                    if BCs[2,i,unknown_index] != "c" && BCs[2,i,unknown_index] != "m"
                        pad_end_cache = 2* BCs[2,i,unknown_index] .- reverse(pad_start,dims = 1)
                    elseif BCs[2,i,unknown_index] == "m"
                        pad_end_cache = reverse(pad_start,dims = 1)
                    else
                        pad_end_cache = pad_end
                    end


                    pad_start = pad_start_cache
                    pad_end = pad_end_cache


                    data = cat([pad_start,data,pad_end]...,dims = 1)
                    data = permutedims(data,new_dims)

                end
                push!(padded_data,data)
            end

            padded_data = cat(padded_data...,dims = dims + 1)

            if size(data)[dims+1] > UPC
                data = data[[(:) for i in 1:dims]...,UPC+1:end,:]
                for i in 1:dims
                    original_dims = stop_gradient() do
                        collect(1:length(size(data)))
                    end
                    new_dims = stop_gradient() do
                        copy(original_dims)
                    end
                    stop_gradient() do
                            new_dims[1] = original_dims[i]
                            new_dims[i] = 1
                    end

                    data = permutedims(data,new_dims)

                    pad_start = data[(end-pad_size[i]+1):end,[(:) for j in 1:(dims+1)]...]
                    pad_end = data[1:pad_size[i],[(:) for j in 1:(dims+1)]...]

                    #@assert one_hot[i][1] != 0 && one_hot[i][2] != 0 "A one-hot encoding of 0 is saved for the corners outside the domain. Use a different number."
                    BC_right = BCs[1,i,:]
                    BC_left = BCs[2,i,:]


                    if BC_right[1] != "c"
                        for j in BC_right
                            @assert j != "c" "Mixing periodic BCs with other BC types along dimension " * string(i) * " is not supported"
                        end
                        for j in BC_left
                            @assert j != "c" "Mixing periodic BCs with other BC types along dimension " * string(i) * " is not supported"
                        end
                        BC_right = 2*(BC_right .== "m") .- 1
                        BC_left = 2*(BC_left .== "m") .- 1

                        pad_start_cache = reverse(pad_end,dims = 1)
                        pad_end_cache = reverse(pad_start,dims = 1)
                        #if mirror_mask != 0
                        #    pad_start_cache = multiply_by_mirror_mask(pad_start_cache,mirror_mask[i][BC_right],UPC)
                        #    pad_end_cache = multiply_by_mirror_mask(pad_end_cache,mirror_mask[i][BC_left],UPC)
                        #end
                        pad_start = pad_start_cache
                        pad_end = pad_end_cache
                    else
                        for j in BC_right
                            @assert j == "c" "Mixing periodic BCs with other BC types along dimension " * string(i) * " is not supported"
                        end
                        for j in BC_left
                            @assert j == "c" "Mixing periodic BCs with other BC types along dimension " * string(i) * " is not supported"
                        end


                    end

                    pad_start



                    data = cat([pad_start,data,pad_end]...,dims = 1)
                    data = permutedims(data,new_dims)

                end
                padded_data = cat(padded_data,data,dims = dims + 1)
            end

        else
            for i in 1:dims

                original_dims = stop_gradient() do
                    collect(1:length(size(data)))
                end
                new_dims = stop_gradient() do
                    copy(original_dims)
                end
                stop_gradient() do
                    new_dims[1] = original_dims[i]
                    new_dims[i] = 1
                end

                data = permutedims(data,new_dims)

                pad_start = data[(end-pad_size[i]+1):end,[(:) for j in 1:(dims+1)]...]
                pad_end = data[1:pad_size[i],[(:) for j in 1:(dims+1)]...]



                #@assert one_hot[i][1] != 0 && one_hot[i][2] != 0 "A one-hot encoding of 0 is saved for the corners outside the domain. Use a different number."
                if BCs[1,i,1] != "c" && BCs[1,i,1] != "m"

                    pad_start_cache = BCs[1,i,1] .* pad_start.^0
                elseif BCs[1,i,1] == "m"
                    pad_start_cache = reverse(pad_end,dims = 1)
                else
                    pad_start_cache = pad_start
                end

                if BCs[2,i,1] != "c" && BCs[2,i,1] != "m"
                    pad_end_cache = BCs[2,i,1] .* pad_end.^0
                elseif BCs[2,i,1] == "m"
                    pad_end_cache = reverse(pad_start,dims = 1)
                else
                    pad_end_cache = pad_end
                end


                pad_start = pad_start_cache
                pad_end = pad_end_cache

                data = cat([pad_start,data,pad_end]...,dims = 1)
                data = permutedims(data,new_dims)
            end
            padded_data = data
        end
    end



    if zero_corners == true && circular == false
        corner_mask = stop_gradient() do
            construct_corner_mask(N,pad_size)
        end
        padded_data = padded_data .- corner_mask .* padded_data
    end

    return  padded_data
end






###################### Neural network code ############

# What to do with padding
# What to do with multiple unknowns, i.e. u and v field
# How to save the neural network

struct model_struct
    eval
    CNN
    r
    UPC
    pad_size
    boundary_padding
    constrain_energy
    conserve_momentum
    dissipation
    kernel_sizes
    channels
    strides
end




function gen_skew_NN(kernel_sizes,channels,strides,r,B;UPC = 0,boundary_padding = 0,constrain_energy = true,conserve_momentum=true,dissipation = true)
    if boundary_padding != 0 && boundary_padding != "c"
        add_input_channel = zeros(Int,size(channels)[1]+1)
        add_input_channel[1] += 1
    else
        add_input_channel = 0
    end

    if dissipation && constrain_energy
       channels = [channels ; 2*r]
    else
       channels = [channels ; r]
    end
    CNN = conv_NN(kernel_sizes,channels .+ add_input_channel,strides) |> gpu
    pad_size = find_padding_size(CNN)

    if UPC == 0
        UPC = length(size(model.CNN[1].weight))-2
    end
    dims = length(size(CNN[1].weight))-2


    B1,B2,B3 = 0,0,0
    if constrain_energy
        B1 = cu(Float32.(Flux.glorot_uniform(Tuple(2*[B...] .+1)...,r,r)))
        B2 = cu(Float32.(Flux.glorot_uniform(Tuple(2*[B...] .+1)...,r,r)))
        if dissipation
            B3 = cu(Float32.(Flux.glorot_uniform(Tuple(2*[B...] .+1)...,r,r)))
        end
    end


    if constrain_energy
        pad_size = [pad_size...]
        pad_size .+= [B...]
        pad_size = Tuple(pad_size)
    end


    B_mats = [B1,B2,B3]


    function NN(input;a = 0,CNN = CNN,r = r,B = [B...],B_mats = B_mats,UPC = UPC,pad_size = pad_size,boundary_padding = boundary_padding,constrain_energy =constrain_energy,conserve_momentum = conserve_momentum,dissipation = dissipation)

        dims = length(size(input)) - 2
        #CNN[1].weight .*= 4

        #a = input[[(:) for i in 1:dims]...,1:r,:]
        #if constrain_energy
        #    input = input[[(2*B[i]+1:end - 2*B[i]) for i in dims]...,:,:]
        #end
        ### deal with BCs in the CNN #######
        ####################################

        if boundary_padding == 0 || boundary_padding == "c"
            output = CNN(padding(input,pad_size,circular = true))

        else
            pad_input = padding(input,pad_size,BCs = boundary_padding)
            boundary_indicator_channel = stop_gradient() do
                ones(size(input)[1:end-2]...,1,size(input)[end])
            end
            boundary_indicator_padding = stop_gradient() do
                copy(boundary_padding)
            end
            stop_gradient() do
                for i in 1:prod(size(boundary_indicator_padding))
                    if boundary_indicator_padding[i] != "c"
                        boundary_indicator_padding[i] = i + 1
                    end
                end
            end
            pad_boundary_indicator_channel = padding(boundary_indicator_channel,pad_size,BCs = boundary_indicator_padding)
            output = CNN(cat([pad_input,pad_boundary_indicator_channel]...,dims = dims + 1))
        end

        #############################
        ##############################


        phi = output[[(:) for i in 1:dims]...,1:r,:]


        if constrain_energy
            psi = 0
            if dissipation
                psi = output[[(:) for i in 1:dims]...,r+1:2*r,:]
                #dTd = sum( d.^2 ,dims = [i for i in 1:dims])
            end


            B1,B2,B3 = B_mats

  
            if conserve_momentum
                B1 = cons_mom_B(B1,channels = collect(1:UPC))
                B2 = cons_mom_B(B2,channels = collect(1:UPC))
                B3 = cons_mom_B(B3,channels = collect(1:UPC))
            end


            B1_T = transpose_B(B1)
            B2_T = transpose_B(B2)
            B3_T = transpose_B(B3)



     # skew_symmetric_form
   

            c_tilde = NNlib.conv(NNlib.conv(a,B1) .* phi,B2_T) - NNlib.conv(NNlib.conv(a,B2) .* phi,B1_T)

            if dissipation
                c_tilde -=  NNlib.conv(psi.^2 .* NNlib.conv(a,B3),B3_T)
            end


            return c_tilde

        else
            return phi

        end
        
    end



    return skew_model_struct(NN,CNN,r,B,B_mats,UPC,pad_size,boundary_padding,constrain_energy,conserve_momentum,dissipation,kernel_sizes,channels,strides)
end




function save_skew_model(model,name)
    if name[end] == "/"
        name = name[1:end-1]
    end
    mkpath(name)
    save(name * "/model_state.jld","CNN_weights_and_biases",[(i.weight,i.bias) for i in model.CNN],"r",model.r,"B",model.B,"B_mats",model.B_mats,"UPC",model.UPC,"pad_size",model.pad_size,"boundary_padding",model.boundary_padding,"constrain_energy",model.constrain_energy,"conserve_momentum",model.conserve_momentum,"dissipation",model.dissipation,"kernel_sizes",model.kernel_sizes,"channels",model.channels,"strides",model.strides)
    print("\nModel saved at directory [" * name * "]\n")
end

function load_skew_model(name)
    if name[end] == "/"
        name = name[1:end-1]
    end
    CNN_weights_and_biases,r,B,B_mats,UPC,pad_size,boundary_padding,constrain_energy,conserve_momentum,dissipation,kernel_sizes,channels,strides = (load(name * "/model_state.jld")[i] for i in ("CNN_weights_and_biases","r","B","B_mats","UPC","pad_size","boundary_padding","constrain_energy","conserve_momentum","dissipation","kernel_sizes","channels","strides"))

    model = gen_skew_NN(kernel_sizes,channels,strides,r,B,boundary_padding = boundary_padding,UPC = coarse_mesh.UPC,constrain_energy = constrain_energy,dissipation = dissipation,conserve_momentum = conserve_momentum)

    for i in 1:length(model.CNN)
        model.CNN[i].weight .= CNN_weights_and_biases[i][1]
        model.CNN[i].bias .= CNN_weights_and_biases[i][2]
    end
    for i in 1:length(model.B_mats)
        model.B_mats[i] = B_mats[i]
    end

    print("\nModel loaded from directory [" * name * "]\n")
    return model
end
