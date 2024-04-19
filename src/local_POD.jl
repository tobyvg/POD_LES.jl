using LinearAlgebra
using FFTW


using Distributions
using JLD

using Flux
using Random

using ProgressBars
using Zygote
using CUDA

stop_gradient(f) = f()
Zygote.@nograd stop_gradient

export reshape_for_local_SVD, carry_out_local_SVD, local_to_global_modes, compute_overlap_matrix, add_filter_to_modes, gen_projection_operators

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


    some_ones = cu(ones(size(modes)[1:end]...,prod(I)))

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

        CUDA.@allowscalar begin
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

function add_filter_to_modes(POD_modes,MP;orthogonalize = false)

    dims = MP.fine_mesh.dims
    UPC = MP.fine_mesh.UPC
    sqrt_omega_tilde = sqrt.(MP.omega_tilde)[[(:) for i in 1:dims]...,1:1,1:1]
    some_zeros = zeros(size(sqrt_omega_tilde))

    modes = cat(sqrt_omega_tilde,[some_zeros for i in 1:UPC-1]...,dims = dims+1)
    modes = cat([circshift(modes,([0 for i in 1:dims]...,j)) for j in 0:(UPC-1)]...,dims = dims + 2)
    if POD_modes != 0

        modes = cat([modes,POD_modes]...,dims = dims + 2)
    end

    r = size(modes)[dims + 2]
    IP = 0


    for i in 2:r

        s_i = [[(:) for k in 1:dims+1]...,i:i]

        mode_i = modes[s_i...]
        if orthogonalize ### orthogonalize basis using gramm-schmidt
            for j in 1:(i-1)
                s_j = [[(:) for k in 1:dims+1]...,j:j]
                mode_j = modes[s_j...]
                IP = sum(MP.one_reconstructor(1/(prod(MP.fine_mesh.N))*MP.one_filter(mode_j .* mode_i)),dims = collect(1:dims+1))
   
                modes[s_i...] .-= (IP) .* mode_j
            end
            mode_i = modes[s_i...]

        end
        norm_i =  sum(MP.one_reconstructor(1/(prod(MP.fine_mesh.N))*MP.one_filter(mode_i .* mode_i)),dims = collect(1:dims+1))

        modes[s_i...] ./= sqrt.(norm_i)

    end

    return modes
end




struct projection_operators_struct
    Phi_T
    Phi
    W
    R
    r
end





#POD_modes

function gen_projection_operators(POD_modes,MP;uniform = false)
    
    dims = MP.fine_mesh.dims
    J = MP.J
    I = MP.I
    r = size(POD_modes)[end]


    sqrt_omega_tilde = sqrt.(MP.omega_tilde)
    inv_sqrt_omega_tilde = 1 ./ sqrt_omega_tilde
    

    if uniform == false

        Phi_T(input,modes = POD_modes,MP = MP) = cat([sum(MP.one_filter(input .* modes[[(:) for i in 1:MP.fine_mesh.dims+1]...,j]),dims = [MP.fine_mesh.dims+1]) for j in 1:size(modes)[end]]...,dims = MP.fine_mesh.dims + 1)

        function Phi(input,modes = POD_modes,MP = MP)
            UPC = MP.fine_mesh.UPC
            dims = MP.fine_mesh.dims
            r = size(modes)[end]
            Phi_mask = ones((size(input)[1:end-2]...,UPC,size(input)[end]))
            result = stop_gradient() do
                zeros((size(modes)[1:dims]...),UPC,size(input)[end])
            end
            for j in 1:r
                result += modes[[(:) for i in 1:dims+1]...,j:j] .* MP.one_reconstructor(input[[(:) for i in 1:dims]...,j:j,:] .* Phi_mask)
            end
            return result
        end
    else
        
        weights = POD_modes[[(1:J[i]) for i in 1:dims]...,:,:]

        #@assert dims <= 1 "Uniform Phi is not supported for dims > 1 at this time, set uniform = false"

        for i in 1:dims
            weights = reverse(weights,dims = i)
        end

        Phi_T = Conv(J, size(weights)[dims+1]=>size(weights)[dims+2],stride = J,pad = 0,bias =false) |> gpu  # First convolution, operating upon a 28x28 image
        Phi = ConvTranspose(J, size(weights)[dims+2]=>size(weights)[dims+1],stride = J,pad = 0,bias =false) |> gpu # First c

        
        Phi_T.weight .= weights
        Phi.weight .= weights

    end





    W(input,Phi_T = Phi_T, sqrt_omega_tilde = sqrt_omega_tilde) =  Phi_T(input .* sqrt_omega_tilde)
    R(input,Phi = Phi,inv_sqrt_omega_tilde =inv_sqrt_omega_tilde) =  inv_sqrt_omega_tilde .*  Phi(input)


    return projection_operators_struct(Phi_T,Phi,W,R,r)
end






