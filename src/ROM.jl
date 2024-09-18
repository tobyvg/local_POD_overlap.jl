using LinearAlgebra
using SparseArrays
using Statistics

export gen_ROM

struct ROM_struct
    basis
    local_basis
    S
    S_inv
    f
    P
    A
    p_A
end

"""
gen_ROM(x, snapshots, f; r=1, I=10, overlap=1, localize=true)

Generates a Reduced Order Model (ROM) based on provided snapshots and operator. This function can localize the basis functions to smaller regions of the domain and apply overlapping techniques for enhanced resolution.

# Arguments:
- `x`: Spatial grid points as a vector.
- `snapshots`: Matrix of snapshot data, where each column represents a state at a different time or parameter.
- `f`: Function used to generate the operator matrix `A` in the reduced model.
- `r`: (Optional) Number of modes to retain in the local SVD basis for each segment. Default is 1.
- `I`: (Optional) Number of subdomains or segments to divide the snapshot data into. Default is 10.
- `overlap`: (Optional) Number of overlapping rows between consecutive segments for localization. Default is 1.
- `localize`: (Optional) Boolean flag indicating whether or not to localize the basis. Default is `true`.

# Returns:
- A `ROM_struct` object containing the following fields:
  - `basis`: The full reduced basis matrix for the entire domain.
  - `local_basis`: The localized basis matrix for individual subdomains.
  - `S`: Gram matrix (inner product) of the basis functions.
  - `S_inv`: Inverse of the Gram matrix `S`.
  - `_f_`: Function to compute the reduced dynamics based on the reduced basis.
  - `P`: Projection operator to project states onto the reduced space.
  - `A`: Operator matrix in the reduced model.
  - `p_A`: Projected operator matrix (detailed in article).
"""
function gen_ROM(x,snapshots,f;r=1,I=10,overlap = 1,localize = true)
    N = size(snapshots)[1]
    J = Int(N/I)
    if localize

        m = reshape( snapshots,(J,I,size(snapshots)[2]))
        snapshots = m
        for i in 1:overlap
            snapshots = [snapshots;circshift(m,(0,-i,0))]
        end

        snapshots = reshape(snapshots,(size(snapshots)[1],size(snapshots)[2]*size(snapshots)[3]))
        if overlap == 0
            local_basis = svd(snapshots).U[:,1:r]
        else

            local_basis = svd(kernel(size(snapshots)[1]) .* snapshots).U[:,1:r]
        end


        offset = 0
        basis = zeros(N,(I)*r)
        total_modes = size(basis)[2]
        width = size(local_basis)[1]
        for i in 1:I
            basis[1:width,1:r] = local_basis
            basis = circshift(basis,(J,r))
        end
    else
        basis = svd(snapshots).U[:,1:I*r]
        local_basis = basis
        total_modes = size(basis)[2]

    end
    if localize
        basis = sparsify_matrix(basis)
    end
    S = basis' * basis
    S_inv = inv(Matrix(S))
    A = gen_operator(x,f,basis)
    _f_ = 0
    S_inv_A = S_inv*A
    if localize == false
        A = Matrix(A)
    end
    P = basis * S_inv * basis'

    p_A = basis *S_inv * A * S_inv*basis'

    function _f_(a,x,t,S=S,S_inv = S_inv,A = A,S_inv_A = S_inv_A,overlap = overlap,localize = localize)

        if overlap >= 1 && localize == true
            dadt = S_inv_A*a
        else
            dadt = A*a
        end
        return dadt
    end

    return ROM_struct(basis,local_basis, S,S_inv,_f_,P,A,p_A)
end
