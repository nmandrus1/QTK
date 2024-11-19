using LinearAlgebra

"""
    tensor_product(v1::Vector, v2::Vector)

Compute the tensor product of two vectors and return a flattened result.

# Arguments
- `v1::Vector`: First vector
- `v2::Vector`: Second vector

# Returns
- `Vector`: Flattened tensor product

# Examples
```julia
julia> v1 = [1, 0]
julia> v2 = [1, 0]
julia> tensor_product(v1, v2)
4-element Vector{Int64}:
 1
 0
 0
 0
```
"""
function tensor_product(v1::Vector, v2::Vector)
    # Reshape vectors to column vectors and compute outer product
    result = reshape(v2, :, 1) * reshape(v1, 1, :)
    
    # Flatten the result using vec()
    return vec(result)
end

"""
    tensor_product_n(vectors::Vector{Vector{T}}) where T<:Number

Compute the tensor product of a list of vectors sequentially.

# Arguments
- `vectors`: List of vectors to compute tensor product of

# Returns
- `Vector`: Flattened tensor product of all input vectors

# Examples
```julia
julia> q0 = [1, 0]
julia> q1 = [0, 1]
julia> tensor_product_n([q0, q1, q0])
8-element Vector{Int64}:
 0
 0
 1
 0
 0
 0
 0
 0
```
"""
function tensor_product_n(vectors::Vector{Vector{T}}) where T<:Number
    if isempty(vectors)
        throw(ArgumentError("Input vector list cannot be empty"))
    end
    
    if length(vectors) == 1
        return vectors[1]
    end
    
    # Start with the first vector
    result = vectors[1]
    
    # Sequentially compute tensor product with remaining vectors
    for i in 2:length(vectors)
        result = tensor_product(result, vectors[i])
    end
    
    return result
end

"""
    tensor_product(M1::Matrix, M2::Matrix)

Compute the tensor product (Kronecker product) of two matrices.

# Arguments
- `M1::Matrix`: First matrix
- `M2::Matrix`: Second matrix

# Returns
- `Matrix`: Tensor product matrix

# Example
```julia
julia> A = [1 2; 3 4]
julia> B = [0 1; 1 0]
julia> tensor_product(A, B)
4×4 Matrix{Int64}:
 0  1  0  2
 1  0  2  0
 0  3  0  4
 3  0  4  0
```
"""
function tensor_product(M1::Matrix, M2::Matrix)
    m1, n1 = size(M1)
    m2, n2 = size(M2)
    
    result = zeros(eltype(M1), m1 * m2, n1 * n2)
    
    for i in 1:m1
        for j in 1:n1
            result[(i-1)*m2 + 1:i*m2, (j-1)*n2 + 1:j*n2] = M1[i,j] * M2
        end
    end
    
    return result
end

"""
    tensor_product_n(vectors::Vector{Vector{T}}) where T<:Number

Compute the tensor product of a list of vectors sequentially.

# Arguments
- `vectors`: List of vectors to compute tensor product of

# Returns
- `Vector`: Flattened tensor product of all input vectors
"""
function tensor_product_n(vectors::Vector{Vector{T}}) where T<:Number
    if isempty(vectors)
        throw(ArgumentError("Input vector list cannot be empty"))
    end
    
    if length(vectors) == 1
        return vectors[1]
    end
    
    result = vectors[1]
    for i in 2:length(vectors)
        result = tensor_product(result, vectors[i])
    end
    
    return result
end

"""
    tensor_product_n(matrices::Vector{Matrix{T}}) where T<:Number

Compute the tensor product of a list of matrices sequentially.

# Arguments
- `matrices`: List of matrices to compute tensor product of

# Returns
- `Matrix`: Tensor product of all input matrices
"""
function tensor_product_n(matrices::Vector{Matrix{T}}) where T<:Number
    if isempty(matrices)
        throw(ArgumentError("Input matrix list cannot be empty"))
    end
    
    if length(matrices) == 1
        return matrices[1]
    end
    
    result = matrices[1]
    for i in 2:length(matrices)
        result = tensor_product(result, matrices[i])
    end
    
    return result
end

# Note: For creating identity matrices, Julia provides a built-in function:
# I(N) or Matrix(I, N, N) where N is the size of the identity matrix
# Example usage: I(2) creates a 2×2 identity matrix

"""
    qu_zero()

Returns the state vector for |0⟩
"""
qu_zero() = [1.0, 0.0]

"""
    qu_one()

Returns the state vector for |1⟩
"""
qu_one() = [0.0, 1.0]

"""
    decompose_quantum_state(state::Vector{Float64})

Try to decompose a quantum state vector into individual qubit states.
Returns a vector of qubit states (each either |0⟩ or |1⟩).

Example:
For a 4-qubit system, if state[10] = 1 (and others 0), this means
the state is |1001⟩ (9 in binary), and should return [|1⟩,|0⟩,|0⟩,|1⟩]
"""
function decompose_quantum_state(state::Vector{Float64})
    n = length(state)
    num_qubits = Int(log2(n))
    
    if 2^num_qubits != n
        error("State vector length must be a power of 2")
    end
    
    # Find the index of the maximum amplitude
    max_idx = argmax(abs.(state))
    
    # Convert to zero-based index for binary representation
    idx = max_idx - 1
    
    # Convert to binary and pad with zeros
    binary = digits(idx, base=2, pad=num_qubits)
    
    # Create array of qubit states
    qubit_states = Vector{Vector{Float64}}(undef, num_qubits)
    for (i, bit) in enumerate(reverse(binary))  # reverse to match conventional ordering
        if bit == 0
            qubit_states[i] = qu_zero()
        else
            qubit_states[i] = qu_one()
        end
    end
    
    return qubit_states
end

"""
    verify_quantum_decomposition(state::Vector{Float64}, qubit_states::Vector{Vector{Float64}})

Verify if the decomposed qubit states produce the given state vector.
Returns the relative error of the reconstruction.
"""
function verify_quantum_decomposition(state::Vector{Float64}, qubit_states::Vector{Vector{Float64}})
    # Compute tensor product of all qubit states
    reconstructed = qubit_states[1]
    for i in 2:length(qubit_states)
        reconstructed = vec(reconstructed * qubit_states[i]')
    end
    
    # Compute relative error
    error = norm(state - reconstructed) / norm(state)
    
    return error
end

"""
    print_quantum_state(qubit_states::Vector{Vector{Float64}})

Pretty print the quantum state in bra-ket notation.
"""
function print_quantum_state(qubit_states::Vector{Vector{Float64}})
    state_str = "|"
    for q in qubit_states
        if q ≈ qu_zero()
            state_str *= "0"
        elseif q ≈ qu_one()
            state_str *= "1"
        else
            state_str *= "?"
        end
    end
    state_str *= "⟩"
    println(state_str)
end


"""
    analyze_quantum_state(state::Vector{Float64}; threshold::Float64=1e-10)

Analyze a quantum state vector to identify significant basis states and their probabilities.
Returns a Dict mapping basis states (in ket notation) to their probabilities.

Parameters:
- state: The quantum state vector
- threshold: Minimum amplitude to consider (for handling numerical noise)
"""
function analyze_quantum_state(state::Vector{Float64}; threshold::Float64=1e-10)
    n = length(state)
    num_qubits = Int(log2(n))
    
    if 2^num_qubits != n
        error("State vector length must be a power of 2")
    end
    
    # Dictionary to store states and their probabilities
    state_probs = Dict{String, Float64}()
    
    # Analyze each amplitude
    for (idx, amplitude) in enumerate(state)
        prob = abs2(amplitude)  # probability is squared magnitude
        
        if prob > threshold
            # Convert to binary representation (zero-based indexing)
            binary = digits(idx-1, base=2, pad=num_qubits)
            
            # Create ket notation
            ket = "|" * join(reverse(binary)) * "⟩"
            
            state_probs[ket] = prob
        end
    end
    
    # Verify probabilities sum approximately to 1
    total_prob = sum(values(state_probs))
    if !isapprox(total_prob, 1.0, atol=0.01)
        @warn "Total probability = $total_prob (should be ≈ 1.0)"
    end
    
    return state_probs
end

"""
    print_quantum_analysis(state::Vector{Float64}; threshold::Float64=1e-10)

Pretty print the analysis of a quantum state vector.
"""
function print_quantum_analysis(state::Vector{Float64}; threshold::Float64=1e-10)
    println("Quantum State Analysis:")
    println("----------------------")
    
    # Get state probabilities
    state_probs = analyze_quantum_state(state, threshold=threshold)
    
    # Sort by probability (descending)
    sorted_states = sort(collect(state_probs), by=x->x[2], rev=true)
    
    # Print each state and its probability
    for (ket, prob) in sorted_states
        println("State $ket: $(round(prob * 100, digits=4))%")
    end
    
    # Print total probability
    total_prob = sum(values(state_probs))
    println("\nTotal probability: $(round(total_prob * 100, digits=4))%")
end

"""
    identify_state_pattern(state::Vector{Float64}; threshold::Float64=1e-10)

Try to identify patterns in the quantum state (e.g., superpositions, bell states).
"""
function identify_state_pattern(state::Vector{Float64}; threshold::Float64=1e-10)
    state_probs = analyze_quantum_state(state, threshold=threshold)
    
    if length(state_probs) == 1
        # Pure basis state
        return "Pure basis state: $(first(keys(state_probs)))"
    elseif length(state_probs) == 2
        # Possible superposition of two states
        probs = collect(values(state_probs))  # Convert to array for direct access
        states = collect(keys(state_probs))   # Convert to array for direct access
        
        if isapprox(first(probs), 0.5, atol=0.01) && isapprox(last(probs), 0.5, atol=0.01)
            return "Equal superposition of states: $(join(states, " and "))"
        else
            return "Superposition of states: $(join(states, " and "))"
        end
    end
    
    return "Complex superposition of $(length(state_probs)) states"
end
