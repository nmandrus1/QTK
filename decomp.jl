include("quantum.jl")

# Example 1: State |1001⟩ (9 in decimal)
state1 = zeros(16)
state1[10] = 1.0  # 9 in zero-based indexing
qubit_states = decompose_quantum_state(state1)
print_quantum_state(qubit_states)  # Should print |1001⟩

# Example 2: Verify reconstruction
err = verify_quantum_decomposition(state1, qubit_states)
println("Reconstruction error: ", err)

# Example 3: Create a state using tensor products

tp = tensor_product_n([qu_zero(), qu_zero(), qu_one(), qu_zero()])
qubit_states = decompose_quantum_state(tp)
print_quantum_state(qubit_states)

