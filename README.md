
# Quantum Computing Toolkit

A lightweight Julia toolkit for quantum computing simulations and state manipulation. This toolkit provides essential functions for working with quantum states, tensor products, and quantum state analysis.

## Features

- Tensor product operations for vectors and matrices
- Quantum state creation and manipulation
- State decomposition and analysis
- Pretty printing of quantum states
- Verification tools for quantum operations

## Quick Start

```julia
include("quantum.jl")

# Create basic quantum states
zero_state = qu_zero()  # Returns |0⟩
one_state = qu_one()    # Returns |1⟩

# Create multi-qubit states using tensor products
# Example: Create |0001⟩ state
state = tensor_product_n([qu_zero(), qu_zero(), qu_zero(), qu_one()])

# Analyze quantum states
qubit_states = decompose_quantum_state(state)
print_quantum_state(qubit_states)  # Prints: |0001⟩

# Perform detailed analysis of quantum states
print_quantum_analysis(state)
```

## Core Functions

### State Creation and Manipulation

- `qu_zero()`: Returns the state vector for |0⟩
- `qu_one()`: Returns the state vector for |1⟩
- `tensor_product(v1, v2)`: Computes tensor product of two vectors or matrices
- `tensor_product_n(vectors)`: Computes tensor product of multiple vectors or matrices

### Analysis Tools

- `decompose_quantum_state(state)`: Decomposes a quantum state vector into individual qubit states
- `verify_quantum_decomposition(state, qubit_states)`: Verifies the accuracy of state decomposition
- `analyze_quantum_state(state)`: Provides detailed analysis of quantum states and their probabilities
- `identify_state_pattern(state)`: Attempts to identify patterns in quantum states (e.g., superpositions)

### Visualization and Output

- `print_quantum_state(qubit_states)`: Pretty prints quantum states in bra-ket notation
- `print_quantum_analysis(state)`: Provides detailed analysis output including probabilities

## Examples

### Creating and Analyzing a Bell State
```julia
# Create a Bell state (|00⟩ + |11⟩)/√2
H = 1/sqrt(2) * [1 1; 1 -1]  # Hadamard gate
bell_state = tensor_product(H, Matrix(I, 2, 2)) * tensor_product_n([qu_zero(), qu_zero()])

# Analyze the state
print_quantum_analysis(bell_state)
```

### Working with Multi-Qubit Systems
```julia
# Create a 4-qubit system |1001⟩
state = zeros(16)
state[10] = 1.0  # 9 in zero-based indexing
qubit_states = decompose_quantum_state(state)
print_quantum_state(qubit_states)  # Prints: |1001⟩

# Verify the decomposition
err = verify_quantum_decomposition(state, qubit_states)
println("Reconstruction error: ", err)
```

## Advanced Usage

The toolkit also supports:
- Arbitrary-sized quantum systems (limited by memory)
- Complex quantum state analysis with customizable thresholds
- Error checking and validation for quantum operations
- Integration with standard Julia linear algebra operations

Check out the included example files (`decomp.jl` and `balance.jl`) for more advanced usage patterns and quantum algorithms.

## Contributing

Feel free to submit issues and enhancement requests!

## Notes

- All operations are performed with floating-point precision
- The toolkit assumes normalized quantum states
- For large quantum systems, be mindful of memory usage as state vectors grow exponentially with the number of qubits
