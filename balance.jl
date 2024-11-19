using LinearAlgebra

include("quantum.jl")

H = 1/sqrt(2) * [1 1; 1 -1]
H2 = tensor_product(H, H)
H3 = tensor_product(H, H2)
H4 = tensor_product(H, H3)

function create_Uf(a::Vector{Int64}, k::Int64=2)
    f(x) = 1/2 * (1 + (-1)^(a'*x % k)) 

    # construct Truth Table based on a
    N = size(a, 1)
    truth_table = zeros(Int64, 2^N)
    
    for i in 1:2^N
        x = digits(i-1, base=2, pad=N)
        truth_table[i] = f(x)
    end

    display(truth_table)

    Uf = zeros(2^(N+1), 2^(N+1)) + I

    # for each 1 in truth table insert NOT matrix
    for (idx, val) in enumerate(truth_table)
        if(val == 0) continue
        else
            row1 = idx*2 - 1
            row2 = idx*2
            temp = Uf[row1, :]
            Uf[row1, :] = Uf[row2, :]
            Uf[row2, :] = temp
        end    
    end

    return Uf
end

a = [1, 2, 3, 4]
# a = [2, 4, 6, 8]
Uf = create_Uf(a)

input = tensor_product_n([[1, 0], [1, 0], [1, 0], [1, 0], [0, 1]])
out = tensor_product(H4, [1 0; 0 1]) * (Uf * (tensor_product(H4, H) * input))
qubit_states = decompose_quantum_state(out)
print_quantum_state(qubit_states)
