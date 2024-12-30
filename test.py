import torch

# Create example tensors
batch_size = 2
seq_length = 4
hidden_dim = 3

# Sample input tensor (batch_size, seq_length, hidden_dim)
moe_inp = torch.tensor([
    [[1.0, 2.0, 3.0],
     [4.0, 5.0, 6.0],
     [7.0, 8.0, 9.0],
     [2.0, 3.0, 4.0]],
    
    [[5.0, 6.0, 7.0],
     [8.0, 9.0, 1.0],
     [2.0, 3.0, 4.0],
     [5.0, 6.0, 7.0]]
])

# Sample output tensor with same shape
moe_outp = torch.tensor([
    [[0.1, 0.2, 0.3],
     [0.4, 0.5, 0.6],
     [0.7, 0.8, 0.9],
     [0.2, 0.3, 0.4]],
    
    [[0.5, 0.6, 0.7],
     [0.8, 0.9, 0.1],
     [0.2, 0.3, 0.4],
     [0.5, 0.6, 0.7]]
])

# 1. Calculate L2 norm along dim=1 (sequence dimension)
l2_norm_c = torch.norm(moe_inp, p=2, dim=1, keepdim=True) + 1e-8

# 2. Normalize input
moe_inp_normalized_c = moe_inp / l2_norm_c

# 3. Element-wise multiplication
result = moe_outp * moe_inp_normalized_c

print(moe_outp)
print("Original input shape:", moe_inp)
print("L2 norm shape:", l2_norm_c)
print("Normalized input shape:", moe_inp_normalized_c)
print("Final result shape:", result)