import torch

def transform_matrix_batch(input_matrix, l, d, device='cuda'):
    """
    Transform the input tensor of shape (batch_size, l, l) into a larger tensor
    with blocks of identity matrices scaled by the corresponding values.
    
    Args:
        input_matrix (torch.Tensor): Tensor of shape (batch_size, l, l).
        l (int): Sequence length.
        d (int): Dimension of each identity matrix block.
        device (str): Device to perform computations ('cuda' or 'cpu').
    
    Returns:
        torch.Tensor: Transformed tensor of shape (batch_size, l*d, l*d).
    """
    # Ensure the input tensor is on the specified device
    input_matrix = input_matrix.to(device)

    # Get the batch size
    batch_size = input_matrix.size(0)
    
    # Create an identity matrix of size (d, d) on the GPU
    identity_block = torch.eye(d, device=device).unsqueeze(0)  # Shape: (1, d, d)
    
    # Scale the identity block by each element of the input matrix
    expanded_blocks = input_matrix.view(batch_size, l, l, 1, 1) * identity_block
    # expanded_blocks shape: (batch_size, l, l, d, d)

    # Rearrange the tensor to form the final block-diagonal matrix
    result = expanded_blocks.permute(0, 1, 3, 2, 4).reshape(batch_size, l * d, l * d)
    # result shape: (batch_size, l*d, l*d)

    return result

# Example usage
batch_size = 8
l = 256
d = 128

# Generate a random input matrix of shape (batch_size, l, l)
input_matrix = torch.randn(batch_size, l, l)
input_matrix = torch.tril(input_matrix)
print('input:', input_matrix[0])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_matrix = transform_matrix_batch(input_matrix, l, d, device='cpu')

# Print the output shape
print("Output Matrix Shape:", output_matrix[0])
