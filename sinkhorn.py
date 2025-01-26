import torch

def sinkhorn_knopp(B, num_iterations=10000, tol=1e-6, device="cuda"):
    """
    Solve the entropy-regularized optimal transport problem using Sinkhorn-Knopp algorithm.
    
    Parameters:
    - B: Cost matrix (T x E) (torch.Tensor)
    - num_iterations: Number of iterations for scaling factors
    - tol: Convergence tolerance
    - device: "cuda" or "cpu"
    
    Returns:
    - A: Optimal transport matrix (T x E) (torch.Tensor)
    """
    B = B.to(device)
    T, E = B.shape
    K = torch.exp(B)  # Gibbs kernel
    u = torch.ones(T, device=device)
    v = torch.ones(E, device=device)

    for _ in range(num_iterations):
        u_new = 1.0 / (K @ v)  # Ensure row sum constraint
        v_new = (T/E) / (K.T @ u_new)  # Ensure column sum constraint
        if torch.norm(u_new - u, p=1) < tol and torch.norm(v_new - v, p=1) < tol:
            break
        u, v = u_new, v_new
    
    A = torch.diag(u) @ K @ torch.diag(v)
    return A

# Example usage:
T, E = 5, 4  # Example sizes
B = torch.randn(T, E)  # Random cost matrix

A_sinkhorn = sinkhorn_knopp(B, device="cuda" if torch.cuda.is_available() else "cpu")
import ipdb; ipdb.set_trace()
print("Sinkhorn Solution:\n", A_sinkhorn)
