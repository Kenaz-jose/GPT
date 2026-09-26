import numpy as np

def softmax(x, axis=-1):
    """
    Numerically stable softmax implementation along a specified axis.
    """
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def main():
    print("=" * 60)
    print("Part 1: Scaled Dot-Product Attention in Pure NumPy")
    print("=" * 60)

    # 1. Define dimensions
    seq_len = 4       # Sequence length / number of tokens (T)
    d_model = 6       # Input embedding dimension (C)
    d_k = 6           # Key/Query/Value projection dimension

    np.random.seed(42)

    # 2. Simulated Input Embeddings (T, d_model)
    X = np.random.randn(seq_len, d_model)
    print(f"\n[Step 1] Input Sequence (X): shape = {X.shape} (T={seq_len}, d_model={d_model})")

    # 3. Projection Weight Matrices (d_model -> d_k)
    W_Q = np.random.randn(d_model, d_k)
    W_K = np.random.randn(d_model, d_k)
    W_V = np.random.randn(d_model, d_k)

    # 4. Compute Queries, Keys, and Values: Q = X @ W_Q, K = X @ W_K, V = X @ W_V
    Q = np.matmul(X, W_Q)
    K = np.matmul(X, W_K)
    V = np.matmul(X, W_V)

    print(f"\n[Step 2] Projections:")
    print(f"  Query (Q) shape: {Q.shape}  (T x d_k)")
    print(f"  Key   (K) shape: {K.shape}  (T x d_k)")
    print(f"  Value (V) shape: {V.shape}  (T x d_k)")

    # 5. Compute Raw Attention Scores: S = Q @ K^T
    scores = np.matmul(Q, K.T)
    print(f"\n[Step 3] Raw Attention Scores (Q @ K^T): shape = {scores.shape} (T x T)")

    # 6. Scale Attention Scores by sqrt(d_k)
    scale_factor = np.sqrt(d_k)
    scaled_scores = scores / scale_factor
    print(f"\n[Step 4] Scaled Scores (scores / sqrt({d_k})):")

    # 7. Apply Softmax to get Attention Weights
    attn_weights = softmax(scaled_scores, axis=-1)
    print(f"\n[Step 5] Attention Weights (Softmax over rows): shape = {attn_weights.shape}")
    print("Check row sums (all should equal 1.0):", np.round(attn_weights.sum(axis=-1), 3))

    # 8. Compute Output Context Matrix: Output = Attention Weights @ V
    output = np.matmul(attn_weights, V)
    print(f"\n[Step 6] Output Matrix (Attn_Weights @ V): shape = {output.shape} (T x d_k)")

if __name__ == "__main__":
    main()
