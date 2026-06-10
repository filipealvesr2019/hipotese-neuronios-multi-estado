import numpy as np
import json

# =========================
# Funções utilitárias
# =========================
def set_seed(seed=42):
    np.random.seed(seed)

def make_circles(n_samples=2000, noise=0.1, seed=42):
    np.random.seed(seed)
    t = 2 * np.pi * np.random.rand(n_samples)
    x = np.stack([np.cos(t), np.sin(t)], axis=1)
    x += noise * np.random.randn(*x.shape)
    y = (t > np.pi).astype(int)
    return x, y

def train_test_split(X, y, test_size=0.2):
    n = X.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(n*(1-test_size))
    return X[idx[:split]], X[idx[split:]], y[idx[:split]], y[idx[split:]]

def normalize(X):
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-8
    return (X - mean) / std, mean, std

def softmax_crossentropy(logits, y_true):
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    loss = -np.mean(np.log(probs[np.arange(len(y_true)), y_true]+1e-12))
    grad = probs.copy()
    grad[np.arange(len(y_true)), y_true] -= 1
    grad /= len(y_true)
    return loss, grad

# =========================
# MultiState Residual Layer V4.3
# =========================
class MultiStateResidualV43:
    def __init__(self, input_dim, output_dim, n_states=4, seed=42, temperature=1.5, gate_hidden=8, skip_scale=0.3):
        self.n_states = n_states
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.temperature = temperature
        self.gate_hidden = gate_hidden
        self.skip_scale = skip_scale

        rng = np.random.RandomState(seed)
        rng2 = np.random.RandomState(seed+100)
        rng3 = np.random.RandomState(seed+200)

        # Estados
        self.W = [rng.randn(input_dim, output_dim)*0.1 for _ in range(n_states)]
        self.b = [np.zeros(output_dim) for _ in range(n_states)]

        # Gate
        self.gate_W1 = rng2.randn(input_dim, gate_hidden)*0.1
        self.gate_b1 = np.zeros(gate_hidden)
        self.gate_W2 = rng2.randn(gate_hidden, n_states)*0.1
        self.gate_b2 = np.zeros(n_states)

        # Skip
        self.skip_W = rng3.randn(input_dim, output_dim)*0.1
        self.skip_b = np.zeros(output_dim)

    def forward(self, x):
        B = x.shape[0]
        states_out = [x @ w + b for w,b in zip(self.W, self.b)]

        # Gate MLP
        h = np.maximum(0, x @ self.gate_W1 + self.gate_b1)
        logits = h @ self.gate_W2 + self.gate_b2
        logits /= self.temperature
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)

        # Top-2 hard routing
        top2 = np.argsort(probs, axis=1)[:, -2:]
        gw = np.zeros_like(probs)
        for i in range(B):
            gw[i, top2[i]] = probs[i, top2[i]]
        gw /= gw.sum(axis=1, keepdims=True)

        # Saída Residual
        out = sum(gw[:,i:i+1]*states_out[i] for i in range(self.n_states))
        out += self.skip_scale*(x @ self.skip_W + self.skip_b)
        return out

# =========================
# Treino simplificado
# =========================
def run_v43_residual(n_seeds=10):
    results = []

    for seed in range(n_seeds):
        print(f"\n=== Seed {seed} ===")
        set_seed(seed)

        X, y = make_circles(n_samples=2000, noise=0.1, seed=seed)
        X, _, _ = normalize(X)
        Xtr, Xva, ytr, yva = train_test_split(X, y)

        model = MultiStateResidualV43(input_dim=2, output_dim=2, n_states=4, seed=seed)

        # Treino simplificado placeholder
        for ep in range(50):
            idx = np.random.permutation(Xtr.shape[0])
            for s in range(0, Xtr.shape[0], 64):
                ids = idx[s:s+64]
                Xb, yb = Xtr[ids], ytr[ids]
                _ = model.forward(Xb)

        acc = np.mean(np.argmax(model.forward(Xva), axis=1) == yva)
        print(f"V4.3 Residual acc (forward only): {acc:.4f}")
        results.append({"seed": seed, "acc": acc})

    # Salvar resultados
    import os, json
    output_path = os.path.join(os.getcwd(), "resultados_finais", "v4_3_residual_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResultados salvos em {output_path}")
    return results

# =========================
# Rodar direto
# =========================
if __name__ == "__main__":
    print("=== V4.3 Residual Test ===")
    run_v43_residual(n_seeds=10)