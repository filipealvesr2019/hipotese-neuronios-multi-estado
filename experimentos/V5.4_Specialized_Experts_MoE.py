import numpy as np
import os
import json

# =========================
# V5.4 — Specialized Experts MoE (Heterogeneous Experts)
# =========================
class V54_SpecializedMoE:
    def __init__(self, input_dim, output_dim, expert_sizes=None, top_k=2, lr=0.01, ema=0.9):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_experts = len(expert_sizes)
        self.top_k = top_k
        self.lr = lr
        self.ema = ema
        self.hidden_sizes = expert_sizes

        rng = np.random.RandomState(42)

        # Experts weights
        self.W1 = []
        self.b1 = []
        self.W2 = []
        self.b2 = []

        for h in self.hidden_sizes:
            self.W1.append(rng.randn(input_dim, h) * 0.1)
            self.b1.append(np.zeros(h))
            self.W2.append(rng.randn(h, output_dim) * 0.1)
            self.b2.append(np.zeros(output_dim))

        # Gate
        self.gW1 = rng.randn(input_dim, 16) * 0.1
        self.gb1 = np.zeros(16)
        self.gW2 = rng.randn(16, self.n_experts) * 0.1
        self.gb2 = np.zeros(self.n_experts)

        # Expert performance tracking
        self.expert_perf = np.zeros(self.n_experts)

    # =========================
    # Softmax
    # =========================
    def softmax(self, x):
        x = x - np.max(x, axis=1, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=1, keepdims=True)

    # =========================
    # Forward pass
    # =========================
    def forward(self, X):
        B = X.shape[0]
        h_list, out_list = [], []
        for i in range(self.n_experts):
            h = np.maximum(0, X @ self.W1[i] + self.b1[i])
            out = h @ self.W2[i] + self.b2[i]
            h_list.append(h)
            out_list.append(out)

        g = np.maximum(0, X @ self.gW1 + self.gb1)
        logits = g @ self.gW2 + self.gb2
        gate_probs = self.softmax(logits)

        topk_idx = np.argsort(gate_probs, axis=1)[:, -self.top_k:]
        gate = np.zeros_like(gate_probs)
        for b in range(B):
            for idx in topk_idx[b]:
                gate[b, idx] = 1.0 / self.top_k

        out = sum(gate[:, i:i+1] * out_list[i] for i in range(self.n_experts))
        return out, gate, out_list, h_list

    # =========================
    # Loss (cross-entropy)
    # =========================
    def loss(self, logits, y):
        probs = self.softmax(logits)
        eps = 1e-9
        return -np.mean(np.log(probs[np.arange(len(y)), y] + eps))

    # =========================
    # Train step with reward-aware credit assignment
    # =========================
    def train_step(self, X, y):
        B = X.shape[0]
        logits, gate, out_list, h_list = self.forward(X)
        probs_out = self.softmax(logits)
        probs_out[np.arange(B), y] -= 1
        probs_out /= B

        expert_acc = np.zeros(self.n_experts)

        for i in range(self.n_experts):
            grad_out = probs_out * gate[:, i:i+1]
            dh = grad_out @ self.W2[i].T
            dh[h_list[i] <= 0] = 0
            dW2 = h_list[i].T @ grad_out
            db2 = np.sum(grad_out, axis=0)
            dW1 = X.T @ dh
            db1 = np.sum(dh, axis=0)

            self.W2[i] -= self.lr * dW2
            self.b2[i] -= self.lr * db2
            self.W1[i] -= self.lr * dW1
            self.b1[i] -= self.lr * db1

            pred = np.argmax(out_list[i], axis=1)
            expert_acc[i] = np.mean(pred == y)

        # EMA for expert performance
        self.expert_perf = self.ema * self.expert_perf + (1 - self.ema) * expert_acc

        return self.loss(logits, y), expert_acc

# =========================
# Dataset helpers
# =========================
def make_xor(n=500, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randint(0, 2, (n, 2))
    y = np.logical_xor(X[:,0], X[:,1]).astype(int)
    return X, y

def make_gaussian(n=500, seed=0):
    rng = np.random.RandomState(seed)
    X0 = rng.randn(n//2, 2)
    X1 = rng.randn(n//2, 2) + 2
    X = np.vstack([X0, X1])
    y = np.array([0]*(n//2) + [1]*(n//2))
    return X, y

def make_spiral(n=500, noise=0.2, seed=0):
    rng = np.random.RandomState(seed)
    n_classes = 2
    X = []
    y = []
    for j in range(n_classes):
        r = np.linspace(0.0, 1, n//n_classes)
        t = np.linspace(j*3.14, (j+1)*3.14, n//n_classes) + rng.randn(n//n_classes)*noise
        x1 = r * np.sin(t)
        x2 = r * np.cos(t)
        X.append(np.vstack([x1, x2]).T)
        y.append([j]*(n//n_classes))
    X = np.vstack(X)
    y = np.hstack(y)
    return X, y

def make_mnist_like(n=500, d=784, c=10, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    y = rng.randint(0, c, n)
    return X, y
def accuracy(model, X, y):
    logits, _, _, _ = model.forward(X)
    pred = np.argmax(logits, axis=1)
    return np.mean(pred == y)

def entropy(gate):
    p = np.mean(gate, axis=0) + 1e-9
    return -np.sum(p * np.log(p))

def collapse(gate):
    return 1 - np.mean(np.var(gate, axis=1))

def mutual_info(gate, y):
    B, n_experts = gate.shape
    mi = 0
    for i in range(n_experts):
        p_e = np.mean(gate[:,i]) + 1e-9
        p_e_y = np.mean(gate[:,i][y==1]) + 1e-9
        mi += p_e_y * np.log(p_e_y/p_e)
    return mi

# =========================
# Experiment runner
# =========================
def run():
    datasets = {
        "xor": make_xor,
        "gaussian": make_gaussian,
        "spiral": make_spiral,
        "mnist_like": make_mnist_like
    }

    # Specialized hidden sizes per dataset
    specialized_sizes = {
        "xor": [16, 32, 16, 16, 16],
        "gaussian": [32, 32, 32, 16, 16],
        "spiral": [32, 32, 32, 16, 16],
        "mnist_like": [128, 128, 64, 64, 64]
    }

    results = {}

    for name, fn in datasets.items():
        print(f"\n===== DATASET: {name} =====")
        X, y = fn()
        print(f"X shape: {X.shape}, y shape: {y.shape}")

        model = V54_SpecializedMoE(
            input_dim=X.shape[1],
            output_dim=len(np.unique(y)),
            expert_sizes=specialized_sizes[name],
            top_k=2,
            lr=0.01
        )

        for epoch in range(3):
            idx = np.random.permutation(len(X))
            batch_size = 64
            for i in range(0, len(X), batch_size):
                batch = idx[i:i+batch_size]
                if len(batch) == 0:
                    continue
                loss_val, expert_acc = model.train_step(X[batch], y[batch])

        acc = accuracy(model, X, y)
        _, gate, _, _ = model.forward(X)
        ent = entropy(gate)
        col = collapse(gate)
        mi = mutual_info(gate, y)
        usage = np.mean(gate, axis=0)

        # FLOPs estimate: sum over experts
        flop = 0
        for i, h in enumerate(model.hidden_sizes):
            flop += X.shape[0] * X.shape[1] * h + X.shape[0] * h * model.output_dim
        # Adjust by gate usage
        flop = flop * np.mean(np.sum(gate, axis=1))

        # Score balances accuracy x FLOPs
        score = acc / (1 + flop/1e6)

        print(f"V5.4 MOE ACC: {acc:.4f}")
        print(f"Entropy: {ent:.4f}")
        print(f"Collapse: {col:.4f}")
        print(f"MI: {mi:.4f}")
        print(f"Usage: {np.round(usage,3)}")
        print(f"FLOPs (est.): {int(flop)}")
        print(f"Score: {score:.4f}")
        print(f"Expert Perf: {model.expert_perf}")

        results[name] = {
            "acc": float(acc),
            "entropy": float(ent),
            "collapse": float(col),
            "MI": float(mi),
            "usage": usage.tolist(),
            "FLOPs": int(flop),
            "score": float(score),
            "expert_perf": model.expert_perf.tolist()
        }

    os.makedirs("resultados_finais", exist_ok=True)
    with open("resultados_finais/v5_4_specialized_experts.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved -> resultados_finais/v5_4_specialized_experts.json")


if __name__ == "__main__":
    run()