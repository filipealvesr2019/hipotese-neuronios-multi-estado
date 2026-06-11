import numpy as np
import json
import os

# =========================
# V5.3B — Routing Credit Assignment MoE (improved)
# =========================
class V53BCreditMoE:
    def __init__(self, input_dim, hidden, output_dim,
                 n_experts=5, top_k=2, seed=42, lr=0.01, ema=0.9,
                 entropy_bonus=0.01):
        self.input_dim = input_dim
        self.hidden = hidden
        self.output_dim = output_dim
        self.n_experts = n_experts
        self.top_k = top_k
        self.lr = lr
        self.ema = ema
        self.entropy_bonus = entropy_bonus

        rng = np.random.RandomState(seed)

        # Experts weights
        self.W1 = [rng.randn(input_dim, hidden) * 0.1 for _ in range(n_experts)]
        self.b1 = [np.zeros(hidden) for _ in range(n_experts)]
        self.W2 = [rng.randn(hidden, output_dim) * 0.1 for _ in range(n_experts)]
        self.b2 = [np.zeros(output_dim) for _ in range(n_experts)]

        # Gate
        self.gW1 = rng.randn(input_dim, 16) * 0.1
        self.gb1 = np.zeros(16)
        self.gW2 = rng.randn(16, n_experts) * 0.1
        self.gb2 = np.zeros(n_experts)

        # Expert performance tracking
        self.expert_perf = np.zeros(n_experts)

    # =========================
    # Softmax
    # =========================
    def softmax(self, x):
        x = x - np.max(x, axis=1, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=1, keepdims=True)

    # =========================
    # Forward
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

        # Top-k soft routing
        topk_idx = np.argsort(gate_probs, axis=1)[:, -self.top_k:]
        gate = np.zeros_like(gate_probs)
        for b in range(B):
            for idx in topk_idx[b]:
                gate[b, idx] = 1.0 / self.top_k

        # Clip to prevent one expert dominance
        gate = np.minimum(gate, 0.7)
        out = sum(gate[:, i:i+1] * out_list[i] for i in range(self.n_experts))
        return out, gate, out_list, h_list

    # =========================
    # Cross-entropy loss
    # =========================
    def loss(self, logits, y):
        probs = self.softmax(logits)
        eps = 1e-9
        return -np.mean(np.log(probs[np.arange(len(y)), y] + eps))

    # =========================
    # Train step with credit assignment
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

        # EMA for dynamic pruning / expert tracking
        self.expert_perf = self.ema * self.expert_perf + (1 - self.ema) * expert_acc

        acc = np.mean(np.argmax(logits, axis=1) == y)
        return acc, gate, expert_acc

# =========================
# Metrics
# =========================
def entropy(gate):
    p = np.mean(gate, axis=0) + 1e-9
    return -np.sum(p * np.log(p))

def mutual_information(gate):
    B, n = gate.shape
    p_y = np.mean(gate, axis=0)
    H_y = -np.sum(p_y * np.log(p_y + 1e-9))
    H_y_given_x = np.mean([-np.sum(gate[b] * np.log(gate[b] + 1e-9)) for b in range(B)])
    return H_y - H_y_given_x

def collapse(gate):
    usage = np.mean(gate, axis=0)
    return np.sum(usage**2)

def usage(gate):
    return np.mean(gate, axis=0)

# =========================
# Dataset helpers
# =========================
def make_xor(n=2000, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randint(0, 2, (n, 2))
    y = np.logical_xor(X[:, 0], X[:, 1]).astype(int)
    return X, y

def make_gaussian(n=2000, seed=0):
    rng = np.random.RandomState(seed)
    X0 = rng.randn(n//2, 2)
    X1 = rng.randn(n//2, 2) + 3
    X = np.vstack([X0, X1])
    y = np.array([0]*(n//2) + [1]*(n//2))
    return X, y

def make_spiral(n=2000, seed=0):
    rng = np.random.RandomState(seed)
    n_classes = 2
    X, y = [], []
    for j in range(n_classes):
        r = np.linspace(0.0, 1, n//n_classes)
        t = np.linspace(j*3.14, (j+1)*3.14, n//n_classes) + rng.randn(n//n_classes)*0.2
        x1 = r * np.sin(t)
        x2 = r * np.cos(t)
        X.append(np.stack([x1, x2], axis=1))
        y.append(np.full(n//n_classes, j))
    return np.vstack(X), np.hstack(y)

def make_mnist_like(n=2000, d=784, c=10, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    y = rng.randint(0, c, n)
    return X, y

# =========================
# Runner
# =========================
def run():
    datasets = {
        "xor": make_xor,
        "gaussian": make_gaussian,
        "spiral": make_spiral,
        "mnist_like": make_mnist_like
    }

    results = {}

    for name, fn in datasets.items():
        print(f"\n===== DATASET: {name} =====")
        X, y = fn()
        model = V53BCreditMoE(
            input_dim=X.shape[1],
            hidden=32,
            output_dim=len(np.unique(y)),
            n_experts=5,
            top_k=2,
            lr=0.01,
            ema=0.9,
            entropy_bonus=0.01
        )

        for epoch in range(3):
            idx = np.random.permutation(len(X))
            batch_size = 64
            for i in range(0, len(X), batch_size):
                batch = idx[i:i+batch_size]
                acc, gate, expert_acc = model.train_step(X[batch], y[batch])

        Ent = entropy(gate)
        Coll = collapse(gate)
        MI = mutual_information(gate)
        Usage = usage(gate)

        print(f"V5.3B MOE ACC: {acc:.4f}")
        print(f"Entropy: {Ent:.4f}")
        print(f"Collapse: {Coll:.4f}")
        print(f"MI: {MI:.4f}")
        print(f"Usage: {np.round(Usage,3)}")
        print(f"Expert Perf: {model.expert_perf}")

        results[name] = {
            "ACC": float(acc),
            "Entropy": float(Ent),
            "Collapse": float(Coll),
            "MI": float(MI),
            "Usage": Usage.tolist(),
            "Expert Perf": model.expert_perf.tolist()
        }

    os.makedirs("resultados_finais", exist_ok=True)
    with open("resultados_finais/v5_3B_routing_credit_assignment.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved -> resultados_finais/v5_3B_routing_credit_assignment.json")

if __name__ == "__main__":
    run()