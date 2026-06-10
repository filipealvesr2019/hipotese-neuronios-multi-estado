import numpy as np
import json
import os
from sklearn.datasets import make_classification, make_moons, make_blobs
from sklearn.preprocessing import StandardScaler

# =========================
# V5.4 Top-k + Curriculum + MNIST-like MoE
# =========================
class V54MoE:
    def __init__(self, input_dim, hidden, output_dim,
                 n_experts=5, k=2, lr=0.01, seed=42, temperature=1.5):
        self.input_dim = input_dim
        self.hidden = hidden
        self.output_dim = output_dim
        self.n_experts = n_experts
        self.k = k
        self.lr = lr
        self.temperature = temperature
        rng = np.random.RandomState(seed)

        # experts
        self.W1 = [rng.randn(input_dim, hidden) * 0.1 for _ in range(n_experts)]
        self.b1 = [np.zeros(hidden) for _ in range(n_experts)]
        self.W2 = [rng.randn(hidden, output_dim) * 0.1 for _ in range(n_experts)]
        self.b2 = [np.zeros(output_dim) for _ in range(n_experts)]

        # gate
        self.gW1 = rng.randn(input_dim, 16) * 0.1
        self.gb1 = np.zeros(16)
        self.gW2 = rng.randn(16, n_experts) * 0.1
        self.gb2 = np.zeros(n_experts)

    def softmax(self, x):
        x = x - np.max(x, axis=1, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=1, keepdims=True)

    def forward(self, x):
        B = x.shape[0]
        expert_outs = []

        for i in range(self.n_experts):
            h = np.maximum(0, x @ self.W1[i] + self.b1[i])
            out = h @ self.W2[i] + self.b2[i]
            expert_outs.append(out)

        g = np.maximum(0, x @ self.gW1 + self.gb1)
        logits = g @ self.gW2 + self.gb2
        probs = self.softmax(logits / self.temperature)

        # Top-k gating
        topk_idx = np.argsort(-probs, axis=1)[:, :self.k]
        gate = np.zeros_like(probs)
        for i in range(B):
            gate[i, topk_idx[i]] = 1.0 / self.k

        out = sum(gate[:, i:i+1] * expert_outs[i] for i in range(self.n_experts))
        return out, gate

    def loss(self, logits, y):
        probs = self.softmax(logits)
        eps = 1e-9
        return -np.mean(np.log(probs[np.arange(len(y)), y] + eps))

    def train_step(self, x, y):
        B = x.shape[0]
        expert_outs = []
        h_list = []

        for i in range(self.n_experts):
            h = np.maximum(0, x @ self.W1[i] + self.b1[i])
            out = h @ self.W2[i] + self.b2[i]
            expert_outs.append(out)
            h_list.append(h)

        g = np.maximum(0, x @ self.gW1 + self.gb1)
        logits_g = g @ self.gW2 + self.gb2
        probs = self.softmax(logits_g / self.temperature)
        topk_idx = np.argsort(-probs, axis=1)[:, :self.k]
        gate = np.zeros_like(probs)
        for i in range(B):
            gate[i, topk_idx[i]] = 1.0 / self.k

        logits = sum(gate[:, i:i+1] * expert_outs[i] for i in range(self.n_experts))

        # gradient output
        probs_out = self.softmax(logits)
        probs_out[np.arange(B), y] -= 1
        probs_out /= B

        for i in range(self.n_experts):
            grad_out = probs_out * gate[:, i:i+1]
            dh = grad_out @ self.W2[i].T
            dh[h_list[i] <= 0] = 0

            dW2 = h_list[i].T @ grad_out
            db2 = np.sum(grad_out, axis=0)
            dW1 = x.T @ dh
            db1 = np.sum(dh, axis=0)

            self.W2[i] -= self.lr * dW2
            self.b2[i] -= self.lr * db2
            self.W1[i] -= self.lr * dW1
            self.b1[i] -= self.lr * db1

        return self.loss(logits, y)

# =========================
# Datasets Controlados
# =========================
def make_dataset(name, n=500, seed=0):
    rng = np.random.RandomState(seed)
    if name == 'xor':
        X = np.array([[0,0],[0,1],[1,0],[1,1]])
        y = np.array([0,1,1,0])
        X = np.tile(X, (n//4, 1))
        y = np.tile(y, n//4)
    elif name == 'gaussian':
        X, y = make_blobs(n_samples=n, centers=2, n_features=2, random_state=seed)
    elif name == 'spiral':
        t = np.linspace(0,4*np.pi,n)
        X = np.vstack([t*np.cos(t), t*np.sin(t)]).T
        y = np.array([0,1]* (n//2))
    elif name == 'mnist_like':
        X, y = make_classification(n_samples=n, n_features=784, n_informative=50,
                                   n_redundant=0, n_classes=10, random_state=seed)
    else:
        raise ValueError('Dataset desconhecido')
    X = StandardScaler().fit_transform(X)
    return X, y

def accuracy(model, X, y):
    logits, _ = model.forward(X)
    pred = np.argmax(logits, axis=1)
    return np.mean(pred == y)

def entropy(p):
    p = p + 1e-9
    return -np.sum(p * np.log(p))

# =========================
# Run Experiment
# =========================
def run():
    datasets = ['xor', 'gaussian', 'spiral', 'mnist_like']
    results = []

    for name in datasets:
        print(f"\n===== DATASET: {name} =====")
        X, y = make_dataset(name, n=500, seed=42)

        model = V54MoE(input_dim=X.shape[1],
                       hidden=32,
                       output_dim=len(np.unique(y)),
                       n_experts=5,
                       k=2,
                       lr=0.01,
                       seed=42)

        for epoch in range(5):
            idx = np.random.permutation(len(X))
            for i in range(0, len(X), 64):
                batch = idx[i:i+64]
                model.train_step(X[batch], y[batch])

        acc = accuracy(model, X, y)
        _, gate_probs = model.forward(X)
        ent = entropy(np.mean(gate_probs, axis=0))
        print(f"V5.4 MOE ACC: {acc:.4f}\nEntropy: {ent:.4f}")

        expert_perf = []
        for i in range(model.n_experts):
            expert_out, _ = model.forward(X)
            expert_pred = np.argmax(expert_out, axis=1)
            perf = np.mean(expert_pred == y)
            expert_perf.append(perf)
        print(f"Expert Perf: {np.array(expert_perf)}\n")

        results.append({
            'dataset': name,
            'acc': float(acc),
            'entropy': float(ent),
            'expert_perf': [float(p) for p in expert_perf]
        })

    os.makedirs("resultados_finais", exist_ok=True)
    with open("resultados_finais/v5_4_topk_moe.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved -> resultados_finais/v5_4_topk_moe.json")

if __name__ == "__main__":
    run()