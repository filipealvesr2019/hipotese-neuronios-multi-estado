import numpy as np
import json
import os


# =========================
# V4.5 — FIXED MoE + ROUTING ANALYSIS
# =========================
class V45RoutingMIAnalyzer:
    def __init__(self, input_dim, hidden, output_dim,
                 n_states=2, seed=42, lr=0.01,
                 temperature=1.5, top1_ratio=0.9,
                 entropy_reg=0.005):

        self.input_dim = input_dim
        self.hidden = hidden
        self.output_dim = output_dim
        self.n_states = n_states
        self.lr = lr
        self.temperature = temperature
        self.top1_ratio = top1_ratio
        self.entropy_reg = entropy_reg

        rng = np.random.RandomState(seed)

        # =========================
        # Experts
        # =========================
        self.W1 = [rng.randn(input_dim, hidden) * 0.1 for _ in range(n_states)]
        self.b1 = [np.zeros(hidden) for _ in range(n_states)]
        self.W2 = [rng.randn(hidden, output_dim) * 0.1 for _ in range(n_states)]
        self.b2 = [np.zeros(output_dim) for _ in range(n_states)]

        # =========================
        # Gate
        # =========================
        self.gW1 = rng.randn(input_dim, 16) * 0.1
        self.gb1 = np.zeros(16)
        self.gW2 = rng.randn(16, n_states) * 0.1
        self.gb2 = np.zeros(n_states)

    # =========================
    def relu(self, x):
        return np.maximum(0, x)

    # =========================
    def softmax(self, x):
        x = x - np.max(x, axis=1, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=1, keepdims=True)

    # =========================
    # FORWARD
    # =========================
    def forward(self, x, add_noise=False):
        B = x.shape[0]

        h_states = []
        out_states = []

        for i in range(self.n_states):
            h = self.relu(x @ self.W1[i] + self.b1[i])
            out = h @ self.W2[i] + self.b2[i]
            h_states.append(h)
            out_states.append(out)

        g = self.relu(x @ self.gW1 + self.gb1)
        logits = g @ self.gW2 + self.gb2

        # quebra simetria
        if add_noise:
            logits += np.random.normal(0, 0.05, logits.shape)

        gate_probs = self.softmax(logits / self.temperature)

        # soft top1 backup routing
        top1 = np.argmax(gate_probs, axis=1)
        gate = np.zeros_like(gate_probs)

        for i in range(B):
            gate[i, top1[i]] = self.top1_ratio
            for j in range(self.n_states):
                if j != top1[i]:
                    gate[i, j] += (1 - self.top1_ratio) / (self.n_states - 1)

        gate = gate / gate.sum(axis=1, keepdims=True)

        out = sum(gate[:, i:i+1] * out_states[i] for i in range(self.n_states))

        return out, gate, out_states, h_states

    # =========================
    def softmax_1d(self, x):
        x = x - np.max(x)
        e = np.exp(x)
        return e / np.sum(e)

    # =========================
    def loss(self, logits, y):
        probs = self.softmax(logits)
        return -np.mean(np.log(probs[np.arange(len(y)), y] + 1e-9))

    # =========================
    # TRAIN STEP (FIXED)
    # =========================
    def train_step(self, x, y):
        B = x.shape[0]

        logits, gate, out_states, h_states = self.forward(x, add_noise=True)

        # =========================
        # output gradient (cross entropy)
        # =========================
        probs = self.softmax(logits)
        probs[np.arange(B), y] -= 1
        probs /= B

        # =========================
        # UPDATE EXPERTS
        # =========================
        for i in range(self.n_states):

            grad = probs * gate[:, i:i+1]

            dh = grad @ self.W2[i].T
            dh[h_states[i] <= 0] = 0

            dW2 = h_states[i].T @ grad
            db2 = np.sum(grad, axis=0)

            dW1 = x.T @ dh
            db1 = np.sum(dh, axis=0)

            self.W2[i] -= self.lr * dW2
            self.b2[i] -= self.lr * db2
            self.W1[i] -= self.lr * dW1
            self.b1[i] -= self.lr * db1

        # =========================
        # ROUTING STABILITY REGULARIZATION
        # =========================
        gate_mean = np.mean(gate, axis=0)
        entropy = -np.sum(gate_mean * np.log(gate_mean + 1e-9))

        # encourage non-collapse
        entropy_loss = -self.entropy_reg * entropy

        return self.loss(logits, y) + entropy_loss


# =========================
# DATASET
# =========================
def make_data(n=2000, d=784, c=10, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    y = rng.randint(0, c, n)
    return X, y


def accuracy(model, X, y):
    logits, _, _, _ = model.forward(X)
    return np.mean(np.argmax(logits, axis=1) == y)


def entropy(p):
    p = p + 1e-9
    return -np.sum(p * np.log(p))


# =========================
# RUN EXPERIMENT
# =========================
def run():
    results = []

    for seed in range(10):
        print(f"\n=== Seed {seed} ===")

        X, y = make_data(seed=seed)

        model = V45RoutingMIAnalyzer(
            input_dim=784,
            hidden=96,
            output_dim=10,
            n_states=2,
            seed=seed,
            lr=0.01,
            temperature=1.5,
            top1_ratio=0.9,
            entropy_reg=0.005
        )

        # training
        for epoch in range(3):
            idx = np.random.permutation(len(X))
            for i in range(0, len(X), 64):
                batch = idx[i:i+64]
                model.train_step(X[batch], y[batch])

        acc = accuracy(model, X, y)

        _, gate, _, _ = model.forward(X)
        ent = entropy(np.mean(gate, axis=0))

        print(f"Acc={acc:.4f} | Entropy={ent:.4f}")

        results.append({
            "seed": seed,
            "acc": float(acc),
            "entropy": float(ent)
        })

    os.makedirs("resultados_finais", exist_ok=True)

    with open("resultados_finais/v4_5_routing_mi_analyzer.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved -> resultados_finais/v4_5_routing_mi_analyzer.json")


if __name__ == "__main__":
    run()