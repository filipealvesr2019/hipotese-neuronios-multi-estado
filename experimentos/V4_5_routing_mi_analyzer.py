import numpy as np
import json
import os


# =========================
# V4.4 STABLE TRAINABLE
# =========================
class V44SoftBackupTrainable:
    def __init__(self, input_dim, hidden, output_dim,
                 n_states=2, seed=42, lr=0.01,
                 temperature=1.5, top1_ratio=0.9):

        self.input_dim = input_dim
        self.hidden = hidden
        self.output_dim = output_dim
        self.n_states = n_states
        self.lr = lr
        self.temperature = temperature
        self.top1_ratio = top1_ratio

        rng = np.random.RandomState(seed)

        # experts
        self.W1 = [rng.randn(input_dim, hidden) * 0.1 for _ in range(n_states)]
        self.b1 = [np.zeros(hidden) for _ in range(n_states)]
        self.W2 = [rng.randn(hidden, output_dim) * 0.1 for _ in range(n_states)]
        self.b2 = [np.zeros(output_dim) for _ in range(n_states)]

        # gate
        self.gW1 = rng.randn(input_dim, 16) * 0.1
        self.gb1 = np.zeros(16)
        self.gW2 = rng.randn(16, n_states) * 0.1
        self.gb2 = np.zeros(n_states)

    # =========================
    def softmax(self, x):
        x = x - np.max(x, axis=1, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=1, keepdims=True)

    # =========================
    def forward(self, x):
        h_states = []
        out_states = []

        for i in range(self.n_states):
            h = np.maximum(0, x @ self.W1[i] + self.b1[i])
            out = h @ self.W2[i] + self.b2[i]
            h_states.append(h)
            out_states.append(out)

        g = np.maximum(0, x @ self.gW1 + self.gb1)
        logits_g = g @ self.gW2 + self.gb2

        gate = self.softmax(logits_g / self.temperature)

        out = sum(gate[:, i:i+1] * out_states[i] for i in range(self.n_states))

        return out, gate

    # =========================
    def train_step(self, x, y):
        B = x.shape[0]

        # forward
        h_states = []
        out_states = []

        for i in range(self.n_states):
            h = np.maximum(0, x @ self.W1[i] + self.b1[i])
            out = h @ self.W2[i] + self.b2[i]
            h_states.append(h)
            out_states.append(out)

        g = np.maximum(0, x @ self.gW1 + self.gb1)
        logits_g = g @ self.gW2 + self.gb2
        gate = self.softmax(logits_g / self.temperature)

        logits = sum(gate[:, i:i+1] * out_states[i] for i in range(self.n_states))

        probs = self.softmax(logits)

        # gradient
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
        # UPDATE GATE (🔥 FIX PRINCIPAL)
        # =========================

        # pseudo-loss: experts que acertam reforçam gate
        expert_scores = np.zeros_like(gate)

        preds = np.argmax(logits, axis=1)

        for i in range(self.n_states):
            expert_scores[:, i] = (preds == y).astype(float)

        gate_grad = (expert_scores - gate) / B

        dg = gate_grad @ self.gW2.T
        dg[g <= 0] = 0

        dWg2 = g.T @ gate_grad
        dbg2 = np.sum(gate_grad, axis=0)

        dWg1 = x.T @ dg
        dbg1 = np.sum(dg, axis=0)

        self.gW2 += self.lr * dWg2
        self.gb2 += self.lr * dbg2
        self.gW1 += self.lr * dWg1
        self.gb1 += self.lr * dbg1

        return -np.mean(np.log(probs[np.arange(B), y] + 1e-9))


# =========================
def make_data(n=2000, d=784, c=10, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    y = rng.randint(0, c, n)
    return X, y


def accuracy(model, X, y):
    logits, _ = model.forward(X)
    return np.mean(np.argmax(logits, axis=1) == y)


def entropy(p):
    p = p + 1e-9
    return -np.sum(p * np.log(p))


# =========================
def run():
    results = []

    for seed in range(10):
        print(f"\n=== Seed {seed} ===")

        X, y = make_data(seed=seed)

        model = V44SoftBackupTrainable(
            input_dim=784,
            hidden=96,
            output_dim=10,
            n_states=2,
            seed=seed
        )

        for epoch in range(3):
            idx = np.random.permutation(len(X))
            for i in range(0, len(X), 64):
                batch = idx[i:i+64]
                model.train_step(X[batch], y[batch])

        acc = accuracy(model, X, y)

        _, gate = model.forward(X)
        ent = entropy(np.mean(gate, axis=0))

        print(f"Acc={acc:.4f} | Entropy={ent:.4f}")

        results.append({
            "seed": seed,
            "acc": float(acc),
            "entropy": float(ent)
        })

    os.makedirs("resultados_finais", exist_ok=True)
    with open("resultados_finais/v4_4_soft_backup_trainable.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved.")


if __name__ == "__main__":
    run()