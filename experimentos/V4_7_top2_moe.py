import numpy as np
import json
import os


# =========================
# V4.7 — TOP-2 STABLE MoE
# =========================
class V47Top2MoE:
    def __init__(self, input_dim, hidden, output_dim,
                 n_states=4, seed=42, lr=0.01,
                 temperature=1.0, lb_coef=0.01):

        self.input_dim = input_dim
        self.hidden = hidden
        self.output_dim = output_dim
        self.n_states = n_states
        self.lr = lr
        self.temperature = temperature
        self.lb_coef = lb_coef

        rng = np.random.RandomState(seed)

        # =====================
        # Experts
        # =====================
        self.W1 = [rng.randn(input_dim, hidden) * 0.1 for _ in range(n_states)]
        self.b1 = [np.zeros(hidden) for _ in range(n_states)]
        self.W2 = [rng.randn(hidden, output_dim) * 0.1 for _ in range(n_states)]
        self.b2 = [np.zeros(output_dim) for _ in range(n_states)]

        # =====================
        # Gate
        # =====================
        self.gW1 = rng.randn(input_dim, 32) * 0.1
        self.gb1 = np.zeros(32)
        self.gW2 = rng.randn(32, n_states) * 0.1
        self.gb2 = np.zeros(n_states)

    # =====================
    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        x = x - np.max(x, axis=1, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=1, keepdims=True)

    # =====================
    # FORWARD
    # =====================
    def forward(self, x):
        B = x.shape[0]

        # experts
        h_states = []
        out_states = []

        for i in range(self.n_states):
            h = self.relu(x @ self.W1[i] + self.b1[i])
            out = h @ self.W2[i] + self.b2[i]
            h_states.append(h)
            out_states.append(out)

        # gate
        g = self.relu(x @ self.gW1 + self.gb1)
        logits = g @ self.gW2 + self.gb2

        probs = self.softmax(logits / self.temperature)

        # =====================
        # TOP-2 ROUTING (CORE FIX)
        # =====================
        top2 = np.argsort(probs, axis=1)[:, -2:]

        gate = np.zeros_like(probs)

        for i in range(B):
            a, b = top2[i]

            pa, pb = probs[i, a], probs[i, b]
            s = pa + pb + 1e-9

            gate[i, a] = pa / s
            gate[i, b] = pb / s

        # normalize safety
        gate = gate / gate.sum(axis=1, keepdims=True)

        out = sum(gate[:, i:i+1] * out_states[i] for i in range(self.n_states))

        return out, gate, probs

    # =====================
    def loss(self, logits, y):
        probs = self.softmax(logits)
        return -np.mean(np.log(probs[np.arange(len(y)), y] + 1e-9))

    # =====================
    # TRAIN STEP
    # =====================
    def train_step(self, x, y):
        B = x.shape[0]

        logits, gate, probs_gate = self.forward(x)

        # =====================
        # OUTPUT GRADIENT
        # =====================
        probs = self.softmax(logits)
        probs[np.arange(B), y] -= 1
        probs /= B

        # =====================
        # EXPERT UPDATE
        # =====================
        for i in range(self.n_states):

            grad = probs * gate[:, i:i+1]

            h = self.relu(x @ self.W1[i] + self.b1[i])

            dh = grad @ self.W2[i].T
            dh[h <= 0] = 0

            self.W2[i] -= self.lr * (h.T @ grad)
            self.b2[i] -= self.lr * np.sum(grad, axis=0)
            self.W1[i] -= self.lr * (x.T @ dh)
            self.b1[i] -= self.lr * np.sum(dh, axis=0)

        # =====================
        # LOAD BALANCING LOSS (IMPORTANT)
        # =====================
        usage = gate.mean(axis=0)
        lb_loss = np.sum(usage * usage)

        return self.loss(logits, y) + self.lb_coef * lb_loss


# =========================
def make_data(n=2000, d=784, c=10, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    y = rng.randint(0, c, n)
    return X, y


def accuracy(model, X, y):
    logits, _, _ = model.forward(X)
    return np.mean(np.argmax(logits, axis=1) == y)


def run():
    results = []

    for seed in range(10):
        print(f"\n=== Seed {seed} ===")

        X, y = make_data(seed=seed)

        model = V47Top2MoE(
            input_dim=784,
            hidden=96,
            output_dim=10,
            n_states=4,
            seed=seed
        )

        for epoch in range(4):
            idx = np.random.permutation(len(X))
            for i in range(0, len(X), 64):
                batch = idx[i:i+64]
                model.train_step(X[batch], y[batch])

        acc = accuracy(model, X, y)

        print(f"Acc={acc:.4f}")

        results.append({
            "seed": seed,
            "acc": float(acc)
        })

    os.makedirs("resultados_finais", exist_ok=True)

    with open("resultados_finais/v4_7_top2_moe.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved -> resultados_finais/v4_7_top2_moe.json")


if __name__ == "__main__":
    run()