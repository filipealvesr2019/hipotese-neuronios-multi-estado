import numpy as np
import os
import json

# =========================
# V5.5 Expert Diversity MoE
# =========================
class V55ExpertDiversityMoE:
    def __init__(self, input_dim, hidden, output_dim,
                 n_experts=5, lr=0.01, diversity_lambda=0.01,
                 seed=42):
        self.input_dim = input_dim
        self.hidden = hidden
        self.output_dim = output_dim
        self.n_experts = n_experts
        self.lr = lr
        self.diversity_lambda = diversity_lambda

        rng = np.random.RandomState(seed)
        # Experts
        self.W1 = [rng.randn(input_dim, hidden) * 0.1 for _ in range(n_experts)]
        self.b1 = [np.zeros(hidden) for _ in range(n_experts)]
        self.W2 = [rng.randn(hidden, output_dim) * 0.1 for _ in range(n_experts)]
        self.b2 = [np.zeros(output_dim) for _ in range(n_experts)]

        # Gate
        self.gW1 = rng.randn(input_dim, 16) * 0.1
        self.gb1 = np.zeros(16)
        self.gW2 = rng.randn(16, n_experts) * 0.1
        self.gb2 = np.zeros(n_experts)

    # Softmax
    def softmax(self, x):
        x = x - np.max(x, axis=1, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=1, keepdims=True)

    # Forward
    def forward(self, x):
        B = x.shape[0]
        h_states = []
        out_states = []

        for i in range(self.n_experts):
            h = np.maximum(0, x @ self.W1[i] + self.b1[i])
            out = h @ self.W2[i] + self.b2[i]
            h_states.append(h)
            out_states.append(out)

        g = np.maximum(0, x @ self.gW1 + self.gb1)
        logits_gate = g @ self.gW2 + self.gb2
        probs_gate = self.softmax(logits_gate)
        top1 = np.argmax(probs_gate, axis=1)

        gate = np.zeros_like(probs_gate)
        for i in range(B):
            gate[i, top1[i]] = 0.9
            for j in range(self.n_experts):
                if j != top1[i]:
                    gate[i, j] += 0.1 / (self.n_experts - 1)

        gate = gate / gate.sum(axis=1, keepdims=True)
        out = sum(gate[:, i:i+1] * out_states[i] for i in range(self.n_experts))
        return out, probs_gate, h_states, out_states

    # Loss
    def loss(self, logits, y):
        probs = self.softmax(logits)
        eps = 1e-9
        return -np.mean(np.log(probs[np.arange(len(y)), y] + eps))

    # Training step
    def train_step(self, x, y):
        B = x.shape[0]
        logits, gate_probs, h_states, out_states = self.forward(x)
        probs_out = self.softmax(logits)
        probs_out[np.arange(B), y] -= 1
        probs_out /= B

        for i in range(self.n_experts):
            grad_out = probs_out * gate_probs[:, i:i+1]

            # Gradient for W2
            dW2 = h_states[i].T @ grad_out
            db2 = np.sum(grad_out, axis=0)

            # Diversity penalty
            if self.diversity_lambda > 0:
                penalty = sum([(self.W2[i] - self.W2[j]) for j in range(self.n_experts) if j != i])
                dW2 += self.diversity_lambda * penalty

            # Gradient for W1
            dh = grad_out @ self.W2[i].T
            dh[h_states[i] <= 0] = 0
            dW1 = x.T @ dh
            db1 = np.sum(dh, axis=0)

            # Update
            self.W2[i] -= self.lr * dW2
            self.b2[i] -= self.lr * db2
            self.W1[i] -= self.lr * dW1
            self.b1[i] -= self.lr * db1

        return self.loss(logits, y)

# =========================
# Dataset simples
# =========================
def make_data(name, n=200, seed=0):
    rng = np.random.RandomState(seed)
    if name == "xor":
        X = np.array([[0,0],[0,1],[1,0],[1,1]])
        y = np.array([0,1,1,0])
    elif name == "gaussian":
        X = rng.randn(n,2)
        y = (X[:,0]+X[:,1]>0).astype(int)
    elif name == "spiral":
        t = np.linspace(0, 4*np.pi, n)
        r = t
        X = np.c_[r*np.cos(t), r*np.sin(t)]
        y = (t>2*np.pi).astype(int)
    else:
        X = rng.randn(n,2)
        y = rng.randint(0,2,n)
    return X, y

def accuracy(model, X, y):
    logits, _, _, _ = model.forward(X)
    pred = np.argmax(logits, axis=1)
    return np.mean(pred == y)

# =========================
# Run experiment
# =========================
def run():
    results = {}
    datasets = ["xor","gaussian","spiral"]
    for ds in datasets:
        print(f"\n===== DATASET: {ds} =====")
        X, y = make_data(ds)
        model = V55ExpertDiversityMoE(input_dim=X.shape[1],
                                      hidden=16,
                                      output_dim=len(np.unique(y)),
                                      n_experts=5,
                                      lr=0.01,
                                      diversity_lambda=0.01,
                                      seed=42)
        for epoch in range(50):
            idx = np.random.permutation(len(X))
            for i in range(0,len(X),32):
                batch = idx[i:i+32]
                model.train_step(X[batch], y[batch])
        acc = accuracy(model,X,y)
        print(f"V5.5 MOE ACC: {acc:.4f}")
        results[ds] = acc

    os.makedirs("resultados_finais",exist_ok=True)
    with open("resultados_finais/v5_5_expert_diversity.json","w") as f:
        json.dump(results,f,indent=2)
    print("\nSaved -> resultados_finais/v5_5_expert_diversity.json")

if __name__=="__main__":
    run()