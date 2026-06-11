# =========================
# V5.6 Competitive Experts MoE
# =========================
import numpy as np
import os
import json

# =========================
# Modelo
# =========================
class V56CompetitiveMoE:
    def __init__(self, input_dim, hidden, output_dim,
                 n_experts=5, top_k=2, lr=0.01,
                 seed=42, diversity_lambda=0.01):
        
        self.input_dim = input_dim
        self.hidden = hidden
        self.output_dim = output_dim
        self.n_experts = n_experts
        self.top_k = top_k
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
        logits = g @ self.gW2 + self.gb2
        probs = self.softmax(logits)
        
        topk_idx = np.argsort(-probs, axis=1)[:, :self.top_k]
        gate = np.zeros_like(probs)
        for i in range(B):
            gate[i, topk_idx[i]] = 1.0 / self.top_k
        
        out = sum(gate[:, i:i+1] * out_states[i] for i in range(self.n_experts))
        return out, gate, h_states, out_states
    
    # =========================
    # Loss
    # =========================
    def loss(self, logits, y):
        probs = self.softmax(logits)
        eps = 1e-9
        return -np.mean(np.log(probs[np.arange(len(y)), y] + eps))
    
    # =========================
    # Treino (train_step)
    # =========================
    def train_step(self, x, y):
        B = x.shape[0]
        
        # Forward
        out, gate, h_states, out_states = self.forward(x)
        probs_out = self.softmax(out)
        probs_out[np.arange(B), y] -= 1
        probs_out /= B
        
        # Experts update
        for i in range(self.n_experts):
            grad_out = probs_out * gate[:, i:i+1]
            dh = grad_out @ self.W2[i].T
            dh[h_states[i] <= 0] = 0
            
            dW2 = h_states[i].T @ grad_out
            db2 = np.sum(grad_out, axis=0)
            dW1 = x.T @ dh
            db1 = np.sum(dh, axis=0)
            
            # Regularização de diversidade
            if self.diversity_lambda > 0:
                div_grad = np.zeros_like(self.W2[i])
                for j in range(self.n_experts):
                    if j != i:
                        div_grad += 2 * (self.W2[i] - self.W2[j])
                dW2 += self.diversity_lambda * div_grad
            
            self.W2[i] -= self.lr * dW2
            self.b2[i] -= self.lr * db2
            self.W1[i] -= self.lr * dW1
            self.b1[i] -= self.lr * db1
        
        acc = np.mean(np.argmax(out, axis=1) == y)
        expert_perf = np.array([np.mean(np.argmax(out_states[i], axis=1) == y) for i in range(self.n_experts)])
        return acc, expert_perf

# =========================
# Datasets sintéticos
# =========================
def make_xor(n=200):
    X = np.random.randint(0, 2, (n, 2))
    y = X[:,0] ^ X[:,1]
    return X.astype(float), y

def make_gaussian(n=200, c=2, d=2, seed=0):
    rng = np.random.RandomState(seed)
    X = np.vstack([rng.randn(n//c, d) + i*3 for i in range(c)])
    y = np.hstack([np.ones(n//c)*i for i in range(c)])
    return X, y.astype(int)

def make_spiral(n=200, noise=0.2):
    X = np.zeros((n*2,2))
    y = np.zeros(n*2, dtype=int)
    for j in range(2):
        r = np.linspace(0.0,1,n)
        t = np.linspace(j*4,(j+1)*4,n) + np.random.rand(n)*noise
        X[j*n:(j+1)*n] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[j*n:(j+1)*n] = j
    return X, y

def make_mnist_like(n=500, d=28*28, c=10):
    X = np.random.randn(n,d)
    y = np.random.randint(0,c,n)
    return X, y

# =========================
# Funções utilitárias
# =========================
def accuracy(model, X, y):
    out, _, _, _ = model.forward(X)
    pred = np.argmax(out, axis=1)
    return np.mean(pred == y)

# =========================
# Rodar Experimentos
# =========================
def run():
    datasets = {
        "xor": make_xor,
        "gaussian": make_gaussian,
        "spiral": make_spiral,
        "mnist_like": make_mnist_like
    }
    
    results = {}
    
    for name, func in datasets.items():
        print(f"\n===== DATASET: {name} =====")
        X, y = func()
        
        model = V56CompetitiveMoE(
            input_dim=X.shape[1],
            hidden=32,
            output_dim=len(np.unique(y)),
            n_experts=5,
            top_k=2,
            lr=0.01,
            diversity_lambda=0.01
        )
        
        # Training
        for epoch in range(5):
            idx = np.random.permutation(len(X))
            for i in range(0, len(X), 32):
                batch = idx[i:i+32]
                acc, expert_perf = model.train_step(X[batch], y[batch])
        
        acc = accuracy(model, X, y)
        print(f"V5.6 MOE ACC: {acc:.4f}")
        results[name] = {"ACC": float(acc), "Expert Perf": expert_perf.tolist()}
    
    os.makedirs("resultados_finais", exist_ok=True)
    with open("resultados_finais/v5_6_competitive_experts.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved -> resultados_finais/v5_6_competitive_experts.json")

# =========================
# Main
# =========================
if __name__ == "__main__":
    run()