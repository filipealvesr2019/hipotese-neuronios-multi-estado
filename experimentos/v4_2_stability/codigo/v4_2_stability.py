import numpy as np
from scripts.bateria_completa import LinearLayer, MLPTradicional, make_circles, normalize, train_test_split, set_seed, softmax_crossentropy

# =========================
# MultiStateLayer V4.2 Stability
# =========================
class MultiStateLayerV42:
    def __init__(self, input_dim, output_dim, n_states=4, seed=42, temperature=1.5, gate_hidden=8, alpha_balance=0.05, skip_scale=0.2):
        self.n_states = n_states
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.temperature = temperature
        self.gate_hidden = gate_hidden
        self.alpha_balance = alpha_balance
        self.skip_scale = skip_scale
        
        rng = np.random.RandomState(seed)
        rng2 = np.random.RandomState(seed + 100)
        rng3 = np.random.RandomState(seed + 200)

        # Estados
        self.W = [rng.randn(input_dim, output_dim)*0.1 for _ in range(n_states)]
        self.b = [np.zeros(output_dim) for _ in range(n_states)]

        # Gate MLP
        self.gate_W1 = rng2.randn(input_dim, gate_hidden)*0.1
        self.gate_b1 = np.zeros(gate_hidden)
        self.gate_W2 = rng2.randn(gate_hidden, n_states)*0.1
        self.gate_b2 = np.zeros(n_states)

        # Skip
        self.skip_W = rng3.randn(input_dim, output_dim)*0.1
        self.skip_b = np.zeros(output_dim)

    def forward(self, x):
        B = x.shape[0]
        states_out = [x@w + b for w,b in zip(self.W, self.b)]
        
        # Gate MLP
        h = np.maximum(0, x@self.gate_W1 + self.gate_b1)
        logits = h@self.gate_W2 + self.gate_b2
        logits = logits / self.temperature
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        
        # Top-2 hard routing
        top2 = np.argsort(probs, axis=1)[:, -2:]
        gw = np.zeros_like(probs)
        for i in range(B):
            gw[i, top2[i]] = probs[i, top2[i]]
        gw = gw / gw.sum(axis=1, keepdims=True)
        
        # Saída
        out = sum(gw[:,i:i+1]*states_out[i] for i in range(self.n_states))
        out += self.skip_scale*(x@self.skip_W + self.skip_b)
        return out

# =========================
# Teste simples
# =========================
def run_v42_stability():
    set_seed(42)
    X, y = make_circles(n_samples=2000, noise=0.1, seed=42)
    X, _, _ = normalize(X)
    Xtr, Xva, ytr, yva = train_test_split(X, y)
    
    # Modelo baseline
    mlp = MLPTradicional(2, 64, 2, 42)
    for ep in range(50):
        idx = np.random.permutation(Xtr.shape[0])
        for s in range(0, Xtr.shape[0], 64):
            ids = idx[s:s+64]
            Xb, yb = Xtr[ids], ytr[ids]
            logits = mlp.forward(Xb)
            _, grad = softmax_crossentropy(logits, yb)
            mlp.backward(grad, 0.01)
    print("MLP baseline acc:", np.mean(mlp.predict(Xva) == yva))

    # Modelo V4.2 Stability
    hidden = 64
    mv = MultiStateLayerV42(2, 2, n_states=4, temperature=1.5, gate_hidden=8, alpha_balance=0.05, skip_scale=0.2)
    
    # Treino simplificado
    for ep in range(50):
        idx = np.random.permutation(Xtr.shape[0])
        for s in range(0, Xtr.shape[0], 64):
            ids = idx[s:s+64]
            Xb, yb = Xtr[ids], ytr[ids]
            out = mv.forward(Xb)
            # Gradient update simplificado (placeholder)
            # Aqui você pode adaptar do V4 original
    print("V4.2 Stability acc (forward only):", np.mean(np.argmax(mv.forward(Xva), axis=1) == yva))

if __name__ == "__main__":
    run_v42_stability()