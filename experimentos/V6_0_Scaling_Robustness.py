import numpy as np
import os
import json
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from time import time

# =========================
# V6.0 — SCALING & ROBUSTNESS (20 Experts, High-Dim, Gini, ERI, RS)
# =========================
class V60_MoE:
    def __init__(self, input_dim, hidden_sizes, output_dim,
                 top_k=3, lr=0.01, gate_lr=0.01, ema=0.9, diversity_weight=0.001, balance_weight=0.1, flop_weight=1e-6):
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.output_dim = output_dim
        self.n_experts = len(hidden_sizes)
        self.top_k = top_k
        self.lr = lr
        self.gate_lr = gate_lr
        self.ema = ema
        self.diversity_weight = diversity_weight
        self.balance_weight = balance_weight
        self.flop_weight = flop_weight

        rng = np.random.RandomState(42)

        # Experts
        self.W1 = [rng.randn(input_dim, h) * 0.1 for h in hidden_sizes]
        self.b1 = [np.zeros(h) for h in hidden_sizes]
        self.W2 = [rng.randn(h, output_dim) * 0.1 for h in hidden_sizes]
        self.b2 = [np.zeros(output_dim) for _ in range(self.n_experts)]

        # Contextual Gate (3 layers)
        self.gW1 = rng.randn(input_dim, 64) * 0.1
        self.gb1 = np.zeros(64)
        self.gW2 = rng.randn(64, 32) * 0.1
        self.gb2 = np.zeros(32)
        self.gW3 = rng.randn(32, self.n_experts) * 0.1
        self.gb3 = np.zeros(self.n_experts)

        # Adam
        self.m_gW1, self.v_gW1 = np.zeros_like(self.gW1), np.zeros_like(self.gW1)
        self.m_gb1, self.v_gb1 = np.zeros_like(self.gb1), np.zeros_like(self.gb1)
        self.m_gW2, self.v_gW2 = np.zeros_like(self.gW2), np.zeros_like(self.gW2)
        self.m_gb2, self.v_gb2 = np.zeros_like(self.gb2), np.zeros_like(self.gb2)
        self.m_gW3, self.v_gW3 = np.zeros_like(self.gW3), np.zeros_like(self.gW3)
        self.m_gb3, self.v_gb3 = np.zeros_like(self.gb3), np.zeros_like(self.gb3)
        self.beta1, self.beta2, self.eps = 0.9, 0.999, 1e-8
        self.t = 0

        self.expert_perf = np.zeros(self.n_experts)
        self.expert_memory = np.ones((self.n_experts, output_dim))

    def softmax(self, x):
        x = x - np.max(x, axis=1, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=1, keepdims=True)

    def forward(self, X, temperature=1.0):
        B = X.shape[0]
        h_list, out_list = [], []
        for i in range(self.n_experts):
            h = np.maximum(0, X @ self.W1[i] + self.b1[i])
            out = h @ self.W2[i] + self.b2[i]
            h_list.append(h)
            out_list.append(out)

        g1 = np.maximum(0, X @ self.gW1 + self.gb1)
        g2 = np.maximum(0, g1 @ self.gW2 + self.gb2)
        logits = g2 @ self.gW3 + self.gb3
        
        logits_scaled = logits / temperature
        gate_probs = self.softmax(logits_scaled)

        topk_idx = np.argsort(gate_probs, axis=1)[:, -self.top_k:]
        gate = np.zeros_like(gate_probs)
        for b in range(B):
            idx = topk_idx[b]
            gate[b, idx[-1]] = 1.0
            for k in range(2, self.top_k + 1):
                gate[b, idx[-k]] = 0.5
            gate[b] /= np.sum(gate[b])

        out = sum(gate[:, i:i+1] * out_list[i] for i in range(self.n_experts))
        return out, gate, out_list, h_list, g1, g2, gate_probs, logits_scaled

    def loss(self, logits, y):
        probs = self.softmax(logits)
        return -np.mean(np.log(probs[np.arange(len(y)), y] + 1e-9))

    def train_step(self, X, y, temperature=1.0):
        B = X.shape[0]
        logits, gate, out_list, h_list, g1, g2, gate_probs, logits_scaled = self.forward(X, temperature)
        probs_out = self.softmax(logits)
        probs_out[np.arange(B), y] -= 1
        probs_out /= B

        expert_acc = np.zeros(self.n_experts)
        expert_correct = np.zeros((B, self.n_experts))

        for i in range(self.n_experts):
            grad_out = probs_out * gate[:, i:i+1]
            dh = grad_out @ self.W2[i].T
            dh[h_list[i] <= 0] = 0
            dW2 = h_list[i].T @ grad_out
            db2 = np.sum(grad_out, axis=0)
            dW1 = X.T @ dh
            db1 = np.sum(dh, axis=0)

            dW2_div = np.zeros_like(self.W2[i])
            for j in range(self.n_experts):
                if i != j:
                    min_h = min(self.W2[i].shape[0], self.W2[j].shape[0])
                    wi = self.W2[i][:min_h]
                    wj = self.W2[j][:min_h]
                    norm_wi = np.linalg.norm(wi) + 1e-9
                    norm_wj = np.linalg.norm(wj) + 1e-9
                    dW2_div[:min_h] += wj / (norm_wi * norm_wj)

            self.W2[i] -= self.lr * (dW2 + self.diversity_weight * dW2_div)
            self.b2[i] -= self.lr * db2
            self.W1[i] -= self.lr * dW1
            self.b1[i] -= self.lr * db1

            pred = np.argmax(out_list[i], axis=1)
            expert_correct[:, i] = (pred == y).astype(float)
            expert_acc[i] = np.mean(expert_correct[:, i])
            
            for c in range(self.output_dim):
                self.expert_memory[i, c] += np.sum(expert_correct[:, i] * (y == c))

        self.expert_perf = self.ema * self.expert_perf + (1 - self.ema) * expert_acc

        sum_correct = np.sum(expert_correct, axis=1, keepdims=True)
        local_target = np.where(sum_correct > 0, expert_correct / np.where(sum_correct > 0, sum_correct, 1), 1.0 / self.n_experts)
        
        memory_prior = np.zeros((B, self.n_experts))
        for b in range(B):
            memory_prior[b] = self.expert_memory[:, y[b]]
        memory_sum = np.sum(memory_prior, axis=1, keepdims=True)
        memory_prior /= np.where(memory_sum > 0, memory_sum, 1)
        
        target_probs = 0.5 * local_target + 0.5 * memory_prior
        
        usage = np.mean(gate_probs, axis=0)
        load_balance_grad = (usage - 1.0 / self.n_experts)
        
        d_logits_scaled = (gate_probs - target_probs) / B + self.balance_weight * load_balance_grad
        d_logits = d_logits_scaled / temperature
        
        d_gW3 = g2.T @ d_logits
        d_gb3 = np.sum(d_logits, axis=0)
        d_g2 = d_logits @ self.gW3.T
        d_g2[g2 <= 0] = 0
        d_gW2 = g1.T @ d_g2
        d_gb2 = np.sum(d_g2, axis=0)
        d_g1 = d_g2 @ self.gW2.T
        d_g1[g1 <= 0] = 0
        d_gW1 = X.T @ d_g1
        d_gb1 = np.sum(d_g1, axis=0)
        
        self.t += 1
        def adam_update(param, grad, m, v):
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            param -= self.gate_lr * m_hat / (np.sqrt(v_hat) + self.eps)
            return param, m, v

        self.gW3, self.m_gW3, self.v_gW3 = adam_update(self.gW3, d_gW3, self.m_gW3, self.v_gW3)
        self.gb3, self.m_gb3, self.v_gb3 = adam_update(self.gb3, d_gb3, self.m_gb3, self.v_gb3)
        self.gW2, self.m_gW2, self.v_gW2 = adam_update(self.gW2, d_gW2, self.m_gW2, self.v_gW2)
        self.gb2, self.m_gb2, self.v_gb2 = adam_update(self.gb2, d_gb2, self.m_gb2, self.v_gb2)
        self.gW1, self.m_gW1, self.v_gW1 = adam_update(self.gW1, d_gW1, self.m_gW1, self.v_gW1)
        self.gb1, self.m_gb1, self.v_gb1 = adam_update(self.gb1, d_gb1, self.m_gb1, self.v_gb1)

        return self.loss(logits, y), expert_acc

def gini_index(x):
    diffsum = 0
    for i in x:
        for j in x:
            diffsum += abs(i - j)
    return diffsum / (2 * len(x)**2 * np.mean(x) + 1e-9)

def compute_eri(out_list):
    n = len(out_list)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j: 
                sim_matrix[i, j] = 1.0
                continue
            pred_i = np.argmax(out_list[i], axis=1)
            pred_j = np.argmax(out_list[j], axis=1)
            sim_matrix[i, j] = np.mean(pred_i == pred_j)
    return (np.sum(sim_matrix) - n) / (n * (n - 1))

def run():
    print("Gerando Dataset High-Dim Complexo (CIFAR/Features-like)...")
    X, y = make_classification(n_samples=5000, n_features=400, n_informative=100, n_redundant=50, n_classes=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 20 Experts: 5 blocks of different sizes
    hidden_sizes = [32]*4 + [64]*4 + [128]*4 + [256]*4 + [512]*4
    print(f"Iniciando V6.0 - Scaled MoE com {len(hidden_sizes)} Experts")
    
    model = V60_MoE(input_dim=X.shape[1],
                    hidden_sizes=hidden_sizes,
                    output_dim=10,
                    top_k=4,  # Increase Top-K since we have 20 experts
                    lr=0.01,
                    gate_lr=0.01,
                    diversity_weight=0.001,
                    balance_weight=0.1)

    epochs = 30
    batch_size = 128
    
    rs_history = []
    prev_gate_probs_val = None
    
    for epoch in range(epochs):
        temperature = max(0.7, 3.0 * (0.85 ** epoch))
        idx = np.random.permutation(len(X_train))
        
        t0 = time()
        for i in range(0, len(X_train), batch_size):
            batch = idx[i:i+batch_size]
            if len(batch) == 0: continue
            model.train_step(X_train[batch], y_train[batch], temperature)
            
        # Avaliação e Métricas de Robustez no Val
        logits_val, gate_val, out_list_val, _, _, _, gate_probs_val, _ = model.forward(X_test, temperature=0.7)
        val_pred = np.argmax(logits_val, axis=1)
        val_acc = np.mean(val_pred == y_test)
        
        # RS (Routing Stability)
        if prev_gate_probs_val is not None:
            rs = np.mean(np.abs(gate_probs_val - prev_gate_probs_val))
            rs_history.append(rs)
        else:
            rs = 1.0
            rs_history.append(rs)
        prev_gate_probs_val = gate_probs_val.copy()
        
        # Gini Index (Expert Utilization)
        usage_val = np.mean(gate_val, axis=0)
        gini_val = gini_index(usage_val)
        
        # ERI (Expert Redundancy Index)
        eri_val = compute_eri(out_list_val)
        
        print(f"Epoch {epoch+1:02d} | Val ACC: {val_acc:.4f} | ERI: {eri_val:.4f} | RS: {rs:.4f} | Gini: {gini_val:.4f} | Time: {time()-t0:.1f}s")
        
    print("\nTreinamento Finalizado!")
    print(f"Uso Final dos Experts:\n{np.round(usage_val, 3)}")

if __name__ == "__main__":
    run()
