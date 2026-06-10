import sys
import os
import numpy as np
import json
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# ==============================
# Ajuste do PYTHONPATH
# ==============================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, PROJECT_ROOT)

V42_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, "experimentos/v4_2_stability/codigo"))
sys.path.insert(0, V42_PATH)

from v4_2_stability import MLPMultiEstadoV4, get_ms_v4_hidden, set_seed, normalize

# ==============================
# Função de Entropia
# ==============================
def compute_entropy(dist):
    dist = np.asarray(dist) + 1e-12
    return float(-np.sum(dist * np.log2(dist)))

# ==============================
# Dataset sintético MNIST-like
# ==============================
def make_synthetic_mnist(n_samples=2000, n_features=784, n_classes=10, seed=0):
    np.random.seed(seed)
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features, n_informative=50,
        n_classes=n_classes, random_state=seed
    )
    X, _, _ = normalize(X)
    return X, y

# ==============================
# Teste de Entropy vs Accuracy
# ==============================
def run_entropy_accuracy(n_seeds=10):
    results = []
    
    for seed in range(n_seeds):
        print(f"\n=== Seed {seed} ===")
        set_seed(seed)
        
        X, y = make_synthetic_mnist(seed=seed)
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        hidden = get_ms_v4_hidden(target_params=242000, input_dim=n_features, output_dim=n_classes, n_states=2)
        model = MLPMultiEstadoV4(input_dim=n_features, h=hidden, output_dim=n_classes, n_states=2, seed=seed)
        
        # Forward
        logits = model.forward(X)
        preds = np.argmax(logits, axis=1)
        acc = np.mean(preds == y)
        print(f"V4.2 Stability acc (forward only): {acc:.4f}")
        
        # Gate weights e entropia
        g1, g2 = model.analyze_gate(X)
        l1_ent = compute_entropy(np.mean(g1, axis=0))
        l2_ent = compute_entropy(np.mean(g2, axis=0))
        
        results.append({
            "seed": seed,
            "acc": float(acc),
            "l1_entropy": l1_ent,
            "l2_entropy": l2_ent
        })
    
    # Salvar resultados
    output_path = os.path.abspath(os.path.join(PROJECT_ROOT, "resultados_finais/v4_entropy_accuracy_results.json"))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResultados salvos em {output_path}")
    return results

# ==============================
# Rodar se chamado diretamente
# ==============================
if __name__ == "__main__":
    print("=== V4 Entropy vs Accuracy Test — MNIST-like Synthetic ===")
    run_entropy_accuracy(n_seeds=10)