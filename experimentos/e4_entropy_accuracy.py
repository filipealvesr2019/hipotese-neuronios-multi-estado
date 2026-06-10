# arquivo: experimentos/e4_entropy_accuracy.py
# rodar da raiz:
# PYTHONPATH=. python experimentos/e4_entropy_accuracy.py

import numpy as np
import json
import os
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# ==========================================
# Utils
# ==========================================

def set_seed(seed):
    np.random.seed(seed)

def normalize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    return (X - mean) / std

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=1, keepdims=True)

def compute_entropy(dist):
    dist = dist + 1e-10
    return float(-np.sum(dist * np.log2(dist)))

# ==========================================
# Hidden equivalente
# ==========================================

def get_ms_v4_hidden(
    target_params=242000,
    input_dim=784,
    output_dim=10,
    n_states=2
):
    h = 16

    while True:

        params = (
            (input_dim * h * n_states)
            + (h * output_dim * n_states)
            + (h * n_states)
            + (output_dim * n_states)
        )

        if params >= target_params:
            return h

        h += 1

# ==========================================
# V4 Simplificado
# ==========================================

class MLPMultiEstadoV4:

    def __init__(
        self,
        input_dim,
        h,
        output_dim,
        n_states=2,
        seed=0
    ):

        rng = np.random.RandomState(seed)

        self.n_states = n_states

        self.W1 = []
        self.b1 = []

        self.W2 = []
        self.b2 = []

        for _ in range(n_states):

            self.W1.append(
                rng.randn(input_dim, h) * 0.05
            )

            self.b1.append(
                np.zeros((1, h))
            )

            self.W2.append(
                rng.randn(h, output_dim) * 0.05
            )

            self.b2.append(
                np.zeros((1, output_dim))
            )

        self.gate1_W = rng.randn(input_dim, n_states) * 0.05
        self.gate2_W = rng.randn(h, n_states) * 0.05

    def relu(self, x):
        return np.maximum(0, x)

    def analyze_gate(self, X):

        g1 = softmax(X @ self.gate1_W)

        hidden_ref = self.relu(
            X @ self.W1[0] + self.b1[0]
        )

        g2 = softmax(hidden_ref @ self.gate2_W)

        return g1, g2

    def forward(self, X):

        g1 = softmax(X @ self.gate1_W)

        expert1 = np.argmax(g1, axis=1)

        hidden = []

        for i in range(len(X)):

            e = expert1[i]

            h = self.relu(
                X[i:i+1] @ self.W1[e]
                + self.b1[e]
            )

            hidden.append(h)

        hidden = np.vstack(hidden)

        g2 = softmax(hidden @ self.gate2_W)

        expert2 = np.argmax(g2, axis=1)

        outputs = []

        for i in range(len(X)):

            e = expert2[i]

            out = (
                hidden[i:i+1] @ self.W2[e]
                + self.b2[e]
            )

            outputs.append(out)

        return np.vstack(outputs)

# ==========================================
# Dataset
# ==========================================

def make_synthetic_mnist(
    seed=0,
    n_samples=2000,
    n_features=784,
    n_classes=10
):

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=50,
        n_redundant=0,
        n_classes=n_classes,
        random_state=seed
    )

    X = normalize(X)

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed
    )

# ==========================================
# Experimento
# ==========================================

def run_entropy_accuracy(
    n_seeds=10
):

    results = []

    print(
        "\n=== V4 Entropy vs Accuracy ===\n"
    )

    for seed in range(n_seeds):

        set_seed(seed)

        X_train, X_test, y_train, y_test = (
            make_synthetic_mnist(seed)
        )

        hidden = get_ms_v4_hidden(
            target_params=242000,
            input_dim=784,
            output_dim=10,
            n_states=2
        )

        model = MLPMultiEstadoV4(
            input_dim=784,
            h=hidden,
            output_dim=10,
            n_states=2,
            seed=seed
        )

        logits = model.forward(X_test)

        preds = np.argmax(
            logits,
            axis=1
        )

        acc = np.mean(
            preds == y_test
        )

        g1, g2 = model.analyze_gate(
            X_test
        )

        l1 = compute_entropy(
            np.mean(g1, axis=0)
        )

        l2 = compute_entropy(
            np.mean(g2, axis=0)
        )

        print(
            f"Seed {seed} | "
            f"Acc={acc:.4f} | "
            f"L1={l1:.4f} | "
            f"L2={l2:.4f}"
        )

        results.append({
            "seed": seed,
            "accuracy": float(acc),
            "l1_entropy": l1,
            "l2_entropy": l2
        })

    out_file = (
        "resultados_finais/"
        "v4_entropy_accuracy_results.json"
    )

    os.makedirs(
        "resultados_finais",
        exist_ok=True
    )

    with open(
        out_file,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            results,
            f,
            indent=2
        )

    print(
        f"\nResultados salvos em {out_file}"
    )

    return results

# ==========================================
# Main
# ==========================================

if __name__ == "__main__":
    run_entropy_accuracy(10)