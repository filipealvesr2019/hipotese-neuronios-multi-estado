Aqui está a árvore completa do projeto:

```
neuronios quanticos/
├── .git/
├── .gitignore
├── README.md
├── README_EN.md
├── CONTEXT_PACKAGE.md
├── refactor_repo.py
├── Checklist de Validação da Arquitetura.md
├── diretrizes das docs.md
├── proximos passos da pesquisa.md
├── rinha de arquiteturas.md
│
├── arena/                          # Competição de arquiteturas
│   ├── CHALLENGERS.md
│   ├── README.md
│   ├── resultados/
│   │   └── README.md
│   ├── results/
│   │   └── v5_competicao_seed1/
│   └── v5_competicao/
│       ├── train.py
│       ├── arena/
│       │   ├── resultados/
│       │   └── results/
│       └── results/
│
├── datasets/                       # Datasets (MNIST, circles, moons, spirals)
│   ├── mnist/   (.gz)
│   ├── circles/
│   ├── moons/
│   └── spirals/
│
├── docs/                           # Documentação EN/PT-BR
│   ├── en-US/reports/  (conclusions, diary, roadmap, etc.)
│   └── pt-BR/relatorios/  (conclusões, diário, roadmap, etc.)
│
├── experimentos/                   # Código-fonte dos experimentos
│   ├── e1_mnist/           (código + __pycache__)
│   ├── v1_multistate/      (código + resultados JSON)
│   ├── v2_gating/          (código + resultados)
│   ├── v3_skip_connections/(código + resultados)
│   ├── v4_1_load_balancing/(código + resultados)
│   └── v4_sparse_routing/  (código + resultados)
│
├── resultados_finais/              # Resultados consolidados
│   ├── comparacoes/
│   ├── e1_mnist_economic/          (JSONs, CSV, REPORT.md)
│   ├── e1_mnist_matrix/            (JSONs, CSV, REPORT.md)
│   ├── e1_mnist_validation/        (JSONs, CSV, REPORT.md)
│   ├── graficos/
│   ├── tabelas/
│   ├── e1_mnist_seed1_result.json
│   └── output.txt
│
├── scripts/                        # Scripts auxiliares
│   ├── bateria_completa.py
│   ├── common.py
│   ├── download_mnist.py
│   ├── runner.py
│   ├── run_teste.bat
│   ├── sumario.py
│   └── test_imports.py
│
└── __pycache__/
```

**Resumo:** Projeto de pesquisa sobre arquiteturas neurais com **neurônios quânticos multi-estado**. Contém experimentos com 5 variações (v1 a v4), datasets sintéticos e MNIST, scripts de treino/validação, e resultados consolidados em `resultados_finais/`. A pasta `arena/` abriga uma competição entre arquiteturas. Documentação bilíngue (pt-BR/en-US).


Here is the complete structured listing with a 1-line description for each of the 32 important files:

---

### Root / Documentation Files

1. **`F:\neuronios quanticos\README.md`** — Portuguese-language project overview describing the Sparse Routing / MoE research, repository structure, current status (V4 ties MLP at ~50% FLOPs), and how to run experiments.

2. **`F:\neuronios quanticos\README_EN.md`** — English-language counterpart of README.md, also including a summary table of V4 vs MLP MNIST validation results (20 seeds, accuracy/FLOPs comparison).

3. **`F:\neuronios quanticos\CONTEXT_PACKAGE.md`** — Auto-generated context package (2026-06-08) documenting the full project vision, arena protocol rules, scoring system, architectural evolution (V1-V4.1), and experiment status.

4. **`F:\neuronios quanticos\refactor_repo.py`** — Repository restructuring script that creates the target directory hierarchy, generates skeleton markdown docs (diary, conclusions, refuted hypotheses), and moves legacy files into proper locations.

### Arena (Competition Framework)

5. **`F:\neuronios quanticos\arena\README.md`** — Defines the "Arena de Arquiteturas" competition protocol: fixed rules for comparing MLP, V4 economic, and new V5+ challengers under identical conditions with mandatory metrics (accuracy, FLOPs, entropies).

6. **`F:\neuronios quanticos\arena\v5_competicao\train.py`** — V5 challenger training script implementing simplified experts without an external gate, using a synthetic 784-feature MNIST-like dataset with `make_classification`.

7. **`F:\neuronios quanticos\arena\CHALLENGERS.md`** — Describes the baselines (MLP, V4 Sparse Top-1) and proposed challengers V5 (no external gate), V6 (Top-2 routing), plus outlines for V7–V9 with open research questions.

### Experimentos / Core Experiments

8. **`F:\neuronios quanticos\experimentos\e1_mnist\codigo\e1_mnist_v4.py`** — Main MNIST experiment script comparing Traditional MLP vs V4 Sparse Routing, including dataset loading, model definitions, FLOPs estimation, timing, and expert usage logging (625 lines).

9. **`F:\neuronios quanticos\experimentos\e1_mnist\README.md`** — Documentation for the E1 MNIST experiment, explaining the central question, how to run, preliminary single-seed results, and plans for an Accuracy/FLOPs matrix.

10. **`F:\neuronios quanticos\experimentos\v1_multistate\codigo\experimento_quantico.py`** — V1 MultiState experiment with a simple weighted-sum multi-state layer, dataset generation pipeline, and training loop comparing traditional MLP against multi-state neurons (original, now refuted hypothesis).

11. **`F:\neuronios quanticos\experimentos\v1_multistate\codigo\test1_XOR.py`** (and siblings `test2_paridade`, `test3_fronteiras`, etc.) — Battery of 12 progressive tests (XOR, parity, boundaries, compression, capacity, low-data, noise, specialization, ablation, info-per-param, distillation, correlation) comparing MLP vs MultiState across different challenges (all share the same `common.py` harness).

12. **`F:\neuronios quanticos\experimentos\v2_gating\codigo\v2_gating.py`** — V2 MultiStateLayer with input-dependent Softmax gating, implementing per-sample weight generation across 4 states with backward pass and temperature control.

13. **`F:\neuronios quanticos\experimentos\v2_gating\codigo\test13_v2_gating.py`** — Tests for V2 gating: 13A gate dominance analysis (which state wins for which inputs) and 13B ablation (removing states and measuring impact), plus a baseline comparison against traditional MLP on circles.

14. **`F:\neuronios quanticos\experimentos\v3_skip_connections\codigo\v3_skip_mlp_gate.py`** — V3 MultiStateLayer adding an MLP-based gate with ReLU hidden layer and a learned skip connection from input to output to remedy the training collapse observed in V2.

15. **`F:\neuronios quanticos\experimentos\v4_sparse_routing\codigo\v4_sparse_routing.py`** — V4 Sparse Routing layer implementing Top-1 hard routing via Straight-Through Estimator (STE), expert states, gate MLP, and skip connection, with entropy and FLOPs computation (302 lines).

16. **`F:\neuronios quanticos\experimentos\v4_1_load_balancing\codigo\v4_1_load_balancing.py`** — V4.1 extension adding an auxiliary load balancing loss (alpha_bal parameter) to encourage uniform expert utilization, attempting to mitigate mode collapse observed in V4.

### Scripts (Orchestration & Utilities)

17. **`F:\neuronios quanticos\scripts\common.py`** — Shared module providing core layer classes (LinearLayer, MultiStateLayer), model definitions (MLPTradicional, MLPMultiEstado), dataset generators, training/validation loops, and I/O utilities used by all V1-era tests.

18. **`F:\neuronios quanticos\scripts\runner.py`** — Sequential test runner that executes each of the 12 V1 tests individually (via subprocess), saves JSON results, and continues even if individual tests fail.

19. **`F:\neuronios quanticos\scripts\bateria_completa.py`** — Complete test battery (793 lines) implementing 12 deterministic experiments comparing MLP vs 4-state MultiState with 30 seeds each, including LinearLayer, MultiStateLayer, training loops, and result aggregation.

20. **`F:\neuronios quanticos\scripts\sumario.py`** — Summary script that loads `resultados.json`, computes aggregate statistics (means, stds, win counts) and prints a formatted comparative table of Traditional vs MultiState accuracies across 30 seeds.

21. **`F:\neuronios quanticos\scripts\download_mnist.py`** — Downloads the 4 MNIST IDX files from Google Storage to `datasets/mnist/`, verifying each file's SHA-256 hash and redownloading on mismatch.

### Resultados Finais (Final Results Reports)

22. **`F:\neuronios quanticos\resultados_finais\e1_mnist_economic\REPORT.md`** — Reports the V4 Economic MNIST matrix (hidden 64/128/96, states 2/4, gate 8/16, skip on/off), finding the best configuration (V4 128/s=2/g=8/no-skip at 93.92% accuracy) that nearly matches MLP 128.

23. **`F:\neuronios quanticos\resultados_finais\e1_mnist_validation\REPORT.md`** — Consolidated 20-seed validation comparing V4 s=2 vs MLP across seeds, including per-seed breakdowns, entropy regimes (COLAPSO/MODERADO/ALTO), and FLOPs comparison showing V4 at 250k vs MLP 235 at 483k.

24. **`F:\neuronios quanticos\resultados_finais\e1_mnist_matrix\REPORT.md`** — Accuracy/FLOPs matrix for V4 with 4 states across hidden sizes 64/128/256 versus MLP baselines, concluding MLP 64 is the most efficient point and V4 overhead was too high in this configuration.

### Research Planning & Validation Documents

25. **`F:\neuronios quanticos\diretrizes das docs.md`** — Documentation guidelines explaining how to organize code, prepare the environment (requirements.txt), structure repos, and write READMEs for third-party reproducibility.

26. **`F:\neuronios quanticos\proximos passos da pesquisa.md`** — Strategic next steps document outlining 5 directions: explore V4 economic tuning, generalize to other datasets (Fashion-MNIST), implement Arena V5-V9, create Accuracy/FLOPs curves, and run real inference benchmarks.

27. **`F:\neuronios quanticos\rinha de arquiteturas.md`** — In-depth analysis (669 lines) arguing that the project needs a formal architecture competition ("rinha") to avoid over-optimizing a single architecture, with detailed evolution critique and future directions.

28. **`F:\neuronios quanticos\Checklist de Validação da Arquitetura.md`** — Practical validation checklist (461 lines) defining 5 solid criteria (multi-dataset stability, efficiency, routing robustness, reproducibility, cost/benefit) for considering the architecture mature enough for real projects.

### Docs (Conclusions)

29. **`F:\neuronios quanticos\docs\pt-BR\conclusoes.md`** — Portuguese conclusions: the original MultiState hypothesis is refuted; the narrower sparse-routing hypothesis is partially confirmed; lists confirmed findings (states learn different reps, gate works, skip helps, Top-1 eliminates noise, V4 ties MLP at ~50% FLOPs) and caveats (no absolute accuracy gain, mode collapse persists).

30. **`F:\neuronios quanticos\docs\en-US\conclusions.md`** — English conclusions mirroring the PT-BR version, adding a maturity assessment (0% for original hypothesis, promising for V4) and MNIST update narrative (economic V4 partially recovered efficiency).

### V4 Stress Test & Validation

31. **`F:\neuronios quanticos\experimentos\v4_sparse_routing\codigo\v4_stress_test.py`** — 30-seed stress test of V4 Sparse vs Traditional MLP across 3 domains (Moons, Spirals, custom 20-features) with 300 epochs each, testing robustness of the V4 discovery across multiple domains.

32. **`F:\neuronios quanticos\experimentos\v4_sparse_routing\codigo\v4_30seeds_validation.py`** — Statistical validation script running V4 Sparse vs Traditional MLP on circles (2000 samples, 30 seeds, 300 epochs) to verify the technical tie and measure accuracy differences across seeds.