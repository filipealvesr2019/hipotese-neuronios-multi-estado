<p align="center">
  <img src="https://img.shields.io/badge/V6.0-Stable-success?style=for-the-badge" alt="V6.0 Stable">
</p>

# V6.0 — Sparse Mixture of Experts Escalável com Roteamento Adaptativo e Pruning Emergente

Este projeto implementa uma arquitetura **Sparse Mixture of Experts (MoE)** com roteamento contextual adaptativo, especialistas heterogêneos e mecanismos emergentes de pruning automático construído inteiramente do zero em NumPy. O sistema demonstra escalabilidade em altíssima dimensionalidade e resolve o tradicional problema de "colapso de redundância", forçando redes a se especializarem em distribuições complexas.

## 🚀 Resultados Principais

O modelo foi submetido a testes de robustez variando de distribuições sintéticas locais (Spiral/XOR) até um teste de stress de 400 dimensões com 20 especialistas simultâneos.

| Dataset                  | Accuracy | Experts Usados | Top-K |
| ------------------------ | -------- | -------------- | ----- |
| **XOR**                  | 1.00     | 5              | 3     |
| **Spiral**               | 0.97     | 5              | 3     |
| **Gaussian**             | 0.91     | 5              | 3     |
| **High-Dim (400D/10-C)** | 0.45     | 20             | 4     |

### Métricas de Integridade Arquitetural (High-Dim)
* **ERI ≈ 0.10** (Expert Redundancy Index): Baixa redundância garantida matematicamente. As redes não copiam comportamento.
* **Gini ≈ 0.82** (Pruning Emergente): De 20 redes disponíveis, o gate decide usar unicamente as 4 mais eficientes e ignora as outras 16, isolando ruído.
* **RS ≈ 0.0008** (Routing Stability): O Roteador convence-se rápido e estabiliza a alocação a partir da 3ª época.

## 🧪 Experimentos e Reprodução

Execute os seguintes scripts para comprovar o desenvolvimento da arquitetura matemática da versão base até o modelo de escala V6.0:

```bash
# V5.8: Introdução de Top-K múltiplo e Adam Router
python experimentos/V5.8.py

# V5.9: Gating Contextual, Confidence Sharpening e Residual Routing
python experimentos/V5.9.py

# V6.0: Teste de Escala Massiva (20 experts) com ERI, RS e Gini Index
python experimentos/V6_0_Scaling_Robustness.py
```

## 📊 Insights e Descobertas

* **Especialização Emerge Automaticamente**: Roteadores lineares criam especialização ruidosa, roteadores contextuais alinham especialização com performance perfeita.
* **Roteador Converge para Subespaços Úteis**: Com t-SNE, é visível que o Gate separa os subespaços de decisão designando-os para redes de arquiteturas específicas (ex: redes maiores lidam com classes densas).
* **Escala não gera colapso, gera Pruning**: Oferecer capacidade redundante não confunde o sistema. O sistema descobre a folga computacional e zera as rotas de experts desnecessários.
* **Adaptive Sparse Computation**: O cálculo final prova que redes heterogêneas operam em perfeita sintonia residual, onde um Top-1 guia e o Top-K provê contexto leve.

## 🖼 Visualização do Roteamento (V5.9)

### 1. Separação Contextual em t-SNE
O Roteador aprende a criar uma fronteira de decisão nítida (Manifold) em seu embedding gerado internamente (`g2`), provando aprendizado espacial robusto antes do roteamento:
![Manifold do Roteador](graficos/tsne_gate_manifold.png)

### 2. Pruning Temporal
A distribuição inicial de uso é aleatória, mas a memória histórica força as probabilidades a convergirem para subconjuntos especialistas em menos de 10 épocas.
![Evolução do Uso](graficos/gate_usage_history.png)

### 3. Distribuição de Especialistas (Heatmap)
Os experts não apenas participam, mas dividem ativamente as classes, com redes generalistas atuando em conjunto com pequenas redes de função localizada.
![Matriz Classe/Expert](graficos/heatmap_class_expert.png)

## 🧩 Estudo de Causalidade (Freeze Study)

Um estudo rigoroso de "Freeze" (congelamento de gradientes) foi conduzido na versão `V6.2` para isolar de onde surge a inteligência da especialização:
1. **Experts Primeiro (Router Congelado)**: Os experts tentam aprender sozinhos e se tornam generalistas idênticos e redundantes. Quando o Roteador é ativado, o Gini Index despenca (0.33) provando o **Colapso da Especialização**.
2. **Router Primeiro (Experts Congelados)**: O Roteador roteia para experts estáticos aleatórios, criando uma geometria divisória forte desde o primeiro momento (Gini 0.47).
3. **Descoberta Final**: *A especialização não emana dos experts. Os experts são apenas argila computacional. A especialização é uma restrição imposta ativamente pelo Roteador dividindo a manifold. O roteador define a geometria do aprendizado.*

## 📁 Estrutura do Projeto

* `experimentos/` - Scripts contendo cada versão arquitetural (V5.8, V5.9, V6.0, V5_9_Visualizer, V6_1_Ablation).
* `resultados_finais/` - Arquivos JSON gerados documentando métricas puras de cada run.
* `graficos/` - Imagens t-SNE, de Uso e Heatmaps provando o roteamento modular.
* `docs/` - Manuscritos teóricos e diários de bordo de toda a jornada da pesquisa científica.
