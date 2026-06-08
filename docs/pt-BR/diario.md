# Diário de Pesquisa

## 2026-06-08

### Experimento 1 — Verificação da Hipótese Base (Teste 12)
**Hipótese:** Estados internos independentes podem existir sem colapsar em uma única transformação linear.
**Resultado:** Confirmado.
**Observação:** A correlação entre estados ficou próxima de zero (~0.01). Os estados não viram cópias um do outro.

---

### Experimento 2 — V1 MultiEstado Simples
**Hipótese:** Somar os estados internos ponderados já é suficiente para superar um MLP tradicional.
**Resultado:** Refutado.
**Observação:** A arquitetura V1 perdeu para o MLP em 29 de 30 seeds (84.27% vs 90.26%). A soma ponderada de estados colapsa matematicamente em uma transformação linear única.

---

### Experimento 3 — V2 Gate Input-Dependente
**Hipótese:** Um gate com Softmax treinável consegue especializar o uso dos estados.
**Resultado:** Parcialmente confirmado.
**Observação:** Na Layer 1, o gate roteou entradas diferentes para estados diferentes (KL ≈ 0.098, ~13x mais especialização que V1). Na Layer 2, o sinal do gate desapareceu (KL ≈ 0.003). A performance caiu para 62.1%.

---

### Experimento 4 — V3 Gate MLP + Skip Connections
**Hipótese:** Substituir o gate Linear por um MLP e adicionar Skip Connections resolve a perda de sinal nas camadas profundas.
**Resultado:** Confirmado.
**Observação:** A performance subiu de 62.1% para 85.0%, próxima do baseline (86.25%). No Teste B de ablação, remover o Estado 1 (L1_E1) *melhorou* o desempenho em +2.75pp — prova de que o Softmax vaza ruído de estados ruins.

---

### Experimento 5 — Teste D: Dinâmica do Gate (Entropia ao Longo do Treino)
**Hipótese:** O colapso da Layer 2 é um processo gradual, não imediato.
**Resultado:** Confirmado.
**Observação:** A Layer 1 manteve entropia saudável (1.94 no final). A Layer 2 começou com entropia = 2.0 (uniforme) e caiu progressivamente até 1.02 na época 300, com o Estado 4 absorvendo 80% do tráfego.

---

### Experimento 6 — V4 Sparse Routing (Top-1 + STE)
**Hipótese:** Roteamento Top-1 Hard elimina o vazamento de ruído do Softmax.
**Resultado:** Confirmado.
**Observação:** L1_E1 removido → impacto = 0.00pp (o estado ruim foi perfeitamente ignorado). A V4 atingiu 85.5% vs 85.0% do Tradicional em seed única. O colapso da Layer 2 ficou ainda mais rápido (100% no Estado 4 na época 90).

---

### Experimento 7 — Validação com 30 Seeds (V4 vs Tradicional)
**Hipótese:** O resultado de 1 seed pode ser ruído estatístico.
**Resultado:** Empate técnico confirmado.
**Dados:**
- Tradicional: 87.84% ± 1.36%
- V4 Sparse: 87.67% ± 1.15%
- V4 Wins: 10/30 | Trad Wins: 14/30 | Empates: 6/30

**Conclusão:** Mesma accuracy, menor custo computacional (~50% dos FLOPs na inferência).

---

### Experimento 8 — V4.1 Load Balancing Loss
**Hipótese:** Uma penalidade de balanceamento (α × Σdist²) força a Layer 2 a usar mais de um estado.
**Resultado:** Nenhuma melhora.
**Dados:**
- V4.1 Sparse: 87.75% ± 1.16%
**Conclusão:** Em redes rasas (2 camadas), a otimização preguiçosa prefere consolidar abstrações em um único especialista. A penalidade não gerou ganho de accuracy.

---

### Experimento 9 — Stress Test (Moons, Spirals, 20-Features)
**Hipótese:** O empate técnico observado em Círculos se mantém em domínios mais complexos.
**Resultado:** Confirmado.
**Dados:**
- Moons: Trad 99.72% vs V4 99.79% | V4 Wins: 8/30
- Spirals: Trad 61.60% vs V4 61.79% | V4 Wins: 15/30
- 20-Features: Trad 90.08% vs V4 89.88% | V4 Wins: 11/30

**Conclusão da Fase 1:** A arquitetura de especialistas compactos com roteamento esparso reproduz de forma robusta e consistente o desempenho de MLPs tradicionais em múltiplos domínios, gastando aproximadamente metade do custo computacional de inferência.
