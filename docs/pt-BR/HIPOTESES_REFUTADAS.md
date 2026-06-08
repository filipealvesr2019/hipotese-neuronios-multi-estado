# Hipóteses Refutadas

*Este arquivo existe para evitar a repetição de experimentos que já falharam. Cada hipótese aqui listada foi testada e descartada com dados concretos.*

---

## H001 — Soma de Estados Melhora o Desempenho

**Hipótese original:**
Um neurônio com N estados internos somados de forma ponderada aprende melhor do que um neurônio tradicional com N×parâmetros equivalentes.

**Experimento:** V1 MultiEstado (30 seeds, dataset Círculos).

**Resultado:** Falso.

**Dados:**
- Tradicional: 90.26%
- MultiEstado V1: 84.27%
- V1 Wins: 0/30 | Trad Wins: 29/30

**Motivo:** A soma ponderada de N estados lineares é matematicamente equivalente a uma única transformação linear. Os estados colapsam.

---

## H002 — Softmax Gate é Suficiente para Isolar Especialistas

**Hipótese:** Um gate com Softmax padrão consegue combinar estados internos sem vazar ruído entre eles.

**Experimento:** Ablação pós-treino na V3.

**Resultado:** Falso.

**Dados:** Remover o Estado 1 (L1_E1) **melhorou** a accuracy em +2.75pp. Um estado nocivo estava sendo ativado mesmo com peso baixo, injetando ruído na saída combinada.

**Motivo:** Softmax garante que a soma dos pesos = 1, mas nunca zera completamente nenhum peso. Sempre há "vazamento" para estados que não deveriam contribuir.

---

## H003 — Load Balancing Loss Resolve o Mode Collapse

**Hipótese:** Uma penalidade auxiliar que pune a concentração de tráfego em poucos especialistas (α × Σdist²) forçará a Layer 2 a diversificar o uso dos estados.

**Experimento:** V4.1 com α=5.0, 30 seeds.

**Resultado:** Falso.

**Dados:**
- V4.1 Sparse: 87.75% ± 1.16% (sem melhora vs V4: 87.67%)
- Entropia da Layer 2 permaneceu baixa em ~50% das seeds.

**Motivo provável:** Em redes rasas (2 camadas), a otimização prefere naturalmente consolidar representações abstratas em um único canal dominante. A penalidade cria ruído no gradiente mas não muda o equilíbrio fundamental do problema.

---

## H004 — Neurônio MultiEstado é Intrinsecamente Superior ao Neurônio Tradicional

**Hipótese inicial do projeto:** Um único neurônio com múltiplos estados internos armazena mais informação por parâmetro e supera redes tradicionais em desempenho absoluto.

**Resultado:** Não confirmado nos experimentos atuais.

**Dados:** Empate técnico em todos os domínios testados (Circles, Moons, Spirals, 20-Features), 30 seeds cada.

**Reformulação:** A hipótese útil não é "neurônio melhor", mas sim "especialistas compactos com roteamento esparso entregam a mesma inteligência gastando menos operações na inferência".
