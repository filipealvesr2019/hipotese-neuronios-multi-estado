# Roadmap de Pesquisa — Roteamento Esparso e Mistura de Especialistas (V4+)

Este documento delineia os próximos passos lógicos para validar, escalar e aplicar a arquitetura de roteamento esparso descoberta na Fase 1 (V1-V4).

## Nível 1 — Validação Científica (Confirmar que não é coincidência)
O objetivo aqui é testar a arquitetura em datasets de visão computacional clássicos para garantir que ela escala para dimensionalidades reais e aprende representações estruturadas.

- **E1: MNIST (28x28, 10 classes)**
  Comparar MLP Tradicional vs V4 Sparse focando em: Accuracy, FLOPs, Tempo de treino e Tempo de inferência. Meta: Mesma accuracy, menos FLOPs.
  
- **E2: Fashion-MNIST**
  Testar se os especialistas aprendem categorias semânticas distintas (ex: um especialista para camisas, outro para calçados) sem programação explícita.

- **E3: CIFAR-10**
  Teste de generalização forte em imagens RGB complexas (aviões, gatos, cachorros).

## Nível 2 — Testar a Hipótese Real (Eficiência Computacional)
Provar de forma definitiva o ganho de "inteligência por operação".

- **E4: Curva Accuracy/FLOPs**
  Treinar modelos densos de larguras variadas (64, 128, 256 neurônios) e modelos V4 com quantidades variadas de especialistas (4, 8, 16). O objetivo é plotar o gráfico e provar que a V4 se mantém acima da curva tradicional (Mais accuracy por FLOP).
  
- **E8: Análise Introspectiva de Especialização**
  Registrar *quem é usado para quê*. Descobrir se E1 foca em bordas, E2 em curvas, E3 em ruído, etc.

## Nível 3 — Aproximar de LLMs (Aplicações em Transformers)
Substituir a camada *FeedForward (FFN)* de um Transformer pela camada *V4 Sparse Experts*.

- **E5: Tiny Transformer (1M a 10M parâmetros)**
  Avaliar se a perplexidade se mantém com custo reduzido.
  
- **E6: TinyStories**
  Treinar um Transformer Dense vs Transformer V4 no dataset TinyStories. Avaliar Loss, Perplexity e Tokens/s.
  
- **E7: GPT Mini (Ex: 10M params local)**
  Substituir apenas os blocos FFN e medir consumo de VRAM, tempo de inferência e qualidade gerativa.
  
- **Experimento "Sonho" (O Alvo Final):**
  Pegar um modelo fundacional pequeno (GPT-2 Small, Phi, TinyLlama), substituir as camadas densas por especialistas V4 e manter a mesma qualidade de texto gastando uma fração dos FLOPs.
