# Relatório de Treinamento: V8 Image-to-Code (Headers)

## O Experimento
Treinamos a arquitetura Multimodal (`V8_2_MultimodalArchitecture`) em 1000 imagens de Headers (divididas em 3 famílias) usando um Tokenizador Word-Level. O objetivo era observar se o Roteador do MoE (5 experts de tamanhos heterogêneos: 16, 32, 64, 128, 256) distribuiria as categorias sintáticas (HTML, CSS, Texto) naturalmente entre os experts.

## O Resultado Estatístico (10 Épocas)
A *Loss* caiu rapidamente de 7.22 para 0.67, indicando que o encanamento imagem-para-texto está funcionando perfeitamente e o modelo está aprendendo a sintaxe. 

No entanto, a Análise Estatística revelou um fenômeno clássico:

```text
==================================================
RELATÓRIO ESTATÍSTICO DE ESPECIALIZAÇÃO FUNCIONAL
==================================================

[Expert 0] Dominou 26.0 tokens únicos:
  -> 30.8% pertencem à categoria: HTML_ESTRUTURAL
  -> 38.5% pertencem à categoria: CSS_LAYOUT
  -> 30.8% pertencem à categoria: CONTEUDO_TEXTO

[Expert 1] Não dominou estatisticamente nenhuma categoria (Colapso)
[Expert 2] Não dominou estatisticamente nenhuma categoria (Colapso)
[Expert 3] Não dominou estatisticamente nenhuma categoria (Colapso)
[Expert 4] Não dominou estatisticamente nenhuma categoria (Colapso)
==================================================
```

## Conclusão Científica: O Colapso do Roteador (Routing Collapse)

O resultado acima é a prova empírica de um dos maiores desafios no design de MoEs: o **Colapso de Roteamento**.

1. **O que aconteceu:** O Expert 0 (provavelmente o menor, de 16 neurônios) inicializou com pesos marginalmente melhores. O Roteador mandou os primeiros tokens para ele. Como ele processou os tokens e recebeu os gradientes, ele ficou mais inteligente. O Roteador percebeu isso e começou a mandar *todos* os tokens para o Expert 0, matando de fome os Experts 1 a 4.
2. **O significado para a Teoria:** A heterogeneidade estrutural por si só não é forte o suficiente para forçar a segregação funcional *espontânea* se o roteador for ganancioso (Greedy Router).

## O Próximo Passo
Para forçar a especialização emergente, precisamos quebrar o monopólio do Expert 0. Na literatura de MoE (Switch Transformer, Mixtral), isso é feito de duas formas:
1. **Load Balancing Loss:** Uma penalidade matemática na *Loss Function* que pune o roteador se ele mandar todos os tokens para um único expert.
2. **Noisy Routing:** Adicionar ruído às probabilidades (Gate Probs) para forçar o roteador a testar outros experts nos estágios iniciais do treinamento.

A fundação do V8 está validada: o pipeline aprende. Agora o trabalho é calibrar o Roteador para induzir a especialização distribuída.
