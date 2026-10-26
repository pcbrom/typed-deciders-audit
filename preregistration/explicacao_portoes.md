# Frente explicativa, portões pré-registrados (final)

Congelado em 2026-10-21 com aprovação do autor, pelo commit marcado `prereg-explicacao-v1` neste repositório. Gasto com o Jev autorizado pelo autor.

## Modelos e versões

- **Jev:** as versões `jev-1.13.0` e `jev-1.14.0`, sempre pelo identificador fixo, nunca pelos apelidos `jev-latest` ou `jev-preview`. Todas as análises do Jev rodam nas duas versões. A comparação entre versões (diferenças pareadas de $W_i$ e $E_i$ e McNemar exato sobre o argmax da entrada inteira) é relatada sem portão.
- **Laya:** os checkpoints multilingual (mmBERT-base) e typed-decisions (ModernBERT-large), revisão de pesos 55cf4c4, em pt-PT e en-US, o que dá quatro casos. O multilingual ajustado em pt-PT entra só como substituto do Jev na validação de R1 (arredondamento e censura), sem ser caso de G5.

## Ordem das chamadas e regra de parada do Jev

- **Ordem:** os itens de relato de cada idioma são embaralhados uma vez com `SeedSequence([20261021, 50])`, antes de qualquer chamada. As chamadas seguem essa ordem item a item, com as coalizões de um item completas antes do próximo, alternando as duas versões por item.
- **Registro:** cada resposta grava `usage.input_tokens`.
- **Erros:** os erros 429 e 529 são repetidos com espera exponencial. O primeiro erro de outro tipo interrompe o braço e é registrado com o código.
- **Créditos:** se os créditos acabarem, valem os itens com jogo completo nas duas versões até a interrupção. A parada não depende de nenhum resultado, e o número efetivo de itens é relatado.
- **Projeção:** o custo real é a soma de `input_tokens` vezes US\$0,042 por milhão. A projeção do restante multiplica as chamadas que faltam pelo número médio de tokens medido. As duas são relatadas.

Commit do autor antes do G0. Nenhum limiar muda depois do commit. Toda estatística é calculada na metade de relato (itens 1000 a 1999). Hiperparâmetros e temperaturas vêm da metade de calibração. Sementes fixas: 20261021 para sorteios e 0, 1 e 2 para treinos.

## Definições comuns

- Unidades: palavras quando $n\le8$, senão 8 trechos contíguos de tamanho quase igual (R1, R3). Em R2, palavras para $9\le n\le12$.
- $y(S)=\operatorname{logit}p_j(x_S)$ com $j$ a opção decidida na entrada inteira, e $y_{jk^*}(S)=\log p_j-\log p_{k^*}$, com $k^*$ a segunda opção na entrada inteira.
- Jogo exato: todas as $2^d$ coalizões. Möbius $m_T=\sum_{U\subseteq T}(-1)^{|T|-|U|}v(U)$. $F_i=m_i$, $W_i=v(N)-v(N\setminus\{i\})$ e $E_i$ é o Shapley exato.
- Evento de cobertura por item: com a semente, sorteia-se uma unidade (ou uma coalizão retida) por item. O evento vale 1 se o valor exato cai no intervalo central de 90%. Os $k$ acertos em $n$ itens dão a posteriori $\mathrm{Beta}(k+1,n-k+1)$, a regra da Beta do autor. $L$ é o quantil de 5% e $\hat c=k/n$.
- PIT aleatorizado: para uma observação no intervalo $[a,b]$, $U\sim\mathcal U(F(a),F(b))$. O evento de cobertura é $0{,}05\le U\le0{,}95$.
- Percentil de deleção $u_i$: área da curva $y(N)-y(N\setminus\pi_{1..k})$, $k=1..d$, na ordem $\pi$ do método. A estatística é o percentil dessa área entre todas as $d!$ ordens do item, com empates em posto médio.

## G0, estabilidade do Jev (não eliminatório)

100 itens de relato por idioma, 3 repetições cada, em cada versão, total de 1.200 chamadas. Para `jev-1.13.0`, passa com argmax igual ao congelado em ao menos 98% e $|\Delta p|\le0{,}01$ em ao menos 95% das probabilidades. Para `jev-1.14.0`, que não tem respostas congeladas, passa com argmax igual entre as três repetições em ao menos 98% dos itens e $|\Delta p|\le0{,}01$ entre repetições em ao menos 95% das probabilidades. Se falhar, todo o braço Jev usa respostas novas da entrada inteira, nunca misturadas às congeladas, e a variância entre repetições entra na verossimilhança como $\sigma_{\text{api}}$. Também registra tokens por chamada. A fração de coalizões censuradas é lida nos 20 primeiros itens da enumeração, sem chamadas extras.

## G1, ordem 2 basta?

$R^2_{\le2}$ é a fração da variância de $v$ sobre as $2^d$ coalizões (pesos iguais) explicada pelos termos de Möbius de ordem até 2. A estatística é a mediana entre itens. Ela é medida em dois conjuntos, cada um com seu critério.

- Classificação por par: itens com $d\ge5$ em trechos, em cada caso do Laya. Mediana de ao menos 0,6 em todos os casos, segue. De 0,4 a 0,6 em algum caso, segue com ressalva no texto daquele caso. Abaixo de 0,4 em algum caso, (2) é relatado como aproximação insuficiente naquele caso.
- R2: os próprios 390 itens de R2 (202 pt e 188 en), unidades por palavra, nos dois casos de R2. Mediana de ao menos 0,6 nos dois, R2 segue. De 0,4 a 0,6 em algum, segue com ressalva. Abaixo de 0,4 em algum, R2 sai do artigo.

## G2, cobertura

- R1 (Laya arredondado a duas casas, pt multilingual ajustado e en multilingual): evento de cobertura de $W_i$ e, separadamente, de $E_i$ exatos sem arredondamento. Passa com $L\ge0{,}85$ e $\hat c\le0{,}97$ para ambos, nos dois casos. Exige-se também $L\ge0{,}85$ no estrato duplamente censurado, nos dois casos e para ambos. Uma unidade é duplamente censurada quando $p_j(x_N)$ e $p_j(x_{N\setminus\{i\}})$ arredondados ficam em 1,00. O evento do estrato sorteia, em cada item que tem ao menos uma dessas unidades, uma delas.
- R1 (Jev, em cada versão separadamente): PIT aleatorizado com validação cruzada em 5 partes, uma coalizão retida por item. Passa com $L\ge0{,}85$ e KS de uniformidade dos PIT com $p\ge0{,}01$ no conjunto e também só nas coalizões não censuradas, com o sorteio refeito nesse subconjunto. Só é lido se R1 no Laya passou. Suposição declarada no texto: nas caudas censuradas o PIT tem pouco poder, e a validade ali se apoia no Laya arredondado como substituto das caudas do Jev, o que não dá para verificar a partir dos dados do Jev.
- R2 (Laya, $N_{\text{fit}}=256$, ajuste sem tamanhos 1 e $d-1$): evento de cobertura da previsão de $v(S)$ numa coalizão retida e de $W_i$ exato. Passa com $L\ge0{,}85$ nos dois. Relato por tamanho de coalizão e com $N_{\text{fit}}=128$ como sensibilidade, sem decidir.
- Por faixa de $p_0$ (quintis da calibração), regra de sucessão de Laplace por faixa, a do autor, apenas relatada.
- Concorrentes medidos com o mesmo evento: BayesSHAP de Slack et al. e KernelSHAP com variância de Covert e Lee.
- Leitura: se R1 no Laya falha, (1) fica descritivo e o Jev é relatado com limites de um lado. Se só R2 falha, R2 sai do artigo.

## G3, verdade plantada

- Aprendizado: acurácia do modelo plantado no teste plantado com $L\ge0{,}90$, por tipo e semente.
- Assinatura no jogo exato do modelo plantado. Único: $m_A\ge1$ e $|\sum_k m_{Ak}|\le0{,}5\,m_A$. Redundante: $m_{A_1},m_{A_2}\ge1$ e $m_{A_1A_2}\le-0{,}5\min(m_{A_1},m_{A_2})$. Sinérgico: $|m_B|,|m_C|\le0{,}5$ e $m_{BC}\ge1$. Unidades em logit. Exige-se ao menos 90% dos itens de teste com a assinatura do tipo plantado, por tipo. Abaixo disso, a alegação de verdade plantada sai do artigo para aquele tipo, e a fração observada é relatada como limite do protocolo.
- Medida: F1 macro por item entre os rótulos único, redundante, sinérgico e nenhum, sobre saídas arredondadas a duas casas. Métodos: nosso Gibbs, interação de Shapley sobre o jogo arredondado, HEDGE e sinal de $H_i$. Teto: Möbius exato sem arredondamento.
- Passa com F1 de ao menos 0,80 e ao menos 0,10 acima do melhor concorrente, com intervalo bootstrap de itens de 95% da diferença acima de zero. F1 de pelo menos 0,80 sem vantagem: (2) é relatado como equivalente ao estado da arte sob censura. Abaixo de 0,80: resultado negativo.

## G4, cabeça aditiva

- Acurácia: diferença pareada (aditiva menos original) nas 3 sementes, com intervalo de 95% por bootstrap de itens. ECE com 15 faixas após temperatura.
- Competitiva se o limite inferior da diferença de acurácia for maior que −2 p.p. e o intervalo de 95% da diferença pareada de ECE, por bootstrap de itens, ficar inteiro entre −0,01 e 0,01. Entre −2 e −5 p.p.: relatada como custo medido e o recuo com resíduo penalizado roda. Abaixo de −5 p.p.: relatada como resultado negativo, com o recuo.
- Portão da regra da Beta (do autor, $q=0{,}05$, alvo 0,95): decisões por 1.000 e promessa quebrada nas 1.000 partições, apenas relatados.
- Fidelidade em itens com $d\ge5$: Spearman médio entre $e_{w,j}-e_{w,k^*}$ e o Shapley exato de $y_{jk^*}$ no mesmo modelo. Nulo de 10.000 permutações dentro do item. Referência: o mesmo Spearman para oclusão e SLALOM na cabeça original contra o Shapley exato dela.
- Descrita como autoexplicação por palavra só se o Spearman médio passar o percentil 99 do nulo, não ficar mais de 0,10 abaixo da melhor referência e a fração mediana da soma fora das palavras for de até 0,2. Senão, é descrita como aditiva nas posições.

## G5, camada hierárquica

- Resposta: $u_i$ da ordem pelo Shapley (exato no Laya, 100 sorteios da posteriori no Jev). Oito ajustes, um por caso (4 do Laya e 4 do Jev, duas versões por dois idiomas), NUTS com 4 cadeias e $\hat R\le1{,}01$.
- No Jev, $b$ é estimado só no estrato não censurado ($p_0<0{,}995$ arredondado), cerca de 272 itens pt e 220 en. Os itens censurados ficam fora desses dois ajustes e são contados.
- Sétimo ajuste declarado: modelo plantado, semente 0, sobre os itens de relato pt mais os itens de teste plantados, mesma especificação. Serve só ao critério das intenções plantadas e não entra na contagem dos 8 casos.
- O sinal está decidido num caso se $P(b>0)\ge0{,}95$ ou $P(b>0)\le0{,}05$.
- Fica no artigo se o sinal estiver decidido em ao menos 6 dos 8 casos (pelo menos dois terços, como os 4 de 6 do desenho com uma versão do Jev), ou se, no modelo plantado, as intenções $X$, $X_2$ e $X_3$ ficarem entre os 10% maiores $\alpha_c$. Senão, é substituída por uma tabela de Spearman($p_0$, $u_i$) com bootstrap de itens.

## Validação 4, opções (sem eliminação)

- $j$ e $k^*$ são fixados pela configuração original, na entrada inteira. Os distratores trocados nunca são $j$ nem $k^*$. Se numa configuração alternativa a decisão na entrada inteira deixa de ser $j$ ou a segunda opção deixa de ser $k^*$, o item sai do cálculo de $D$ para aquela configuração, e a fração desses itens é relatada por configuração.
- Empate do argmax: o conjunto máximo $A$ reúne as unidades a até 0,01 logit do máximo de $W^{jk^*}$ (média posterior no Jev). Há mudança quando $A_{\text{orig}}\cap A_{\text{alt}}=\varnothing$.
- $D$ = fração de itens com mudança entre a configuração original e cada alternativa, menos a mesma fração no placebo de ordem permutada. Intervalo de 95% por bootstrap de itens.
- $D\ge0{,}10$ com limite inferior acima de zero: resultado central. $0{,}02\le D<0{,}10$ com limite inferior acima de zero: dependência detectável, relatada como secundária. Intervalo contendo zero ou $D<0{,}02$: invariância aproximada.

## Validação 2, deleção e inserção (sem eliminação)

- Média de $u_i$ por método e operador (deleção e troca por palavra do unigrama de treino sem alta PMI com as opções), em MoRF e em inserção.
- Métodos ordenados só se o Kendall entre os rankings dos dois operadores for de ao menos 0,6. Senão, relato por operador, sem ranking.

## Orçamento congelado

- Jev: 908.068 chamadas nas duas versões (454.034 por versão: 600 do G0, 251.834 da enumeração do relato, 50.400 do segundo operador em 200 itens por idioma e 151.200 das opções sem empacotar). A 384 tokens por chamada, que é teto porque as coalizões são mais curtas, são 348,7 milhões de tokens e US\$14,65 a US\$0,042 por milhão, cerca de 10,3 h a 24,6 chamadas por segundo. O custo efetivo segue a regra de parada acima.
- R2 roda em dois casos, pt multilingual e en multilingual, que usam o mesmo checkpoint multilingual: 280.064 passadas pt e 273.408 en, total de 553.472.
- Laya: 2,16 milhões de passadas, somando 630.000 dos jogos exatos dos 4 casos e do substituto de censura, 553.472 de R2, 252.000 do segundo operador nos 2 casos multilingual, 151.200 das opções, 307.200 do plantio e 264.000 da cabeça aditiva. Mais 12 treinos de cerca de 1 min.
- O tempo por passada em GPU (1 ms no multilingual, 3 ms no root) não foi medido. É estimativa, e o primeiro bloco de GPU mede o tempo antes de seguir. Se passar de 2 ms no multilingual, os blocos são replanejados, sem mudar nenhum limiar.
