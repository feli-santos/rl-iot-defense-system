# Preparação para a Defesa — Material de Estudo

**Felipe Augusto Oliveira dos Santos** · FEEC/UNICAMP
Orientador: Prof. Dr. Denis Fantinato
Banca: Prof. Dr. Denis Fantinato (UNICAMP) · Prof. Dr. Alexandre da Silva Simões (UNESP) · Prof. Dr. João Kleinschmidt (UFABC)
Campinas — 31 de agosto de 2026

> Resumo condensado da dissertação + banco de perguntas e respostas.
> Fonte: `tex/main.pdf` (versão entregue à banca). Todos os números conferidos
> contra `tex/generated/numbers.tex` e os capítulos 3–5.

---

# PARTE 0 — CONTROLE DE TEMPO (leia antes de ensaiar)

O ensaio de 09:13→10:03 levou **50 minutos**, com o Ato I sozinho consumindo
14. O deck foi reescrito para **37 minutos**, deixando ~3 min de folga.

**O relógio é o instrumento.** Anote o horário de início e cheque nestes
quatro pontos. Se você estiver atrasado num checkpoint, corte no ato
**seguinte** — nunca acelere o slide do cruzamento (23).

| Checkpoint | Ao terminar o slide | Relógio alvo | Se passar disso |
|---|---|---|---|
| Fim da abertura | 3 · Agenda | **02:00** | você falou demais sobre si — corte a bio |
| Fim do Ato I | 8 · POMDP | **09:40** | pule o slide 25 (sweeps) mais tarde |
| Fim do Ato III | 20 · Protocolo | **23:30** | pule o slide 27 (recall) mais tarde |
| Fim dos resultados | 27 · Recall | **32:40** | vá direto ao 29; pule limitações em detalhe |
| Fim | 31 · Obrigado | **37:00** | — |

**Slides marcados como cortáveis, em ordem de corte:**

1. **27 — No Dependence on Detector Recall** → `[OPCIONAL]`. Responde uma
   objeção sofisticada que talvez ninguém faça. Cortar não enfraquece o
   slide 26.
2. **25 — Robustness sweeps** → apresente **só o painel da esquerda**
   (dificuldade) e siga.

**Nunca corte:** 10 (objetivos), 23 (cruzamento), 28 (limitações), 29
(conclusões).

---

## O que mudou depois do primeiro ensaio

| Feedback | Onde foi resolvido |
|---|---|
| [2] "não detalhar aluno especial; passar mais rápido" | Slide 2, nota com teto de 35 s e regra explícita |
| [3] "contract?" | Jargão eliminado do deck inteiro → "reward function" |
| [4] "não dá pra perceber onde está o que é falado" | Slide 4 com selos ①–⑥; notas citam o número antes de descrever |
| [4] "como a defesa deve acontecer?" | Nova linha de fecho no slide 4: a defesa vive no **gateway** |
| [4] "por que usar RL?" | Novo cartão "Why RL here?" no slide 7 |
| [5] "muita informação verbal, sem descritivo" | Regra nº 2 do `notes.js`: nunca narrar o que não está no slide |
| [6] "incluir um slide de objetivos" | **Novo slide 10** — objetivo geral + 5 específicos |
| [7] "por que os modelos foram escolhidos?" | Novo cartão "Why these three?" (inclui exclusão de SAC/TD3) |
| [7] "DQN usa replay, não chegam os mesmos dados" | Linha "Controlled:" corrigida → orçamento de **interação**, e o reuso é a regra sob teste |
| [9] "explicar como um classificador é usado" | Nova esteira `fluxo → RF → tabela → age` no slide 9 |
| [13] "34 classes não ficou claro" | Slide 13: "+ benigno = 34 rótulos"; slide 14 idem |
| [13] "ilustrar o alfa com um exemplo" | Slide 17: "em α=0,4, ~4 de cada 10 linhas vêm do estágio vizinho" |
| [14] "o que é PC?" | Slide 15: cartão definindo componentes principais |
| [17] "falou SPARSE, slide diz OUTCOME" | Rótulos duplos: `OUTCOME · sparse` e `COUPLED · shaped` |
| [24] "é possível selecionar alguns resultados" | Slide 25 marcado como cortável pela metade |
| [25] "10 classes? aumentou?" | Slide 26: "as mesmas 10 reservadas antes" |
| [25] "Prevention Rate não ficou claro" | Slide 26: definição formal de `P_prev` |
| [26] "daria para pular" | Slide 27 marcado `[OPCIONAL]` na nota |
| [28] "footprint poderia pular" | Reduzido a uma cláusula; números vão para o **backup B7** |
| Arthur: "adiantou metodologia nos slides 7 e 13" | Notas dos slides 7, 8 e 14 proíbem citar α/hiperparâmetros |

---

# PARTE I — RESUMO PARA ESTUDO

## 0. O "elevator pitch" em três durações

**30 segundos.**
Defesa autônoma de redes IoT com aprendizado por reforço profundo. A pergunta não é
"RL funciona?", é **"quando o RL genuinamente ganha de um classificador?"**. A resposta:
só quando a observação é ambígua. Eu transformei essa ambiguidade num botão controlado
(α), e mostrei que existe um ponto de cruzamento — empate em α=0, vitória estatística a
partir de α=0,4.

**2 minutos.**
Uma intrusão é uma campanha em cinco estágios (kill chain). O defensor tem cinco ações de
força crescente. Se o defensor **visse** o estágio, o problema colapsaria em classificação
e um RandomForest bem ajustado bastaria. Meu compromisso central de modelagem é que ele
**nunca vê**: estágios adjacentes emitem características sobrepostas com probabilidade α,
o que torna a tarefa um POMDP genuíno. Construí um ambiente Gymnasium fechado com atacante
reativo sobre linhas reais do CICIoT2023, treinei PPO/A2C/DQN com janela de 5 observações
(290 dimensões), e comparei contra um RandomForest ajustado acoplado à regra de ação
(RF-Acting), políticas triviais e um oráculo. Três resultados: (i) o cruzamento do aliasing;
(ii) a vantagem sobrevive à remoção do sinal moldado; (iii) em dez classes retidas, o RL
previne mais em todas, sem dependência do recall do detector.

**5 minutos** — use a estrutura do Ato I→V dos slides.

---

## 1. Motivação e problema (Cap. 1)

**Números de escala:** 19,8 bi dispositivos IoT (2025) → 40,6 bi (2034), Statista/Taylor.
Prejuízo com cibercrime: **USD 12,2 trilhões/ano até 2031** (Cybersecurity Ventures/Braue).

**Três falhas estruturais da segurança tradicional em IoT:**

1. **Dispositivos restritos** — sem orçamento computacional/energético para agentes de
   segurança embarcados (Chen et al. 2021).
2. **Heterogeneidade extrema** — protocolos e SOs díspares; política uniforme é
   impraticável (Al-Garadi et al. 2020).
3. **Defesas estáticas** — IDS por assinatura falha em zero-day; IDS por anomalia gera
   falsos positivos demais em tráfego heterogêneo (Khraisat 2019; DeMedeiros 2023).

**A formulação do problema (a frase que a banca vai testar):**

> "Se cada fluxo revela individualmente a intenção do atacante, o problema de defesa
> colapsa em classificação por fluxo, e um classificador bem ajustado é a solução natural.
> O aprendizado por reforço só se justifica quando a decisão é genuinamente **sequencial**
> e **parcialmente observada**."

O trabalho **remove deliberadamente** uma simplificação de formulações anteriores:
amostragem de características independente por passo — sob a qual um classificador sem
memória é ótimo por construção e o RL não tem razão estrutural para vencer.

---

## 2. Fundamentação (Cap. 2)

### 2.1 Kill chain como frame de decisão

Cinco estágios + mapeamento de ação recomendada (segue IoTWarden, Alam et al. 2024):

| Estágio | Descrição | Ação recomendada | Táticas MITRE ATT&CK |
|---|---|---|---|
| BENIGN | tráfego de rotina | OBSERVE | — |
| RECON | varredura, fingerprinting | LOG | Reconnaissance (TA0043), Discovery (TA0007) |
| ACCESS | foothold: brute force, injeção | RESTRICT | Initial Access (TA0001), Credential Access (TA0006) |
| MANEUVER | movimentação lateral, botnet | BLOCK | Lateral Movement (TA0008), Defense Evasion (TA0005) |
| IMPACT | DDoS, exfiltração | ISOLATE | Impact (TA0040) |

Propriedade-chave: **o custo de oportunidade do defensor cresce monotonicamente com o
estágio**. Interromper cedo é barato; tarde, caro.

### 2.2 RL: MDP → POMDP

MDP = quíntupla (S, A, P, R, γ). Funções de valor V^π e Q^π; equação de otimalidade de
Bellman.

**A transição conceitual crítica:** o simulador carrega um estado latente markoviano
(s_t^stage, h_t), mas o defensor nunca observa s_t^stage — isso é informação privilegiada
acessível **apenas ao oráculo**. Logo o defensor resolve um POMDP (S, A, O, P, Z, R, γ),
com Z = núcleo de observação.

A política POMDP-ótima age sobre uma **crença** b_t(s) = Pr(s_t = s | o_{1:t}, a_{1:t−1}),
atualizada pelo filtro de Bayes (Eq. 2.1). **Esta dissertação aproxima essa crença por uma
janela de tamanho fixo** — a política treinada é uma aproximação condicionada ao histórico,
não uma política MDP condicionada ao estado. Frase da dissertação: *"esta lacuna de
percepção (não a complexidade de controle) é a dificuldade central da tarefa, e o oráculo,
que lê o estágio latente diretamente, mede o valor de fechá-la."*

### 2.3 Os três algoritmos

- **DQN** (Mnih 2015): off-policy, baseado em valor. Replay buffer + rede alvo fixa.
  Perda = erro quadrático de Bellman.
- **A2C** (Mnih 2016): on-policy, ator-crítico síncrono. Vantagem estimada pelo resíduo TD
  de um passo. Perda = ator + crítico + bônus de entropia.
- **PPO** (Schulman 2017): on-policy, ator-crítico com **surrogate recortado** e GAE.
  Impede atualizações destrutivas.

Justificativa da seleção: cobrem as famílias model-free primárias; recomendados pelo guia
do SB3 para ações discretas; validados pelas surveys de Kheddar (2025) e Yang (2026) como
os mais avaliados em resposta a intrusão. SAC/TD3/Dreamer excluídos porque o menu discreto
de 5 ações não os exige.

### 2.4 Posicionamento na literatura (Tabela 2.1)

| Sistema | Ambiente | Dataset | Head-to-head? | Estresse de obs. parcial? |
|---|---|---|---|---|
| IoTWarden (2024) | trigger-action sintético | logs sintéticos | Não | Não |
| RESTRAIN (2025) | trigger-action multiagente | logs sintéticos | Não | Não |
| RL-IoTIDS (2026) | classificação de fluxo | CICIoT2023 | Sim (só F1) | Não |
| HoneyIoT (2023) | honeypot simulado | atacante sintético | Não | Não |
| **Este trabalho** | linhas reais CICIoT2023 | CICIoT2023 | **Sim** (oráculo, RF-Acting, 3 triviais, 10 seeds, CIs bootstrap) | **Sim (α)** |

**O contraste mais importante é com RL-IoTIDS (Odeh, 2026):** também é DRL sobre
CICIoT2023, mas usa RL para **rotular** cada fluxo (a recompensa premia classificação
correta por fluxo) — logo a política aprendida é fundamentalmente um detector one-shot,
não um controlador de kill chain. É exatamente a formulação que esta dissertação argumenta
ser insuficiente sob observabilidade parcial.

**IoTWarden é inspiração, não baseline.** Ambiente diferente (trigger-action sintético),
espaço de ação diferente → comparação numérica direta não seria metodologicamente sólida.
Por isso a afirmação numérica principal é ancorada no **oráculo in-domain**, nunca contra
um número externo.

---

## 3. Metodologia (Cap. 3)

### 3.1 Modelo de ameaça e fronteiras

**Capacidades:** atacante de camada de rede, capaz de executar todos os 33 tipos de ataque
do CICIoT2023.

**Fronteira declarada (isto é escopo, não afirmação sobre todos os atacantes):**

- Estritamente sequencial, nunca pula estágio, nunca recua autonomamente.
- **Uma exceção:** onset BENIGN→ACCESS (credencial pré-adquirida), sem salto para
  MANEUVER/IMPACT.
- **Não** é co-treinado adversarialmente.
- **Não** captura cenários APT com kill chains não-lineares.
- Variante com salto de estágio existe, mas apenas como ablação de estresse.

### 3.2 Dataset e projeção

**CICIoT2023** (Neto et al. 2023): 105 dispositivos IoT físicos, 33 tipos de ataque em 7
categorias, ataques executados **por** dispositivos IoT comprometidos contra outros.
~46,7 milhões de linhas brutas.

Cada linha = registro de fluxo pré-agregado com 46 características estatísticas.
**Sem timestamps por pacote e sem chave de sessão recuperável** (isto é crucial para a
limitação de coerência de sessão).

**Funil de seleção à prova de vazamento (ajustado só no treino):**

- 46 → 42: remove variância zero (DHCP, IRC, SMTP, Telnet)
- 42 → 35: remove variância < 0,01 (ARP, DNS, IPv, LLC, SSH, cwr_flag, ece_flag)
- 35 → 29: remove correlacionadas |Pearson| > 0,95 (Magnitue, Number, Radius, Srate, Std, Weight)

**As 29 características por grupo temático:**

| Grupo | # | Exemplos |
|---|---|---|
| Temporização e taxa | 5 | flow_duration, Duration, Rate, Drate, IAT |
| Cabeçalho e tamanho | 6 | Header_Length, Tot sum, Min, Max, AVG, Tot size |
| Flags TCP | 10 | fin/syn/rst/psh/ack_flag_number, ack/syn/fin/urg/rst_count |
| Indicadores de protocolo | 6 | Protocol Type, HTTP, HTTPS, TCP, UDP, ICMP |
| Momentos de distribuição | 2 | Covariance, Variance |

Isso é o que torna o aliasing um estressor **genuíno**: linhas de IMPACT separam nitidamente
por contagem de flags e taxa, enquanto BENIGN e RECON permanecem próximas.

**Projeção ψ_kc:** {1,…,34} → {0,…,4}, determinística. Cada estágio induz uma distribuição
empírica p_data(x|s).

**Distribuição por estágio (linhas):** BENIGN 100.000 (limitado); RECON 50.746;
ACCESS 36.950; MANEUVER 60.605; IMPACT 193.936. Total após rebalanceamento: n = 442.237.

Duas assimetrias que a banca pode questionar, ambas **consequências da projeção, não
acidentes de amostragem**:

1. BENIGN é a maior classe porque é o comportamento de referência que o defensor não pode
   perturbar — mantido em alto volume em vez de subamostrado.
2. IMPACT é o estágio mais populoso porque a maioria das classes de ataque do CICIoT2023
   são variantes de DoS/DDoS/Mirai de alto volume, e o mapeamento colapsa todas no estágio
   terminal.

**Splits (treino / val / teste):** BENIGN 70.000/10.000/20.000; RECON 35.522/5.075/10.149;
ACCESS 25.865/3.695/7.390; MANEUVER 42.424/6.060/12.121; IMPACT 135.755/19.394/38.787.
Pool de treino: **235.324 linhas**.

**Dez classes retidas** (≥2 por estágio não-benigno), disjunção garantida por asserções na
suíte de testes.

**Sobreposição real (Fig. 3.4):** PCA com PC1 = 19,6% e PC2 = 12,1% da variância. Estágios
adjacentes se sobrepõem substancialmente — mesmo com informação completa, não são
linearmente separáveis.

### 3.3 Atacante reativo

**Onset autônomo (só a partir de BENIGN):** p_onset = 0,35 para RECON; p_onset,access = 0,10
direto para ACCESS; caso contrário dormente. **Sem salto** — é essa escolha que torna
always-BLOCK uma política cara mas batível, em vez de trivialmente dominante.

**Regra de força com sinal:** d_t = a_t − rec(s_t)

| Regime | Condição | Consequência |
|---|---|---|
| Subdimensionado | d ≤ −1 | avança com p_up^eff (senão segura) |
| Proporcional | d = 0 | recua um estágio com p_down = 0,90 (ISOLATE: 0,98) |
| Superdimensionado | d ≥ +1 | apenas segura — e paga o custo de disponibilidade |

**Escalada acoplada à proximidade:** λ = s/4 ∈ {0; 0,25; 0,5; 0,75; 1,0}

    p_up^eff = p_up · (σ_min + (1 − σ_min)·λ),   σ_min = 0,4

Modela adversário cujo momentum se compõe com o progresso, e **remove o ponto de operação
arbitrário** que um orçamento fixo de recursos imporia.

**Prevenção:** manter abaixo de IMPACT durante todo o horizonte (100 passos) → bônus +50.
**Empate na fronteira de impacto é resolvido a favor do atacante**, então prevenção exige
genuinamente ficar à frente da cadeia, não empatar com ela.

**Persistência evasiva (opcional):** atacante que acabou de sentir força defensiva em
RECON/ACCESS endurece contra a **próxima** tentativa de expulsão. A resposta correta ainda
segura a linha (o atacante não avança) — a mecânica nunca premia mis-forcing, só torna o
atacante mais difícil de remover.

### 3.4 Detector supervisionado

RandomForest ajustado por **grade de 54 células** sobre macro-F1 de validação. Ótimo
interior: 200 árvores, profundidade máx. 20, mín. 2 amostras/folha, pesos balanceados.
**1,7M nós, 181 MB.**

Macro-F1 no teste balanceado retido: **0,924**. Por classe: BENIGN 0,924 / RECON 0,870 /
ACCESS 0,869 / MANEUVER 0,956 / IMPACT 0,998.

Confusão (normalizada por linha): BENIGN 94,7% / RECON 82,2% (**13,2% → ACCESS**) /
ACCESS 92,9% / MANEUVER 92,0% / IMPACT 99,8%.

**Argumento de justiça:** macro-F1 de validação plano em 0,927 ± 0,005 sobre toda a grade
(amplitude 0,016) → o classificador **não está sub-ajustado**; a lacuna residual para o
oráculo é propriedade da observabilidade parcial da tarefa, não de um classificador mal
configurado.

**Recall OOD (Tabela 4.2) — espectro graduado, não dicotomia:**

| Classe | Estágio | Recall | Categoria |
|---|---|---|---|
| DNS_Spoofing | MANEUVER | 0,201 | quase-cego |
| VulnerabilityScan | RECON | 0,224 | quase-cego |
| SqlInjection | ACCESS | 0,247 | quase-cego |
| DDoS-SlowLoris | IMPACT | 0,657 | parcial |
| Recon-OSScan | RECON | 0,705 | parcial |
| XSS | ACCESS | 0,961 | quase-perfeito |
| DDoS-HTTP_Flood | IMPACT | 0,993 | quase-perfeito |
| DDoS-ACK_Fragmentation | IMPACT | 0,993 | quase-perfeito |
| Mirai-udpplain | MANEUVER | 0,996 | quase-perfeito |
| DoS-SYN_Flood | IMPACT | 0,998 | quase-perfeito |

**Nota de proveniência importante:** esses recalls são medidos **dentro do ambiente de
benchmark**, onde o RF classifica a linha mais recente de uma janela com aliasing e
coerência de sessão — exatamente a condição sob a qual o RF-Acting é pontuado.

### 3.5 O ambiente (POMDP)

**Observação:** o_t = φ(x_{t−4:t}) = [x_{t−4}, Δx_{t−4}, …, x_t, Δx_t] ∈ ℝ^290,
onde 290 = 5 × 29 × 2.

**Núcleo de observação (mistura de dois componentes):**

    Z(o_t | s_t, a_{t−1}) = (1 − α)·p_data(x_t | s_t) + α·p̃_data(x_t | s_t)

com adjacência limitada (clamp) nos extremos BENIGN e IMPACT.

**Amostragem coerente com sessão:** dentro de um dwell num estágio, o motor sorteia uma
sequência **contígua e não-repetida** de linhas daquele estágio; a classe de ataque age como
identificador de sessão proxy.

**Ciclo de vida do episódio:**

- Começa em BENIGN, t = 0.
- **Clamp de piso:** qualquer IMPACT alcançado antes do passo t = 20 é rebaixado a MANEUVER
  (impacto real exige tempo de execução). Sob a dinâmica estritamente sequencial isso
  raramente dispara, mas é mantido como piso.
- **Contrato terminal primário:** alcançar IMPACT **não** encerra o episódio imediatamente —
  o agente tem uma última decisão. (`impact_is_terminal=True` só em estudo de caso
  secundário de mis-especificação.)
- Truncamento em 100 passos → se o atacante está abaixo de IMPACT, episódio **prevenido**.

### 3.6 Recompensa

**Contrato primário: OUTCOME (esparso).** Só custo de ação + contabilidade terminal +
bônus de prevenção.

**Contrato COUPLED (só ablação):** reinstala os cinco componentes de moldagem condicionados
ao estágio.

Os **seis componentes** por passo (usados integralmente só no coupled):

1. Custo de ação: −κ·C_action(a_t), sempre aplicado
2. Bônus de proporcionalidade: +5 se |a_t − rec(s_t)| ≤ 1
3. Penalidade de desproporção: −5 se |a_t − rec(s_t)| ≥ 2
4. Bônus benigno-passivo: +10 em BENIGN com OBSERVE/LOG
5. Penalidade de exagero em benigno: −50 em BENIGN com RESTRICT+
6. Guardrails assimétricos: −100 (block em benigno) / −50 (block em recon)

"Independente" significa que cada c_i é computado separadamente e os seis são **somados** —
vários podem disparar no mesmo passo. Um BLOCK em BENIGN dispara simultaneamente c₃, c₅ e
c₆: **os guardrails empilham por design**.

**Contabilidade terminal (Eq. 3.11):** −200 sempre, mais:

- +250 se ação terminal ∈ {BLOCK, ISOLATE}
- −150 se ação terminal ∈ {OBSERVE, LOG}
- 0 se RESTRICT (resultado não mitigado)

**Sinais desacoplados:** de-escalada +15 por push-back (teto 150/episódio); prevenção +50
(uma vez).

**Vetor de custo de ação:** C_action = [0; 0,1; 0,3; 0,5; 0,8], κ = 1,0. Monitoramento
passivo é grátis; isolamento é o controle mais caro.

**Calibração (procedimento explícito):**

- Terminais (−200/+250) definem a escala.
- Prevenção (+50) e de-escalada (+15) uma ordem de grandeza menores → o agente não pode
  inflar retorno ciclando de-escaladas.
- Bônus/penalidade por passo simétricos (±5).
- Guardrails benignos maiores que a moldagem por passo → sobre-bloquear nunca é lucrativo.
- Tetos (150, 100) removem o incentivo de exploração de recompensa.
- **Verificada por testes unitários** que asseveram que a política de ação recomendada é
  net-positiva em expectativa e always-OBSERVE é net-negativa.

**Botões desabilitados (zero em todos os experimentos):** penalidade lagrangiana de FP
benigno (β) e probabilidade de recuo não-monotônico. Nenhum afeta qualquer número reportado.

**Não é transformação baseada em potencial** no sentido estrito de Ng et al. (1999): mistura
custos dependentes de ação e estágio, guardrails e tetos sem função potencial associada.
Isso é declarado, não escondido.

### 3.7 Protocolo de treino

10 seeds, **5.000.000 passos fixos, sem parada antecipada**. Justificativa explícita: os três
algoritmos convergem em ritmos diferentes, e uma regra de parada calibrada para um trunca
outro antes de estabilizar. Melhor checkpoint em validação (não o último) é levado ao
benchmark.

Determinismo por seeding completo e álgebra linear single-threaded. Rede idêntica:
MlpPolicy, MLP 2×64 ReLU, para os três — isola a **regra de aprendizado** de efeitos de
capacidade.

**Hiperparâmetros (Apêndice C, grid search por algoritmo):**

| | PPO | A2C | DQN |
|---|---|---|---|
| lr | 3×10⁻⁴ | 7×10⁻⁴ | 5×10⁻⁴ |
| γ | 0,99 | 0,99 | 0,99 |
| n_steps | 2048 | **256** (default é 5) | — |
| λ_GAE | 0,95 | 0,95 | — |
| entropia / valor | 0,01 / 0,50 | 0,01 / 0,50 | — |
| épocas × batch | 10 × 64 | — | batch 64 |
| replay / learning starts | — | — | 200.000 / 5000 |
| target update / exploração | — | — | 5000 / 1,0→0,05 (20%) |
| max grad norm | 0,50 | 0,50 | 10 |

### 3.8 Baselines e oráculo

- **Random** — limite inferior.
- **Always-OBSERVE** — força zero, throughput máximo, todo impacto acontece.
- **Always-BLOCK** — força máxima, disrupção garantida.
- **RF-Acting** (implantável) — RF congelado → estágio predito → regra de ação recomendada.
  Roda predição **própria e independente** sobre a linha mais recente; não consome nenhum
  campo de estágio fornecido pelo ambiente.
- **Regra de Ação Recomendada (ORÁCULO — não é competidor)** — mesma regra lendo o estágio
  **verdadeiro** de decisão (não o próximo). É limite superior sobre o valor da percepção
  perfeita.

**Protocolo de avaliação:** n = 300 episódios por política. RL e RF-Acting:
10 seeds × 30 episódios. Determinísticas + oráculo: 1 seed × 300 episódios (logo não
carregam incerteza entre seeds de treino). CIs bootstrap descritivos; separação =
não-sobreposição de 95%.

---

## 4. Resultados (Cap. 4)

### Guia de leitura (Tabela 4.1) — cada avaliação mata uma explicação alternativa

| Avaliação | Explicação alternativa descartada | Desfecho |
|---|---|---|
| Cruzamento do aliasing | "o ambiente foi feito para favorecer RL" | Empate em α=0; gap +26,9 no headline |
| Acoplamento de recompensa | "a vantagem vem de moldagem privilegiada" | Melhor agente lidera sob ambas (−63,0) |
| Dificuldade do ambiente | "a vantagem é específica de um ajuste" | A2C lidera em toda taxa de de-escalada |
| Varredura de evasão | "controle colapsa contra atacante reativo" | A2C passa no critério de 25% |
| Fora da distribuição | "a vantagem só rastreia pontos cegos" | Vantagem +0,70–+0,78, sem tendência |

### 4.1 Treinamento

Melhor checkpoint em α = 0,4:

- **A2C +138,7**, sd ≈ 9, CI [+128,0; +150,0]
- **PPO +121,3**, sd ≈ 15, CI [+108,6; +133,1]
- **DQN +72,5**, sd ≈ 52, seeds de ≈ −15 a +132

A média do A2C fica **acima do limite superior do CI do PPO**; os intervalos se sobrepõem
apenas na borda inferior do A2C.

Explicação do DQN: o replay off-policy, tão eficaz sob recompensa moldada, **não propaga
crédito suficiente através do episódio esparso de 100 passos**.

### 4.2 Duas doutrinas (Fig. 4.5)

**A2C — prevenir na manobra:** bloqueia 84% dos passos em MANEUVER, nunca isola (0,00 em
ISOLATE em todos os estágios). Distribuição: BENIGN 0,27 LOG / 0,19 RESTRICT / 0,54 BLOCK;
MANEUVER 0,84 BLOCK.

**PPO — conter no impacto:** distribuição mais espalhada, admite o estágio tardio;
0,61 BLOCK e 0,11 ISOLATE em IMPACT.

**DQN:** concentra em LOG (0,82 em BENIGN, 0,79 em RECON) — política tímida; só chega a
0,59 BLOCK em IMPACT.

Por que A2C pontua mais: cada episódio adicionalmente prevenido **ganha o bônus de prevenção
e evita a penalidade de impacto**, enquanto as intervenções tardias do PPO concedem esse
custo terminal.

### 4.3 O cruzamento (Tabela 4.3) — decore esta tabela

| α | PPO [95% CI] | DQN | A2C | RF-Acting [95% CI] | Oráculo |
|---|---|---|---|---|---|
| 0,0 | **+138,6** [129,0; 147,7] | +116,6 | +147,1 | **+136,5** [126,1; 145,3] | +194,8 |
| 0,2 | +121,6 [109,1; 133,2] | +83,1 | +153,6 | +113,2 [99,4; 125,6] | +194,8 |
| **0,4** | **+121,3** [108,6; 133,1] | +72,5 | +138,7 | **+94,4** [81,4; 107,5] | +194,8 |
| 0,6 | +113,3 [100,4; 126,1] | +77,4 | +151,4 | +64,0 [49,7; 78,5] | +194,8 |
| 0,8 | +135,2 [124,9; 145,1] | +75,7 | +137,5 | +20,5 [5,4; 35,4] | +194,8 |
| 1,0 | +131,9 [120,7; 141,9] | +67,6 | +142,0 | **−29,3** [−46,3; −12,5] | +194,8 |

**ΔR = R_PPO − R_RF:** +2,1 → +8,4 → **+26,9** → +49,3 → +114,7 → **+161,2**

**Corroboração estatística adicional em α = 0,4:** teste de Wilcoxon dos postos sinalizados,
unilateral, das 10 médias por seed do PPO contra a média do RF-Acting: **p < 10⁻³**, com
**todas as 10 seeds do PPO excedendo o RF**.

**Triviais:** always-OBSERVE ≈ **−350**; always-BLOCK ≈ **0**; random ≈ **−100**. O contraste
entre os dois extremos confirma que a recompensa premia força proporcional, não passividade
nem bloqueio indiscriminado.

**Segurança em tráfego benigno:** PPO 0,89% / DQN 0,46% / A2C 0,66% — todos abaixo de 1%.
Random 41,3%. Always-BLOCK 100%. Always-OBSERVE 0%.

### 4.4 Ablação de acoplamento (Tabela 4.4)

Estimativa **separada e mais densa**: cada média agrupa as 10 seeds a 300 episódios cada
(n = 3.000).

| Recompensa | Melhor RL | Valor | RF-Acting [95% CI] | RF − melhor-RL |
|---|---|---|---|---|
| Coupled | DQN | +226,2 | +163,1 [146,5; 181,5] | **−63,1** |
| Outcome | A2C | +146,1 | +83,1 [69,7; 97,8] | **−63,0** |

Sob coupled: PPO +162,4; A2C +144,8. Sob outcome: PPO +126,2; DQN **−8,6**.

**Atenção a uma armadilha:** o DQN lê +72,5 na Tabela 4.3 (melhor checkpoint por seed) e
−8,6 aqui (média agrupada das 10 seeds). **Não é contradição** — são estimadores diferentes;
o DQN de alta variância lê mais baixo no agrupado. Os agentes on-policy, sendo estáveis,
concordam entre os dois.

**Desvio-padrão entre seeds do DQN: ≈17 (coupled) → ≈52 (outcome).** Triplica. Mesmo agente,
mesmo ambiente, só mudou a recompensa.

### 4.5 Dificuldade do ambiente (Fig. 4.8)

Políticas **fixas** (treinadas uma vez em p_down = 0,90), reavaliadas.

| p_down | A2C | PPO | DQN |
|---|---|---|---|
| 0,0 | −7,2 [−21,0; +5,4] | −46,2 [−62,4; −32,2] | −79,6 [−118,2; −43,1] |
| 0,6 | — | +90,9 | +44,6 |
| 1,0 | +147,3 | +142,6 [132,6; 152,2] | +78,3 |

Ordenação **A2C > PPO > DQN preservada em toda a varredura**. Oráculo: +35,2 → +195,2.

**Descontinuidade do oráculo em p_down = 0** (item que a banca pode cutucar): com p_down = 0
a resposta proporcional nunca de-escala, então o oráculo ganha **zero bônus de de-escalada**
e sua recompensa é prevenção (+50) menos custos de ação. Para qualquer p_down > 0,
de-escaladas repetidas **saturam o teto de +150**, produzindo o salto para ≈ +190.
**Isso é forma da recompensa, não capacidade do oráculo** — está declarado como limitação.

### 4.6 Evasão (Fig. 4.9)

Critério **pré-registrado**: o limite inferior do CI de 95% na evasão máxima não pode cair
mais que 25% da média sem evasão daquele defensor.

| | 0,00 → 0,75 | Comprometimento | Queda do limite inferior | Veredito |
|---|---|---|---|---|
| A2C | +142,6 → +112,7 | 0,233 → 0,412 | 28,8 | **passa** |
| PPO | +123,2 → +91,9 | 0,451 → 0,556 | 33,9 | **falha por pouco** |
| DQN | +68,7 → +50,5 | 0,677 → 0,706 | 16,8 | dentro da banda, mas base baixa |

Frase honesta da dissertação sobre o DQN: *"seu retorno absoluto baixo lhe deixa pouco a
perder, então isto não deve ser lido como defesa mais forte."*

### 4.7 OOD (Fig. 4.10)

Métrica: taxa de prevenção P_prev = Pr(max_t s_t < IMPACT).

| Classe | A2C | PPO | DQN | RF-Acting | Oráculo |
|---|---|---|---|---|---|
| DNS_Spoofing | 0,71 | 0,30 | 0,21 | 0,01 | 1,00 |
| VulnerabilityScan | 0,85 | 0,59 | 0,32 | 0,15 | 1,00 |
| SqlInjection | 0,76 | 0,53 | 0,31 | 0,00 | 1,00 |
| DDoS-SlowLoris | 0,76 | 0,52 | 0,32 | 0,01 | 1,00 |
| Recon-OSScan | 0,80 | 0,56 | 0,32 | 0,03 | 1,00 |
| XSS | 0,74 | 0,51 | 0,33 | 0,03 | 1,00 |
| DDoS-HTTP_Flood | 0,76 | 0,49 | 0,33 | 0,03 | 1,00 |
| DDoS-ACK_Frag | 0,75 | 0,50 | 0,33 | 0,03 | 1,00 |
| Mirai-udpplain | 0,75 | 0,50 | 0,36 | 0,03 | 1,00 |
| DoS-SYN_Flood | 0,76 | 0,54 | 0,32 | 0,03 | 1,00 |

**O par decisivo:** DoS-SYN_Flood (recall 0,998) → vantagem **+0,73**.
VulnerabilityScan (recall 0,224) → vantagem **+0,70**. Praticamente iguais.

Estatística: Spearman ρ = 0,22 (p = 0,54); Pearson r = −0,02 (p = 0,95); CI bootstrap da
inclinação OLS **[−0,08; +0,04]**, contém zero.

**Mecanismo (estrutural, não perceptual):** o RF-Acting escolhe observação ou registro em
**aproximadamente dois terços dos passos**, o que deixa o atacante avançar para o impacto.
Ele então bloqueia no próprio passo de impacto e registra "mitigado", não "prevenido".

**Nota obrigatória sobre always-BLOCK:** atinge prevenção perfeita 1,0 em todas as classes,
mas age agressivamente em 100% dos fluxos benignos e tem recompensa média 0,0 —
**operacionalmente inadmissível**. A comparação significativa é entre políticas que previnem
**permanecendo seguras em tráfego benigno**.

### 4.8 Footprint

Política vencedora ≈ **90 KB, 23K parâmetros**, contra ≈ **181 MB / 1,7M nós** do
RandomForest ajustado. Razão ≈ **1956×**.

**Nenhuma afirmação de latência de inferência é feita** — nem a baseline nem a política
foram otimizadas para inferência; comparação justa de latência é trabalho futuro.

---

## 5. Ameaças à validade (Seção 4.8) — memorize as cinco

1. **Dinâmica de atacante projetada + condicionalidade de classe.** O cruzamento é
   condicional a essa classe de atacante: a escalada acoplada à proximidade premia força
   proporcional sustentada, que é precisamente o que um classificador sem memória não pode
   fornecer. **Um adversário qualitativamente diferente — indiferente à força defensiva, ou
   escalando em cronograma fixo — poderia encolher ou apagar o gap.**
2. **Coerência de sessão é abstração de modelagem.** α é botão controlado, não propriedade
   medida. A contribuição é a **forma** da resposta, ancorada pelo empate em α=0.
3. **Baseline sem memória por construção.** O benchmark isola controle em janela vs.
   controle sem memória — **não** isola RL de aprendizado supervisionado *per se*. Uma
   baseline supervisionada com janela é o fortalecimento mais direto.
4. **Avaliação em dataset único.** Validade externa asseverada por construção, não
   demonstrada. Bot-IoT é o alvo natural.
5. **Poder estatístico.** Distinções mais finas entre algoritmos estão abaixo da resolução
   de dez seeds e **não são afirmadas**. OOD com n = 10: ausência de tendência detectável ≠
   independência provada.

---

## 6. Sete direções futuras

1. Políticas recorrentes com estado de crença (**ensaio preliminar não superou a janela sob
   o orçamento atual**)
2. Atacante não-monotônico e adaptativo
3. Aumento de dados com classes OOD em tempo de treino
4. Implantação em hardware de borda + quantização/poda
5. RL multiagente e adversário em self-play
6. Aprendizado federado multissítio
7. Formulação como MDP com restrições (garantia de FP via multiplicador de Lagrange)

---

# PARTE II — BANCO DE PERGUNTAS E RESPOSTAS

## A. Fundamentais / conceituais

**A1. Por que aprendizado por reforço, e não simplesmente um classificador melhor?**
Essa é exatamente a pergunta que a dissertação faz a si mesma, e eu não a respondo com
retórica — respondo com um experimento. Se cada fluxo revelasse o estágio, o RL seria
desperdício de complexidade, e eu demonstro isso: em α = 0 o RF-Acting **empata** com o PPO
(+136,5 vs +138,6, intervalos sobrepostos). O RL só justifica seu custo quando a decisão é
sequencial **e** parcialmente observada. O trabalho localiza o ponto onde isso passa a
valer: α = 0,4.

**A2. O que exatamente torna isto um POMDP e não um MDP com ruído?**
A distinção é que o estado latente é **markoviano no par (estágio, histórico)**, mas o
defensor recebe observações sorteadas de um núcleo Z que é uma mistura de dois componentes.
A política ótima teria que agir sobre a crença b_t(s), atualizada por filtro de Bayes.
Não é ruído aditivo sobre a observação — é **massa de probabilidade vinda da distribuição de
outro estado**, o que é qualitativamente diferente: nenhuma quantidade de filtragem por
passo recupera o estágio, só acumulação temporal.

**A3. Por que uma janela e não uma rede recorrente?**
A janela é o substituto prático padrão para controle POMDP (Sutton & Barto). E eu testei o
recorrente: um ensaio preliminar com política recorrente **não superou** a janela
feed-forward sob o orçamento de treino atual. Isso está registrado honestamente como
Direção 1 de trabalho futuro — caracterizar por quê, e se um horizonte mais longo, outra
arquitetura ou um objetivo explícito de estado de crença ajudariam.

**A4. Por que w = 5?**
Fornece contexto temporal mantendo a dimensão de observação gerenciável em 290. Declaro
explicitamente na Seção 3.6.6 que um estudo de tamanho de janela em contrato final é
trabalho de continuação, **não** evidência estabelecida aqui. Não afirmo que 5 é ótimo;
afirmo que é suficiente para demonstrar o efeito.

---

## B. Modelagem e metodologia

**B1. Você não fabricou a ambiguidade para o RL vencer?**
Três respostas empilhadas. Primeiro: a Figura 3.4 mostra que a sobreposição é **real** — na
projeção PCA, mesmo com informação completa de características, estágios vizinhos não são
linearmente separáveis. O α amplifica uma propriedade dos dados, não inventa uma. Segundo:
o empate em α = 0 é a âncora — se eu tivesse enviesado o ambiente, o RL já venceria ali.
Terceiro: o **mesmo** fluxo com aliasing alimenta todas as políticas, então ninguém é
privilegiado.

**B2. A coerência de sessão é imposta artificialmente. Isso não invalida o resultado?**
Invalidaria se eu afirmasse um número absoluto de campo. Eu declaro explicitamente: o
CICIoT2023 traz registros de fluxo pré-agregados **sem chave de sessão recuperável**, então
a estrutura temporal é imposta na camada de ambiente para representar as correlações
intra-sessão que um detector real enfrentaria. A contribuição é a **forma da resposta a α**
— a degradação monotônica do RF contra o PPO plano — ancorada pelo empate. É a Limitação 1,
declarada em três lugares do texto.

**B3. O atacante é projetado por você. Não é conveniente demais?**
É uma fronteira de escopo declarada, não uma conveniência. E note o que ela **custa** ao RL:
o atacante nunca pula estágios e nunca recua autonomamente, o que torna a política
always-BLOCK uma alternativa cara mas **batível** — não trivialmente dominante. Se eu
quisesse facilitar, teria feito o contrário. A honestidade que eu faço questão de manter é a
Ameaça (1): o cruzamento é **condicional a essa classe de atacante**. Um adversário
indiferente à força defensiva poderia encolher o gap. A varredura de persistência evasiva é
o relaxamento mais próximo que testei; self-play é a validação mais forte, e está deferida.

**B4. Por que a escalada acoplada à proximidade em vez de um orçamento de intrusão?**
Porque um orçamento fixo impõe um ponto de operação único e **arbitrário** que eu teria que
justificar. O acoplamento à proximidade — p_up^eff = p_up(σ_min + (1−σ_min)λ), com λ = s/4 e
σ_min = 0,4 — modela um adversário cujo momentum se compõe com o progresso, o que é
comportamento adversarial plausível, e torna a pressão endógena em vez de calibrada por mim.

**B5. Por que p_down = 0,90?**
Foi justificado por varredura (Seção 3.6.4). Está na região quase-ótima mas permanece
**sub-certo**, o que garante três coisas: (i) resposta proporcional reverte o atacante de
forma confiável mas não determinística; (ii) a estocasticidade residual impede uma solução
degenerada de tabela de consulta e mantém o aprendizado de valor significativo; (iii) a
recompensa da política treinada fica na mesma ordem de grandeza do teto do oráculo.

**B6. O clamp de piso em t = 20 não é arbitrário?**
Ele reflete a premissa do modelo de ameaça de que impacto no mundo real exige tempo de
execução. Sob a dinâmica estritamente sequencial do headline ele **raramente dispara** — é
mantido como piso de segurança. E é justamente por causa dele que eu reporto desfechos de
episódio (prevenção, comprometimento, FP benigno) em vez de sumários de tempo-até-
comprometimento, que seriam menos informativos.

---

## C. Recompensa

**C1. A recompensa não entrega o rótulo ao agente?**
Essa é a objeção que eu levo mais a sério, e é por isso que ela tem uma ablação dedicada. O
contrato **primário** é o esparso: só custo de ação, contabilidade terminal e prevenção —
**nenhuma dica de estágio por passo**. O contrato moldado existe apenas como ablação. E o
resultado fecha a objeção: o melhor agente lidera sob **ambos** (−63,1 e −63,0,
praticamente idênticos). A moldagem privilegiada é um atalho aprendível, não uma
pré-condição para a vantagem.

**C2. Como você calibrou as constantes? Não são arbitrárias?**
Há um procedimento explícito na Seção 3.6.3. Os terminais (−200/+250) definem a escala;
prevenção e de-escalada são uma ordem de grandeza menores para impedir inflação de retorno
por ciclagem; bônus e penalidade por passo são **simétricos** (±5) para que ignorar a ação
recomendada seja penalizado na mesma magnitude que cumprir é premiado; os guardrails
benignos são maiores que a moldagem para que sobre-bloquear **nunca** seja lucrativo; e os
tetos removem exploração de recompensa. Além disso, a calibração é **verificada por testes
unitários** que asseveram que a política de ação recomendada é net-positiva em expectativa e
always-OBSERVE é net-negativa.

**C3. A moldagem é invariante de política no sentido de Ng et al. (1999)?**
Não, e eu declaro isso explicitamente na Seção 3.7. Ela mistura custos dependentes de ação e
de estágio, guardrails e tetos por episódio que não têm função potencial associada. Por isso
mesmo eu não assumo suas consequências — eu as **sondo diretamente** com a ablação de
acoplamento.

**C4. Por que IMPACT não termina o episódio imediatamente?**
Porque o defensor merece uma última decisão — é o momento em que ISOLATE ainda tem valor
operacional, e a contabilidade terminal de três vias premia exatamente isso (+250 se
bloqueia/isola, −150 se estava passivo). Tratar IMPACT como imediatamente terminal é mantido
apenas como **estudo de caso secundário de mis-especificação de recompensa**, não como
resultado principal.

---

## D. Baselines e justiça

**D1. Seu RandomForest não é um espantalho?**
Deliberadamente não. É ajustado por busca em grade de **54 células** sobre macro-F1 de
validação, com **ótimo interior** — 200 árvores, profundidade 20, pesos balanceados. E a
evidência decisiva de que não está sub-ajustado: o macro-F1 de validação é **plano em
0,927 ± 0,005 sobre toda a grade**, amplitude de 0,016. Mais capacidade não ajudaria. Logo a
lacuna residual para o oráculo é propriedade da **observabilidade parcial da tarefa**, não de
um classificador mal configurado.

**D2. Você está comparando RL com supervisionado ou janela com sem-memória?**
Janela com sem-memória — e eu digo isso explicitamente como Ameaça (3). O RF-Acting consome
uma linha por decisão por construção. Uma baseline **supervisionada com janela** — por
exemplo um HMM ou uma cabeça de convolução temporal composta com a regra de ação — é a
baseline intermediária que separaria as duas coisas. Ela não foi executada e é nomeada como
o fortalecimento mais direto da afirmação central (Limitação 5). O que os experimentos
atuais **estabelecem** é que a vantagem não é artefato de moldagem privilegiada nem de um
ambiente construído para favorecer RL.

**D3. Por que o oráculo não conta como competidor?**
Porque ele lê o estado do simulador. Um defensor real não pode. Ele é **instrumento de
medida**: precifica o valor da percepção perfeita em +194,8 e define quanto da lacuna o
aprendizado consegue fechar. Toda afirmação numérica principal desta dissertação é enquadrada
contra esse teto in-domain — nunca contra um número externo de outro paper, porque ambientes
diferentes tornam comparação direta metodologicamente insustentável.

**D4. Always-BLOCK previne 100%. Por que não é a resposta?**
Porque ele age agressivamente em **100% dos fluxos benignos** e sua recompensa média é 0,0.
Um defensor que coloca em quarentena todo fluxo legítimo não é um defensor. A fronteira que
importa é prevenir **permanecendo seguro em tráfego benigno** — e nessa fronteira só as
políticas aprendidas vivem, com FP abaixo de 1%.

---

## E. Resultados e estatística

**E1. Dez seeds são suficientes?**
Para as afirmações que faço, sim; e eu delimito onde não são. O cruzamento principal repousa
em **intervalos disjuntos** em α = 0,4 e α = 0,6 — um critério conservador, mais exigente que
um teste de médias. Em α = 0,4 corroborei com Wilcoxon unilateral: **p < 10⁻³, com todas as
10 seeds do PPO excedendo o RF**. Distinções mais finas entre os algoritmos aprendidos estão
abaixo da resolução de dez seeds e **não são afirmadas** — isso está escrito na Ameaça (5).

**E2. O DQN aparece com +72,5 numa tabela e −8,6 em outra. Isso é inconsistência?**
Não, são estimadores diferentes e o texto explica isso. A Tabela 4.3 usa o melhor checkpoint
por seed; a Tabela 4.4 agrupa as 10 seeds inteiras a 300 episódios cada, n = 3.000 — uma
estimativa mais densa. O DQN, sendo de alta variância, lê mais baixo no agrupado. Os agentes
on-policy, sendo estáveis, **concordam de perto** entre os dois. A discrepância é, ela
própria, evidência do achado de confiabilidade.

**E3. O A2C é o melhor agente. Por que o PPO é a política de referência na narrativa?**
Porque seu platô de curva de aprendizado é mais plano e lê mais limpo nas varreduras fora de
distribuição. Mas isso é escolha narrativa, declarada — **A2C e DQN são reportados ao lado em
toda parte**, e eu afirmo explicitamente que o A2C atinge a política de maior recompensa no
ponto de referência e que a estabilidade é propriedade da **família on-policy**, não do PPO
isoladamente.

**E4. A curva do PPO sobe de novo em α = 0,8. Isso não é estranho?**
É não-monotonicidade dentro de intervalos sobrepostos. A afirmação que faço é que o PPO
permanece **plano dentro dos intervalos de confiança, sem tendência monotônica** — não que
seja constante. A flutuação de +113,3 para +135,2 está dentro da variabilidade amostral; o
que importa é o contraste com a queda monotônica e estatisticamente clara do RF.

**E5. E a descontinuidade do oráculo em p_down = 0?**
É dependente da forma da recompensa e eu a declaro como limitação. Em p_down = 0, a resposta
proporcional nunca de-escala, então o oráculo ganha zero bônus de de-escalada e sua
recompensa é prevenção (+50) menos custos de ação. Para qualquer p_down > 0, de-escaladas
repetidas **saturam o teto de +150**, produzindo o salto para ≈ +190. É o desenho da
recompensa aparecendo, não capacidade do oráculo.

---

## F. Robustez e OOD

**F1. O RL só vence onde o detector é cego, certo?**
Essa hipótese faz uma **predição falsificável**: a vantagem deveria encolher conforme o
recall sobe. Eu construí as dez classes retidas cobrindo o espectro inteiro, de 0,20 a 0,998,
exatamente para testá-la. Resultado: nenhuma tendência detectável. Spearman ρ = 0,22
(p = 0,54), Pearson r = −0,02 (p = 0,95), CI da inclinação OLS [−0,08; +0,04] contendo zero.
Concretamente: em DoS-SYN_Flood, recall 0,998, a vantagem é +0,73; em VulnerabilityScan,
recall 0,224, é +0,70. A ressalva que faço sozinho: n = 10 é ausência de tendência
detectável, **não** prova de independência. Mas a tendência que a objeção **exige** —
negativa — está ausente.

**F2. Então por que o RF falha mesmo onde vê bem?**
Por razão **estrutural, não perceptual**. Prevenção exige força proporcional sustentada e
bem-cronometrada ao longo de todo o horizonte. O RF-Acting compromete-se a uma ação por
linha e, mesmo com predição correta, joga um mix de ações largamente passivo: escolhe
observação ou registro em **cerca de dois terços dos passos**. Isso deixa o atacante alcançar
o impacto; ele então bloqueia no próprio passo de impacto e registra "mitigado", não
"prevenido". Um classificador de um disparo é **estruturalmente incapaz de expressar controle
temporal**, por mais acuradamente que rotule qualquer fluxo individual.

**F3. As taxas absolutas de prevenção são modestas — 0,71 a 0,85. Isso não é fraco?**
É moderado e eu digo isso. O teto do oráculo (≈ +194,8) está bem acima da faixa da melhor
política aprendida (+138,7 a +151,4), confirmando que ela captura só parte do valor do
oráculo. A contribuição é a **vantagem relativa** sobre a baseline implantável e o **custo da
observabilidade parcial** quantificado pela lacuna do oráculo — não uma afirmação de
segurança quase perfeita.

**F4. O protocolo de injeção de características não é artificial?**
É um teste de estresse controlado, e é assim que eu o descrevo. A injeção é de **estágio
único**: as linhas da classe retida substituem a realização apenas no estágio ao qual aquela
classe pertence, enquanto os outros quatro continuam sorteando dados em distribuição. Isso
isola o deslocamento precisamente onde importa, evitando o confundimento de forçar estágios
que a classe não ocupa a usar características desconhecidas. É explicitamente **não**
evidência de generalização a zero-day implantado.

---

## G. Trabalhos relacionados

**G1. Como isto se compara ao IoTWarden?**
Adoto o **mapeamento de ação recomendada** do IoTWarden como escolha de projeto, e o trato
como inspiração, não baseline head-to-head. Ele foi avaliado num ambiente diferente —
trigger-action sintético, IFTTT — com espaço de ação diferente, então comparação numérica
direta não seria metodologicamente sólida. Por isso a afirmação numérica principal desta
dissertação é enquadrada contra o teto do oráculo in-domain, **nunca** contra um número
externo do IoTWarden.

**G2. E o RL-IoTIDS, que também usa DRL sobre CICIoT2023?**
É o contraste mais direto e mais instrutivo. Ele acopla um agente DQN a um extrator CNN-LSTM
para **classificar** tráfego malicioso nas 33 classes, e a recompensa premia classificação
correta por fluxo. Logo a política aprendida é fundamentalmente um **detector one-shot**, não
um controlador de kill chain. É precisamente a formulação que eu argumento ser insuficiente
sob observabilidade parcial: onde o RL-IoTIDS otimiza acurácia de detecção por fluxo, este
trabalho otimiza controle defensivo sustentado e bem-cronometrado sobre uma janela temporal.
O Capítulo 4 mostra que o segundo previne ataques que o primeiro não pode, com vantagem que
não rastreia o recall de rotulagem.

**G3. E o RESTRAIN?**
É o análogo multiagente mais próximo e a contribuição é **ortogonal, não competidora**. O
RESTRAIN ataca coordenação multiagente numa plataforma trigger-action sintética; esta
dissertação ataca observabilidade parcial por fluxo em tráfego real do CICIoT2023. Os dois
eixos — coordenação cooperativa e crença de agente único sob aliasing — são complementares.
Integrar o estressor de observabilidade parcial num cenário estilo RESTRAIN é a fusão
natural, deixada como trabalho futuro.

> **Nota de preparação:** o Prof. João Kleinschmidt é **co-autor citado** na sua bibliografia
> (Okey et al., 2023, *IEEE Access*, e Okey et al., 2024, *ISWCS* — federated learning com
> ensembles CNN-GRU/LSTM-GRU para detecção de intrusão IoT). Vale ter isso na ponta da
> língua: se ele perguntar sobre aprendizado federado ou modelos profundos para IDS, você
> pode conectar diretamente à **Direção 6** (federado multissítio) e reconhecer o trabalho
> dele como parte da linha que motiva a direção.

---

## H. Prática e implantação

**H1. Isto roda numa rede real?**
A política é um MLP de 2×64 com **90 KB e 23 mil parâmetros** — cerca de 1956× menor que o
RandomForest de 181 MB que ela supera. Isso a coloca confortavelmente no orçamento de um
gateway de borda. Faço questão de **não** afirmar latência de inferência, porque nem a
baseline nem a política foram otimizadas para inferência; uma comparação justa de latência é
ela própria trabalho futuro (Direção 4).

**H2. E os falsos positivos? Um defensor autônomo que bloqueia demais é pior que nada.**
Concordo, e por isso reporto o eixo de disponibilidade diretamente. Ação agressiva sobre
fluxos verdadeiramente benignos: PPO 0,89%, A2C 0,66%, DQN 0,46% — todos abaixo do limiar
operacional de 1%. Para contexto, a política aleatória perturba 41,3%. As políticas aprendidas
alcançam sua vantagem de prevenção **mantendo** disrupção benigna abaixo de 1%, e essa é a
combinação que importa para implantação.

**H3. Como um operador confiaria numa política de caixa-preta?**
Esse é um argumento que a Figura 4.5 sustenta melhor que qualquer métrica. As políticas não
aprenderam apenas escores — aprenderam **doutrinas inspecionáveis**. O A2C bloqueia 84% dos
passos em MANEUVER e nunca isola: é "prevenir na manobra". O PPO admite o estágio tardio e
isola: é "conter no impacto". Um operador pode olhar essa tabela e dizer em português qual é
a estratégia. E nenhuma foi programada — ambas emergiram da mesma recompensa esparsa.

**H4. Quanto custou computacionalmente?**
Tudo em **CPU apenas**, estação de trabalho comum, sem GPU em nenhum ponto. Python 3.9,
PyTorch CPU, scikit-learn, Stable Baselines3, Gymnasium. Cada agente: 5 milhões de passos,
10 seeds. A cadeia completa de avaliação roda em poucas horas, dominada pela varredura de
acoplamento de recompensa. E a **verificação** da reprodutibilidade não exige retreinar nada.

**H5. Como você garante que os números são reproduzíveis?**
Cada figura e tabela acompanha um `manifest.json` com hash de conteúdo de cada artefato de
entrada, o commit git produtor e o comando exato. Cada manifesto também registra o hash de
cada **saída**, para que um componente a jusante possa fixar uma saída a montante por hash,
não por nome de arquivo. Uma rotina de verificação repercorre a cadeia num clone limpo. São
**462 testes**. Os números do texto são gerados por macros a partir de JSONs canônicos —
nenhum foi digitado à mão, o que elimina uma classe inteira de erro.

---

## I. Limitações e futuro

**I1. Qual é a maior fraqueza do trabalho?**
A baseline supervisionada sem memória por construção. O benchmark isola controle em janela
contra controle por fluxo — ele **não** isola aprendizado por reforço de aprendizado
supervisionado *per se*. Uma baseline supervisionada com a mesma janela de 5 linhas
atribuiria a vantagem especificamente à política de controle aprendida. Ela não foi executada
e eu a nomeio como o fortalecimento mais direto da afirmação central. O que os experimentos
atuais **descartam** são as duas explicações alternativas mais fortes: moldagem privilegiada
(ablação de acoplamento) e ambiente enviesado (empate em α = 0).

**I2. Por que só um dataset?**
Toda a validade externa aqui é asseverada por construção, não demonstrada — a projeção de
kill chain, o espectro de recall do detector e a família de classes OOD são todos
propriedades deste corpus. Replicar o cruzamento do aliasing num corpus independente com mix
de dispositivos comparável — **Bot-IoT** é o alvo natural — é requisito antes de afirmar
independência de dataset, e está declarado assim.

**I3. Se você tivesse mais seis meses, o que faria?**
Duas coisas, nessa ordem. Primeiro, a baseline supervisionada com janela, porque é a que mais
aperta o argumento científico. Segundo, o atacante em self-play co-adaptativo — porque é a
validação mais forte do cruzamento e, além disso, produziria uma distribuição de treino mais
difícil que poderia estreitar a lacuna remanescente até o teto do oráculo.

---

## J. Perguntas-armadilha (prepare-se para estas)

**J1. "Você não mediu tempo até comprometimento. Por quê?"**
Porque o clamp de piso de ciclo de vida torna afirmações irrestritas de temporização menos
informativas que desfechos diretos de episódio. Reporto prevenção, comprometimento e taxa de
falso positivo benigno — que são as três quantidades que um operador de fato precisa.

**J2. "Seu A2C nunca usa ISOLATE. Isso não é uma política defeituosa?"**
Não — é uma doutrina coerente. O A2C estrangula a campanha em MANEUVER, onde BLOCK **é** a
ação proporcional. Se ele previne o episódio, nunca chega a IMPACT, e ISOLATE nunca é a ação
recomendada. Usar ISOLATE seria super-dimensionar e pagar custo de disponibilidade sem ganho.
A política está consistente com a estrutura de incentivo, e sua taxa de FP benigno de 0,66%
confirma que ela não é agressiva de forma indiscriminada.

**J3. "Você escolheu α = 0,4 como headline. Isso não é cherry-picking?"**
É o **primeiro** ponto onde os intervalos se tornam disjuntos — ou seja, o ponto mais
**conservador** em que a afirmação de separação se sustenta. Se eu quisesse maximizar o
efeito teria escolhido α = 1,0, onde o gap é +161,2. Escolher o primeiro ponto significativo é
o oposto de cherry-picking. E reporto a curva inteira, incluindo os pontos onde não há
separação.

**J4. "Seu DQN vai mal. Você não o ajustou o suficiente?"**
Ele foi ajustado por busca em grade como os outros, com buffer de replay maior (200 mil) e
atualizações de alvo mais lentas justamente para estabilizar o bootstrap off-policy. E há uma
evidência que descarta sub-ajuste: sob o contrato **coupled**, o mesmo DQN com os mesmos
hiperparâmetros é o **melhor** agente, com +226,2 e desvio entre seeds de apenas ≈17. Ele não
está mal configurado — ele é estruturalmente mal-adaptado à atribuição de crédito esparsa.
Essa é a descoberta, não um defeito de execução.

**J5. "Você usou IA generativa na escrita?"**
Sim, e está declarado formalmente antes da bibliografia, conforme a Deliberação
CONSU-A-005/2026 da UNICAMP. A ferramenta foi usada como auxílio de redação e formatação —
copidesque, condensação, LaTeX e figuras. Ela **não** projetou os experimentos, não rodou as
simulações, não gerou os dados nem produziu qualquer resultado. Revisei e editei todo o texto
assistido e assumo responsabilidade integral. Cada afirmação técnica, número e figura foi
verificado independentemente contra os artefatos experimentais e a cadeia reprodutível.

**J6. "Qual é, afinal, a contribuição científica — vocês só mostraram que RL funciona?"**
Não. Mostrar que RL "funciona" não seria contribuição — seria repetição. A contribuição é um
**relato controlado de quando e por que ele funciona**, com a fronteira localizada
quantitativamente e com as três explicações alternativas mais plausíveis explicitamente
descartadas por experimento: ambiente enviesado (empate em α=0), recompensa privilegiada
(ablação de acoplamento) e pontos cegos do detector (independência de recall). Mais o
instrumento em si — um ambiente de kill chain genuinamente parcialmente observável sobre
tráfego real, reprodutível por terceiros sem retreinar nada.

---

## Cola final — os dez números que você não pode errar

| # | Número | O que é |
|---|---|---|
| 1 | **+138,6 vs +136,5** | Empate em α = 0 (PPO vs RF) — a âncora de honestidade |
| 2 | **+26,9** | Gap em α = 0,4, primeiro ponto de CIs disjuntos |
| 3 | **−29,3** | RF em α = 1,0 — torna-se net-harmful |
| 4 | **+194,8** | Teto do oráculo |
| 5 | **−63,1 / −63,0** | Gap RF−RL sob coupled / outcome |
| 6 | **9 / 15 / 52** | Desvio entre seeds A2C / PPO / DQN |
| 7 | **0,71–0,85 vs 0,00–0,15** | Prevenção OOD A2C vs RF |
| 8 | **ρ = 0,22 (p = 0,54)** | Independência do recall |
| 9 | **0,924** | Macro-F1 do RF ajustado |
| 10 | **90 KB vs 181 MB** | Footprint — razão 1956× |
