"use strict";
// ---------------------------------------------------------------------------
// Speaker notes (pt-BR) — single source of truth for the spoken script.
//
// CONTRACT FOR EVERY NOTE IN THIS FILE
//   1. First line is an explicit time budget: [inicio -> fim | alvo N s].
//      The clock is elapsed talk time, starting at 00:00.
//   2. NEVER narrate a fact that is not printed on the slide. If it matters
//      enough to say out loud, it belongs on the slide.        (feedback [5])
//   3. NEVER pull material from a later act into an earlier one. Fundamentals
//      stay fundamentals; method stays in the method.  (feedback: Arthur 7/13)
//   4. Vocabulary on the slide == vocabulary in the mouth. Say "OUTCOME, a
//      esparsa", never "sparse" alone.                        (feedback [17])
//   5. Anchor the audience physically before explaining: "bloco um, a
//      esquerda", "cartao vermelho, embaixo".                  (feedback [4])
//
// TOTAL: 40 minutes of talk (slides 1-31), leaving the rest for questions.
// Two slides are marked [OPCIONAL] and may be skipped if running late.
// ---------------------------------------------------------------------------

const NOTES = {};

// ===========================================================================
// OPENING — slides 1-3 — 00:00 -> 02:00
// ===========================================================================

NOTES.title =
  "[00:00 -> 00:45 | alvo 45 s]\n\n" +
  "Bom dia. Agradeco a banca: Prof. Dr. Denis Fantinato, meu orientador; Prof. Dr. Alexandre da Silva Simoes, da UNESP; " +
  "e Prof. Dr. Joao Kleinschmidt, da UFABC. Obrigado a todos que vieram acompanhar.\n\n" +
  "O titulo e 'Um Arcabouco de Aprendizado por Reforco Profundo para Defesa Cibernetica Autonoma em Redes IoT'.\n\n" +
  "A pergunta que orienta o trabalho e facil de enunciar e dificil de responder com honestidade: quando um agente que aprende " +
  "a decidir ao longo do tempo realmente supera um classificador bem ajustado? Nao 'se' supera - 'quando', e 'por que'.\n\n" +
  "-- DISCIPLINA: 45 segundos. Nao improvise aqui. A banca ja leu a dissertacao.";

NOTES.speaker =
  "[00:45 -> 01:20 | alvo 35 s]\n\n" +
  "Rapidamente sobre mim. Sou mestrando em Engenharia Eletrica, area de Engenharia de Computacao, na FEEC/UNICAMP.\n\n" +
  "Sou formado em Engenharia Eletrica pela UFPB, com enfase em Controle e Automacao, e passei um ano na Universidade de " +
  "Wisconsin-Milwaukee pelo Ciencia sem Fronteiras.\n\n" +
  "Hoje sou arquiteto de solucoes em plataformas IoT e Edge-AI na Globant. E dai que vem o interesse pratico neste tema: " +
  "os setores nos logos abaixo sao todos operacoes com frota de dispositivo conectado - aviacao, energia, telecom, manufatura.\n\n" +
  "-- REGRAS DESTE SLIDE (feedback do orientador):\n" +
  "   . NAO mencionar aluno especial. Nao acrescenta nada.\n" +
  "   . NAO percorrer os logos um a um. Uma frase agregando, e segue.\n" +
  "   . 35 segundos. Este slide e para o publico, nao para a banca.";

NOTES.agenda =
  "[01:20 -> 02:00 | alvo 40 s]\n\n" +
  "O roteiro tem cinco partes.\n\n" +
  "Primeiro eu construo o vocabulario: o que e uma rede IoT, por que a seguranca estatica nao serve, o que e a kill chain, " +
  "e o que e aprendizado por reforco.\n\n" +
  "Segundo, a pergunta de pesquisa e os objetivos.\n" +
  "Terceiro, o arcabouco que construimos.\n" +
  "Quarto, a evidencia.\n" +
  "E quinto, o que isso significa, o que nao significa, e para onde vai.\n\n" +
  "-- DISCIPLINA: 40 segundos. So o nome de cada ato e uma frase. O conteudo vem no ato correspondente. " +
  "Nao comece a explicar a kill chain aqui.";

// ===========================================================================
// ACT I — CONTEXT — slides 4-8 — 02:00 -> 12:00  (era ~14 min sozinho)
// ===========================================================================

NOTES.iotNetwork =
  "[02:00 -> 03:10 | alvo 70 s]\n\n" +
  "Antes de falar de vulnerabilidade, vamos concordar sobre o que e, concretamente, uma rede IoT. " +
  "O slide esta numerado - vou seguir os numeros, e peco que acompanhem por eles.\n\n" +
  "BLOCO UM, a esquerda: as coisas. Camera, sensor de presenca, fechadura, termostato, caixa de som, vestivel. " +
  "Dispositivos baratos, restritos, sem fio.\n\n" +
  "BLOCO DOIS, no centro: o gateway, ou hub. Ele agrega o trafego e traduz protocolo - Zigbee, BLE e Wi-Fi do lado dos " +
  "dispositivos; MQTT e HTTP do lado da Internet.\n\n" +
  "BLOCO TRES, a direita: os servicos de nuvem - analise, paineis, controle remoto.\n\n" +
  "A telemetria sobe, os comandos descem, e cada salto desse caminho e trafego de rede.\n\n" +
  "Embaixo, tres caracteristicas que importam para seguranca. QUATRO: o mundo fisico esta dentro do laco - dispositivo " +
  "comprometido tem consequencia fisica. CINCO: o hardware e classe microcontrolador - kilobytes de RAM; nao ha espaco para " +
  "antivirus. SEIS: o conjunto e heterogeneo e sempre conectado.\n\n" +
  "E a linha final responde a pergunta obvia - 'entao onde a defesa acontece?'. Ela acontece no BLOCO DOIS, o gateway: " +
  "e o unico ponto que ve todos os fluxos. E exatamente ali que este trabalho atua.\n\n" +
  "-- ANCORAGEM: fale sempre o numero antes de descrever. Sem isso a plateia nao acha o que voce esta citando.\n" +
  "-- NAO diga 'aliasing', 'POMDP' nem 'alfa' aqui. E cedo demais.";

NOTES.whyIoT =
  "[03:10 -> 04:20 | alvo 70 s]\n\n" +
  "Agora a escala, porque ela sozinha ja muda o problema.\n\n" +
  "Numero da esquerda: 19,8 bilhoes de dispositivos IoT hoje, projetados para 40,6 bilhoes ate 2034. " +
  "No centro: 12,2 trilhoes de dolares de prejuizo anual projetado com cibercrime ate 2031. " +
  "A direita: 33 tipos de ataque executados dispositivo contra dispositivo no dataset que eu uso.\n\n" +
  "Abaixo da linha, tres razoes estruturais pelas quais a seguranca tradicional de TI nao se transplanta para ca.\n\n" +
  "Cartao um - dispositivos restritos: uma camera de trinta dolares nao roda antivirus.\n" +
  "Cartao dois - heterogeneidade extrema: protocolos e sistemas operacionais dispares tornam politica uniforme impraticavel.\n" +
  "Cartao tres - defesas estaticas: IDS por assinatura nao pega zero-day, porque zero-day nao tem assinatura; e IDS por " +
  "anomalia afoga o operador em falso positivo.\n\n" +
  "A lacuna, portanto, e bem definida, e esta na ultima linha: precisamos de defesa que se adapte, aprenda dos dados e aja " +
  "com autonomia.\n\n" +
  "-- DISCIPLINA (feedback [5]): tudo que voce disser aqui esta escrito no slide. Se sentir vontade de acrescentar um exemplo " +
  "que nao esta ali, NAO acrescente. Foi exatamente isso que estourou o tempo na primeira versao.";

NOTES.killChain =
  "[04:20 -> 06:00 | alvo 100 s]\n\n" +
  "Este e o slide de vocabulario. Tudo que vem depois se apoia nele.\n\n" +
  "A ideia central: uma intrusao nao e um evento, e uma campanha em estagios. O atacante nao aparece do nada com um DDoS.\n\n" +
  "Fileira de cima, da esquerda para a direita: BENIGN, trafego normal. RECON, reconhecimento - varredura, fingerprinting; " +
  "ele esta mapeando a casa. ACCESS, obtencao de acesso - forca bruta, injecao; ele conseguiu a chave. MANEUVER, movimentacao " +
  "lateral e preparacao de botnet; ele esta se posicionando. IMPACT, a detonacao - DDoS, exfiltracao.\n\n" +
  "Fileira de baixo, o espelho da defesa: cada estagio tem uma resposta proporcional. OBSERVE, LOG, RESTRICT, BLOCK, ISOLATE.\n\n" +
  "E a tensao que da vida ao trabalho esta na seta na base: o custo da interrupcao cresce ao longo da cadeia. Interromper " +
  "cedo e barato, tarde e caro. Mas exagerar cedo tambem custa - voce nao pode isolar a rede a cada varredura de porta. " +
  "Isso e dano de disponibilidade autoinfligido.\n\n" +
  "Defender bem nao e bloquear tudo. E acertar a dose e o momento.\n\n" +
  "-- TETO RIGIDO: 100 segundos. Na primeira versao este slide levou 4 minutos. Se voce chegar aos 06:00 do relogio " +
  "e ainda estiver aqui, PULE para o proximo imediatamente.";

NOTES.rlPrimer =
  "[06:00 -> 07:40 | alvo 100 s]\n\n" +
  "Aprendizado por reforco. A banca conhece o assunto, entao isto e calibracao de vocabulario, nao aula.\n\n" +
  "A esquerda, o laco: o agente observa, escolhe uma acao, e o ambiente devolve recompensa e o proximo estado. Ele aprende " +
  "uma politica que maximiza recompensa acumulada. Sem rotulos.\n\n" +
  "Logo abaixo, no cartao verde, a pergunta que interessa: POR QUE reforco aqui, e nao aprendizado supervisionado? Tres razoes.\n" +
  "Primeira: a resposta muda a ameaca. Quando eu bloqueio, o atacante reage. Isso e controle, nao rotulagem.\n" +
  "Segunda: o sucesso e um desfecho de episodio - o ataque foi contido ou nao - e nao um veredito por fluxo.\n" +
  "Terceira, e a mais decisiva: nao existe 'acao correta' rotulada para supervisionar. Ninguem anotou qual seria a resposta " +
  "certa em cada passo.\n\n" +
  "A direita, os tres algoritmos. DQN, off-policy, baseado em valor, com replay buffer. PPO, on-policy, ator-critico, com " +
  "atualizacoes recortadas. A2C, tambem on-policy, sincrono e mais simples.\n\n" +
  "No cartao cinza, POR QUE estes tres: eles cobrem as duas familias model-free para acao discreta, sao o conjunto " +
  "recomendado pelo guia do Stable-Baselines3 e o mais avaliado na literatura de resposta a intrusao. SAC, TD3 e Dreamer " +
  "ficam de fora porque um menu discreto de cinco acoes nao precisa deles.\n\n" +
  "E a linha de baixo, que e uma correcao importante de rigor: o que eu mantenho fixo e a rede, o orcamento de 5 milhoes de " +
  "passos DE INTERACAO com o ambiente, e as sementes. Nao e o numero de amostras de gradiente - o DQN reaproveita cada " +
  "transicao muitas vezes via replay. Esse reaproveitamento nao e um artefato: ele E a regra de aprendizado sob teste.\n\n" +
  "-- Se perguntarem sobre hiperparametros: BACKUP B2. Nao os cite aqui (feedback do Arthur: voce adiantou metodologia " +
  "neste slide na primeira versao).";

NOTES.pomdp =
  "[07:40 -> 09:40 | alvo 120 s]\n\n" +
  "Este e o conceito central da dissertacao. E o unico slide do Ato I onde eu quero que voces gastem tempo comigo.\n\n" +
  "Caixa da esquerda: num Processo de Decisao de Markov classico, um MDP, o agente enxerga o estado verdadeiro. E repare na " +
  "consequencia, que esta escrita ali: se o estado e visivel, uma regra sem memoria pode ser otima. Ou seja, classificacao " +
  "bastaria. Voce classifica o estado, consulta a acao recomendada, e acabou.\n\n" +
  "Caixa da direita, que e o nosso caso: o defensor nunca enxerga o estagio verdadeiro da kill chain. Ele e latente. " +
  "O que ele ve sao linhas de caracteristicas de trafego - e estagios adjacentes emitem caracteristicas que se sobrepoem.\n\n" +
  "Isso e observabilidade parcial: um POMDP. A observacao vem de um nucleo Z condicionado ao estagio oculto, e a politica " +
  "otima precisa agir sobre uma CRENCA construida a partir do historico.\n\n" +
  "A metafora que eu gosto: um medico que nunca ve a doenca, so os sintomas - e doencas vizinhas compartilham sintomas. " +
  "Uma foto instantanea nao diagnostica. Um historico de caso diagnostica.\n\n" +
  "A linha de baixo mostra o caminho que o agente percorre: uma janela com as ultimas cinco observacoes; dela, uma crenca " +
  "implicita; a politica condicionada nessa janela; e a acao defensiva proporcional. A janela e a unica maquinaria de " +
  "memoria que o agente tem.\n\n" +
  "E a frase que fecha o slide e a tese em uma linha: se um instantaneo nao diagnostica, a memoria vira a defesa.\n\n" +
  "-- NAO mencione a taxa de aliasing alfa aqui. Na primeira versao a nota terminava adiantando alfa, e foi assim que a " +
  "metodologia vazou para dentro da fundamentacao. Alfa aparece no Ato III, e nao antes.";

// ===========================================================================
// ACT II — QUESTION + OBJECTIVES — slides 9-11 — 09:40 -> 13:00
// ===========================================================================

NOTES.question =
  "[09:40 -> 11:00 | alvo 80 s]\n\n" +
  "Agora eu posso enunciar a pergunta com precisao. E peco que reparem no que ela NAO e.\n\n" +
  "O discurso ingenuo seria 'RL supera classificador'. Isso nao e afirmacao cientifica, e marketing. So vira ciencia quando " +
  "voce pergunta QUANDO e POR QUE.\n\n" +
  "Antes da objecao, a faixa do meio responde a uma pergunta que ficou no ar na primeira versao: como e que um classificador " +
  "supervisionado defende, concretamente? Assim, da esquerda para a direita: chega UM fluxo, uma linha de 29 caracteristicas; " +
  "o RandomForest prediz o estagio; uma tabela fixa converte estagio em acao recomendada; e ele age. " +
  "Repare na ultima caixa: sem memoria nenhuma do passado. Essa e a baseline que eu preciso vencer, e ela e forte - " +
  "e o que um profissional levaria para producao hoje.\n\n" +
  "Cartao vermelho, a objecao do cetico, que a dissertacao levanta contra si mesma: 'se cada fluxo revela o estagio, seu " +
  "agente de RL e so um classificador caro - um modelo supervisionado deveria empatar com ele.'\n\n" +
  "Eu levo essa objecao a serio. A estrategia de resposta esta na ultima linha: transformar a ambiguidade de estagio em um " +
  "botao controlado, manter todo o resto fixo, e medir onde a classificacao deixa de bastar.\n\n" +
  "-- Aqui voce PODE dizer 'taxa de aliasing alfa' pela primeira vez, porque esta escrito no slide.";

NOTES.objectives =
  "[11:00 -> 12:10 | alvo 70 s]\n\n" +
  "Slide de objetivos - incluido a pedido da banca na primeira apresentacao.\n\n" +
  "O objetivo geral, no cartao dourado: projetar, implementar e avaliar um arcabouco de defesa adaptativa em laco fechado " +
  "para redes IoT, em que um defensor por reforco, ciente da kill chain, enfrenta um adversario reativo sobre trafego real " +
  "do CICIoT2023 - formulado como um problema genuino de observabilidade parcial, e nao como classificacao disfarcada.\n\n" +
  "Essa ultima oracao e o compromisso central do trabalho, e e o que separa esta dissertacao das formulacoes anteriores.\n\n" +
  "Os objetivos especificos, numerados:\n" +
  "Um: construir o ambiente parcialmente observavel.\n" +
  "Dois: localizar onde o controle em janela ultrapassa o controle por fluxo.\n" +
  "Tres: testar se essa vantagem depende de uma recompensa privilegiada.\n" +
  "Quatro: medir generalizacao para classes de ataque nunca treinadas.\n" +
  "Cinco: relatar confiabilidade entre algoritmos e tornar tudo reproduzivel.\n\n" +
  "E a linha final e a promessa de estrutura: cada objetivo especifico corresponde a uma contribuicao e a um slide de " +
  "resultado mais adiante. Quando chegarmos la, eu vou fechar cada um deles.\n\n" +
  "-- 70 segundos. Leia os cinco em ritmo firme; nao explique nenhum em detalhe - eles voltam com evidencia.";

NOTES.contributions =
  "[12:10 -> 13:00 | alvo 50 s]\n\n" +
  "As cinco contribuicoes, uma respiracao cada. Sao os cinco objetivos do slide anterior, agora do ponto de vista do que foi " +
  "entregue.\n\n" +
  "Um: o ambiente de kill chain genuinamente parcialmente observavel - o instrumento de medida.\n" +
  "Dois: o cruzamento controlado - empate em alfa igual a zero, por construcao, e separacao estatistica conforme a " +
  "ambiguidade cresce.\n" +
  "Tres: a ablacao de recompensa, que responde a objecao do sinal privilegiado.\n" +
  "Quatro: vantagem de prevencao em dez classes de ataque jamais vistas.\n" +
  "Cinco: relato honesto de confiabilidade entre algoritmos, mais a cadeia reprodutivel com verificacao por hash.\n\n" +
  "-- 50 segundos. Este slide e um indice, nao um argumento. Se estiver atrasado, leia so os cinco titulos em negrito " +
  "e siga.";

// ===========================================================================
// ACT III — FRAMEWORK — slides 12-20 — 13:00 -> 23:30
// ===========================================================================

NOTES.architecture =
  "[13:00 -> 14:10 | alvo 70 s]\n\n" +
  "O arcabouco em uma visao. A figura tem tres blocos.\n\n" +
  "Preparacao offline, a esquerda: projetamos o CICIoT2023 sobre os cinco estagios; especificamos o atacante em forma " +
  "fechada; e treinamos o detector supervisionado de estagio - que, atencao, alimenta a BASELINE, e nao o nosso agente.\n\n" +
  "O laco online, no centro, que e onde a pesquisa acontece: o atacante emite um estagio; o motor de realizacao amostra uma " +
  "linha real correspondente aquele estagio; o ambiente monta a observacao em janela; o agente age; e o nucleo de escalada " +
  "move o atacante em funcao da forca aplicada.\n\n" +
  "Avaliacao, a direita: benchmark em dados retidos contra as baselines e o oraculo, e depois as ablacoes.\n\n" +
  "A frase do cartao azul e a que eu peco que guardem: o laco e genuinamente fechado, inclusive durante a avaliacao. " +
  "Nao estamos reproduzindo um traco gravado. Se o defensor age diferente, o atacante evolui diferente. " +
  "Sem isso nao haveria adversario - haveria um video.";

NOTES.dataset =
  "[14:10 -> 15:10 | alvo 60 s]\n\n" +
  "Por que este dataset? Porque e contemporaneo, grande e real.\n\n" +
  "Coletado pelo Canadian Institute for Cybersecurity sobre um testbed de 105 dispositivos IoT fisicos. " +
  "33 tipos de ataque em 7 categorias - e reparem na linha logo abaixo: 33 ataques MAIS o trafego benigno somam 34 rotulos " +
  "no total. Vou usar os dois numeros adiante, entao fixem essa conta agora.\n\n" +
  "O detalhe que importa: os ataques sao disparados POR dispositivos IoT comprometidos contra outros dispositivos IoT. " +
  "Comportamento realista de botnet, nao trafego sintetico de laboratorio.\n\n" +
  "Cada linha e um registro de fluxo pre-agregado com 46 caracteristicas. Aplicamos um funil de selecao a prova de " +
  "vazamento - ajustado APENAS na particao de treino - removendo variancia zero, baixa variancia e colunas muito " +
  "correlacionadas. Sobram 29 caracteristicas.\n\n" +
  "-- A conta 33 + benigno = 34 foi um ponto de confusao na primeira versao. Diga-a explicitamente, uma vez, aqui.";

NOTES.projection =
  "[15:10 -> 16:20 | alvo 70 s]\n\n" +
  "Dois movimentos, e ambos sao decisoes de projeto auditaveis.\n\n" +
  "Primeiro movimento, cartao azul: um mapa deterministico projeta cada um dos 34 rotulos - os 33 ataques mais o benigno - " +
  "em exatamente um estagio da kill chain. Varreduras vao para RECON; forca bruta e injecao para ACCESS; spoofing e " +
  "preparacao de Mirai para MANEUVER; inundacoes DoS e DDoS para IMPACT. O efeito pratico e que cada estagio passa a ter " +
  "uma distribuicao empirica de caracteristicas reais, em vez de dados sinteticos.\n\n" +
  "Segundo movimento, cartao vermelho, o protocolo de particao: DEZ classes de ataque sao reservadas, no minimo duas por " +
  "estagio nao-benigno. Elas nunca sao vistas no treino - nem pelo detector, nem pelos agentes. " +
  "VulnerabilityScan, SqlInjection, DNS_Spoofing, DoS-SYN_Flood sao exemplos.\n\n" +
  "Guardem esse numero dez. Essas mesmas dez classes voltam no Ato IV como o nosso teste tipo zero-day. " +
  "E a disjuncao nao e uma promessa verbal: e asseverada por teste automatizado.\n\n" +
  "-- NAO fale de alfa nem de aliasing aqui (feedback do Arthur: foi neste slide que a metodologia do embaralhamento " +
  "vazou na primeira versao). Alfa e o proximo-proximo slide, e tem slide proprio.";

NOTES.overlap =
  "[16:20 -> 17:20 | alvo 60 s]\n\n" +
  "Este slide antecipa uma objecao legitima: 'voces fabricaram a ambiguidade para que o RL vencesse'.\n\n" +
  "Primeiro, o que e o grafico, porque isso nao ficou claro na primeira versao. Esta no cartao cinza a direita: " +
  "PC1 e PC2 sao componentes principais - as duas direcoes que carregam mais variancia. E uma forma de espremer 29 " +
  "caracteristicas num plano que a gente consegue olhar. Cada ponto e um fluxo real, colorido pelo estagio.\n\n" +
  "Agora leiam com calma: as inundacoes de IMPACT se separam com nitidez - contagem de flags e taxa as denunciam. " +
  "Mas BENIGN, RECON e ACCESS se interpenetram fortemente.\n\n" +
  "Ou seja: mesmo com informacao COMPLETA de caracteristicas, estagios vizinhos nao sao linearmente separaveis. " +
  "Nenhuma linha de fluxo isolada revela o estagio. A ambiguidade e propriedade do dado, nao suposicao que eu adicionei.\n\n" +
  "E uma honestidade que eu faco questao de declarar, e que consta nas limitacoes: o CICIoT2023 nao tem chave de sessao. " +
  "A coerencia de sessao e imposta por nos na camada do ambiente. E uma abstracao de modelagem, declarada como tal.";

NOTES.attacker =
  "[17:20 -> 18:50 | alvo 90 s]\n\n" +
  "Agora o atacante. E ele nao e um script gravado: ele reage a forca do defensor.\n\n" +
  "O mecanismo e um cabo de guerra sobre uma quantidade so: a folga de forca com sinal, d, que e a acao escolhida menos a " +
  "acao recomendada para o estagio verdadeiro. Repare na ironia, na linha central: o defensor nunca ve esse estagio - " +
  "ele precisa inferi-lo. Mas o atacante reage como se o defensor soubesse.\n\n" +
  "Tres regimes, os tres cartoes de cima.\n" +
  "Proporcional, d igual a zero: o atacante e empurrado de volta um estagio com probabilidade 0,90 - e 0,98 no caso de " +
  "ISOLATE. Dose certa, recuo.\n" +
  "Subdimensionado, d menor ou igual a menos um: o atacante avanca.\n" +
  "Superdimensionado, d maior ou igual a mais um: o atacante apenas se mantem - nao recua. E a recompensa continua cobrando " +
  "o custo de disponibilidade. Exagerar nao compra seguranca extra; so compra dano colateral.\n\n" +
  "Cartao vermelho embaixo: a escalada e acoplada a proximidade. Quanto mais fundo o pe na porta, mais forte ele empurra. " +
  "Nao existe orcamento fixo de intrusao; a pressao e endogena.\n\n" +
  "Cartao verde: prevencao e manter o atacante abaixo de IMPACT durante todo o horizonte de 100 passos. Um bloqueio isolado " +
  "nao basta - e preciso pressao sustentada.\n\n" +
  "-- Limite de escopo, se perguntarem: o atacante e reativo mas de forma fechada, NAO co-treinado. " +
  "Isso esta declarado nas limitacoes.";

NOTES.aliasing =
  "[18:50 -> 20:20 | alvo 90 s]\n\n" +
  "Como a observabilidade parcial e instrumentada, concretamente. Este e o slide do alfa.\n\n" +
  "Figura de cima, a maquina de estados. As setas solidas sao as transicoes do atacante que acabamos de ver. " +
  "As setas TRACEJADAS sao o aliasing: com probabilidade alfa, a linha de caracteristicas emitida vem de um estagio " +
  "ADJACENTE, e nao do verdadeiro.\n\n" +
  "Formalmente, no primeiro item a direita: o nucleo de observacao Z e uma mistura de dois componentes - " +
  "(1 menos alfa) do estagio proprio, mais alfa de estagios adjacentes.\n\n" +
  "E no cartao cinza, embaixo, o exemplo concreto que faltou na primeira versao: em alfa igual a 0,4, de cada dez linhas " +
  "que o defensor ve, cerca de QUATRO foram emitidas por um estagio vizinho. Ele esta olhando para uma varredura e sendo " +
  "mostrado um arrombamento - e vice-versa. E esse o nivel de ruido no canal de percepcao.\n\n" +
  "Terceiro item: a garantia de justica do experimento. O MESMO fluxo com aliasing alimenta todas as politicas. " +
  "Nenhum competidor recebe dado mais limpo que outro.\n\n" +
  "Figura de baixo, o que o agente de fato enxerga: as ultimas cinco linhas mais suas diferencas temporais, empilhadas em " +
  "um vetor de 290 dimensoes. Essa janela e a unica maquinaria de crenca que ele possui - nao ha rede recorrente, nao ha " +
  "filtro de Bayes explicito. Enquanto isso, a baseline le UMA linha.\n\n" +
  "Janela contra instantaneo. E essa a comparacao.";

NOTES.reward =
  "[20:20 -> 21:50 | alvo 90 s]\n\n" +
  "Esta e, na minha avaliacao, a decisao de projeto mais delicada da dissertacao.\n\n" +
  "ATENCAO AO VOCABULARIO: os dois nomes estao escritos nos cabecalhos. A da esquerda e a OUTCOME - a ESPARSA. " +
  "A da direita e a COUPLED - a MOLDADA. Use sempre os dois termos juntos; nao diga so 'esparsa'.\n\n" +
  "Comeco pela direita, a COUPLED, moldada. Existe uma forma tentadora de recompensar o agente: pagar a cada passo quando " +
  "ele escolhe a acao recomendada para o estagio verdadeiro. O problema e sutil e fatal: isso recompensa exatamente aquilo " +
  "que um classificador supervisionado prediz. Se eu treinasse so assim, eu nao conseguiria distinguir 'aprendeu a defender' " +
  "de 'imitou uma tabela de consulta'. Eu estaria entregando o rotulo de estagio pela porta dos fundos.\n\n" +
  "Por isso a PRIMARIA e a OUTCOME, esparsa, a esquerda. So ha custo de acao, contabilidade terminal e bonus de prevencao: " +
  "episodio prevenido rende +50; defesa terminal bem-sucedida +250; comprometimento -200, e mais -150 se o agente estava " +
  "passivo no impacto. Nenhuma dica de estagio a cada passo.\n\n" +
  "Isso torna a atribuicao de credito muito mais dificil - a recompensa chega cerca de cem passos depois da decisao que a " +
  "causou. Mas ela mede DEFESA, e nao imitacao.\n\n" +
  "A COUPLED e mantida exclusivamente como a ablacao que responde a objecao do sinal privilegiado - e voces vao ve-la no " +
  "Ato IV.\n\n" +
  "O principio de projeto esta na faixa inferior: se a recompensa entrega o rotulo do estagio, o classificador vence por " +
  "construcao.\n\n" +
  "-- Constantes completas: BACKUP B1. Nao as recite aqui.";

NOTES.contenders =
  "[21:50 -> 22:50 | alvo 60 s]\n\n" +
  "Quem entra na comparacao. Quatro linhas.\n\n" +
  "Primeira, os agentes que aprendem: PPO, A2C e DQN. Leem a janela de 290 dimensoes, nunca veem o estagio verdadeiro e " +
  "nunca consomem o detector.\n\n" +
  "Segunda, a baseline implantavel, RF-Acting - e eu faco questao de defende-la, porque a forca do meu resultado depende da " +
  "forca dela. RandomForest com hiperparametros ajustados por busca em grade, macro-F1 de 0,924. Nao e um espantalho: e o " +
  "classificador mais forte que um profissional levaria para producao. E o mecanismo e o que eu mostrei no slide da pergunta: " +
  "classifica o estagio de uma linha e aplica a regra de acao. Sem memoria.\n\n" +
  "Terceira, as triviais: sempre-observar, sempre-bloquear e aleatoria. Elas delimitam a escala de recompensa.\n\n" +
  "Quarta, o ORACULO, na linha dourada, que exige cuidado: ele usa a mesma regra de acao, mas LENDO o estagio verdadeiro. " +
  "Ele nao e um competidor - e um instrumento de medida. Ele precifica quanto vale a percepcao perfeita: um teto de +194,8.\n\n" +
  "-- Evidencia de que o RF nao e espantalho: BACKUP B4, com a matriz de confusao. Otimo slide para puxar em pergunta.";

NOTES.protocol =
  "[22:50 -> 23:30 | alvo 40 s]\n\n" +
  "Slide de rigor. Rapido, mas com firmeza - ele sustenta todos os numeros seguintes.\n\n" +
  "Dez sementes por algoritmo, orcamento fixo de 5 milhoes de passos, sem parada antecipada. Essa escolha foi deliberada: " +
  "uma regra de parada calibrada para um algoritmo trunca outro - e, no nosso caso, foi justamente rodar o orcamento inteiro " +
  "que expos a instabilidade do DQN.\n\n" +
  "Na avaliacao: 300 episodios por politica, intervalos de confianca de 95% por bootstrap, e o criterio de separacao e " +
  "intervalos DISJUNTOS - que e conservador, mais exigente que um teste de medias.\n\n" +
  "Tudo em CPU. E a reprodutibilidade: cada figura acompanha um manifesto com hashes e o commit; 462 testes; " +
  "a cadeia inteira se reverifica num clone limpo.\n\n" +
  "A faixa inferior e a garantia experimental: mesmo fluxo de observacoes, mesmas particoes, mesma recompensa para todos. " +
  "A UNICA variavel e o controlador.\n\n" +
  "-- 40 segundos. Voce esta entrando nos resultados; nao perca tempo aqui.";

// ===========================================================================
// ACT IV — RESULTS — slides 21-27 — 23:30 -> 34:00
// ===========================================================================

NOTES.learning =
  "[23:30 -> 24:40 | alvo 70 s]\n\n" +
  "Primeiro resultado. Curvas de aprendizado sob a recompensa OUTCOME, esparsa, em alfa igual a 0,4 - o nosso ponto de " +
  "referencia.\n\n" +
  "Na figura, a faixa cinza tracejada e a referencia do RF-Acting; a linha traco-ponto preta e o teto do oraculo.\n\n" +
  "A leitura esta nos tres cartoes a direita. Os dois on-policy saem do regime negativo e estabilizam: A2C chega a +138,7 " +
  "com desvio entre sementes de cerca de 9 - o mais consistente de todos; PPO chega a +121,3 com desvio de cerca de 15.\n\n" +
  "O DQN, off-policy, desestabiliza sem o sinal moldado: melhor checkpoint +72,5, mas desvio de cerca de 52. " +
  "Ou seja: dependendo da semente, voce recebe um defensor razoavel ou um defensor pior do que nao fazer nada.\n\n" +
  "O achado, com formulacao cuidadosa, esta na linha final: a vantagem on-policy aqui e CONFIABILIDADE DE TREINAMENTO, " +
  "e nao retorno de pico.\n\n" +
  "E note que nos reportamos os tres algoritmos, inclusive o instavel. Um trabalho que mostrasse so o melhor esconderia " +
  "justamente a informacao util para quem for implantar.\n\n" +
  "-- Fecha o OBJETIVO 5 (confiabilidade). Diga isso: 'este e o objetivo cinco'.";

NOTES.doctrines =
  "[24:40 -> 25:40 | alvo 60 s]\n\n" +
  "Este e o meu slide favorito para ensinar, porque o resultado e qualitativo e, confesso, nao foi antecipado.\n\n" +
  "Os algoritmos nao diferiram apenas em pontuacao: eles aprenderam FILOSOFIAS DE DEFESA diferentes. " +
  "A figura mostra a distribuicao de acoes por estagio - linhas sao os algoritmos, colunas vao de BENIGN a IMPACT.\n\n" +
  "Cartao de cima: o A2C aprendeu o que eu chamo de 'prevenir na manobra'. Ele bloqueia 84,4% dos passos em MANEUVER, " +
  "sufoca o avanco no meio da cadeia, e nunca recorre a ISOLATE. Ele estrangula a campanha antes que ela chegue ao fim.\n\n" +
  "Cartao de baixo: o PPO aprendeu 'conter no impacto'. Tolera penetracao mais profunda e age decisivamente no estagio final.\n\n" +
  "Duas doutrinas legitimas, com perfis de risco diferentes - e NENHUMA delas foi programada. Ambas emergiram da mesma " +
  "recompensa esparsa.\n\n" +
  "E ambas sao seguras no trafego legitimo: acao agressiva em menos de 1% dos fluxos benignos.\n\n" +
  "Para um operador isso nao e detalhe estetico: significa que a politica e inspecionavel como estrategia, e nao apenas " +
  "como um escore de caixa-preta.";

NOTES.crossover =
  "[25:40 -> 27:40 | alvo 120 s]  *** SLIDE PRINCIPAL DA DEFESA ***\n\n" +
  "Este e O slide. Se voce so tiver tempo para um resultado, e este. Va com calma.\n\n" +
  "No eixo horizontal, a taxa de aliasing alfa - o nosso botao de ambiguidade. No vertical, a recompensa media por episodio, " +
  "com intervalos de 95%.\n\n" +
  "Quatro movimentos, e eles estao nos cartoes a direita.\n\n" +
  "MOVIMENTO UM - a ancora de honestidade, em alfa igual a zero. PPO faz +138,6; RF-Acting faz +136,5. Empate estatistico, " +
  "intervalos sobrepostos. E eu quero que este seja o ponto mais lembrado da apresentacao: o ambiente NAO favorece RL por " +
  "construcao. Quando a tarefa realmente e classificacao por fluxo, o classificador empata. Se eu tivesse montado um " +
  "ambiente enviesado, o RL ja venceria aqui - e nao vence.\n\n" +
  "MOVIMENTO DOIS - o RF-Acting degrada monotonicamente conforme alfa cresce, ate -29,3 em alfa igual a 1. " +
  "Reparem no SINAL: negativo. Sob ambiguidade total o classificador nao e apenas inutil - ele e ativamente prejudicial, " +
  "porque age com confianca sobre inferencia errada, causando dano de disponibilidade sem entregar prevencao.\n\n" +
  "MOVIMENTO TRES - o PPO em janela permanece PLANO. A janela absorve a ambiguidade: quando uma observacao isolada engana, " +
  "a sequencia recente ainda carrega sinal.\n\n" +
  "MOVIMENTO QUATRO - a separacao. A partir de alfa igual a 0,4 os intervalos ficam DISJUNTOS: diferenca de +26,9, " +
  "crescendo ate +161,2.\n\n" +
  "E o oraculo, plano em +194,8, precifica a percepcao perfeita - mostrando que ainda ha margem, e que nos nao estamos " +
  "alegando ter resolvido o problema.\n\n" +
  "Em uma frase: quando a observacao se torna ambigua, a classificacao sem memoria colapsa; o controle em janela, nao.\n\n" +
  "-- Fecha o OBJETIVO 2. Diga em voz alta: 'este e o objetivo dois, e e o resultado principal da dissertacao'.";

NOTES.coupling =
  "[27:40 -> 28:50 | alvo 70 s]\n\n" +
  "Agora eu enfrento de frente a objecao do sinal privilegiado - a objecao mais forte que se pode fazer a este trabalho.\n\n" +
  "O procedimento: retreinar tudo sob as DUAS recompensas, e pontuar o RF-Acting sob a mesma recompensa em cada caso. " +
  "Comparacao pareada, sem vantagem para nenhum lado.\n\n" +
  "Cartao de cima, a COUPLED, moldada: o melhor agente e o DQN, com +226,2. Ele prospera com sinal denso - exatamente o que " +
  "a teoria preve para metodos off-policy baseados em valor. A diferenca RF menos melhor-RL e -63,1: o RL lidera.\n\n" +
  "Cartao de baixo, a OUTCOME, esparsa: o melhor agente e o A2C, com +146,1, e a diferenca e -63,0. Praticamente identica.\n\n" +
  "Duas licoes, independentes.\n" +
  "Primeira: a separacao entre RL e RF NAO depende do sinal moldado privilegiado. A objecao esta respondida com dado, " +
  "nao com argumento.\n" +
  "Segunda, talvez mais interessante para quem for implantar: a recompensa muda QUAL algoritmo vence. A vitoria do DQN sob " +
  "sinal denso evapora sob esparsidade - ele cai para -8,6. A mesma maquinaria de replay que explora sinal denso falha na " +
  "atribuicao de credito esparsa.\n\n" +
  "Como esta na faixa: a moldagem muda quem vence; nao muda SE o controle aprendido lidera.\n\n" +
  "-- Fecha o OBJETIVO 3.";

NOTES.sweeps =
  "[28:50 -> 30:00 | alvo 70 s]   [PODE ENCURTAR: se estiver depois de 30:00, apresente so o painel da ESQUERDA " +
  "e passe adiante]\n\n" +
  "Dois testes de estresse fora da distribuicao de treino. O detalhe metodologico que importa: as politicas sao FIXAS - " +
  "treinadas uma vez e nunca retreinadas por condicao. E assim que se mede robustez, e nao capacidade de reajuste.\n\n" +
  "A ESQUERDA, dificuldade do ambiente: nos reduzimos a probabilidade de recuo, ou seja, acoes proporcionais passam a " +
  "empurrar o atacante de volta com menos frequencia. A resposta e monotona, sem inversao de ordenacao - bom sinal de que " +
  "o ambiente e bem-comportado. E o A2C lidera em TODAS as dificuldades. A doutrina de 'prevenir na manobra' e a que perde " +
  "menos terreno exatamente onde aperta.\n\n" +
  "A DIREITA, evasao: o atacante endurece contra a expulsao depois de sentir a forca do defensor. " +
  "O criterio foi PRE-REGISTRADO, antes de rodar: o limite inferior do intervalo nao pode cair mais que 25% da media sem " +
  "evasao. O A2C passa, de +142,6 para +112,7. O PPO fica ligeiramente abaixo do criterio.\n\n" +
  "A conclusao e degradacao graciosa, nao colapso - e uma separacao real entre A2C e PPO que so aparece sob estresse.\n\n" +
  "-- Feedback [24]: 'e possivel selecionar alguns resultados'. Este slide e o primeiro candidato a corte parcial.";

NOTES.ood =
  "[30:00 -> 31:20 | alvo 80 s]\n\n" +
  "A sonda tipo zero-day. E a primeira coisa que eu preciso deixar clara, porque gerou duvida na primeira versao: " +
  "sao as MESMAS dez classes que eu reservei la no slide da projecao. Nao aumentou, nao mudou. Sao aquelas dez, " +
  "que nunca entraram no treino - nem do detector, nem dos agentes.\n\n" +
  "A metrica esta definida no cartao cinza, e tambem gerou duvida: TAXA DE PREVENCAO e a fracao de episodios em que o " +
  "atacante nunca alcanca IMPACT, durante os 100 passos. Nao e acuracia. Nao e deteccao. E o desfecho que interessa a um " +
  "operador: o ataque foi contido ou nao.\n\n" +
  "Na grade a esquerda: linhas sao politicas, colunas sao as classes. O melhor RL em janela, o A2C, previne entre 0,71 e " +
  "0,85 em TODAS as classes. O RF-Acting fica entre 0,00 e 0,15. A vantagem vai de +0,70 a +0,78 - e nenhuma classe " +
  "apresenta vantagem negativa.\n\n" +
  "Duas honestidades que eu nao quero que fiquem por conta da banca perguntar.\n" +
  "Primeira, e esta na nota de rodape: a politica sempre-bloquear 'previne' 1,0 em tudo. Mas ela consegue isso colocando " +
  "100% do trafego legitimo em quarentena. E operacionalmente inadmissivel. A fronteira que importa e prevenir MANTENDO-SE " +
  "seguro no trafego benigno - e so os agentes aprendidos vivem nessa fronteira.\n" +
  "Segunda: as taxas absolutas sao moderadas. A afirmacao que eu faco e sobre a vantagem RELATIVA, e nao sobre seguranca " +
  "quase perfeita.\n\n" +
  "-- Fecha o OBJETIVO 4.";

NOTES.recall =
  "[31:20 -> 32:40 | alvo 80 s]   [OPCIONAL - PULE ESTE SLIDE se o relogio passou de 32:00. " +
  "Ele responde uma objecao sofisticada que talvez ninguem faca; se pular, o argumento do slide anterior continua de pe.]\n\n" +
  "Por que o resultado anterior importa cientificamente? Porque ele permite testar a versao mais sofisticada da objecao do " +
  "cetico: 'o RL so vence onde o detector e cego; voces escolheram classes dificeis'.\n\n" +
  "Se isso fosse verdade, haveria uma assinatura obrigatoria nos dados: a vantagem do RL deveria ENCOLHER conforme o recall " +
  "do detector aumenta. E uma predicao falsificavel, e nos a testamos.\n\n" +
  "No grafico: eixo horizontal, o recall do RandomForest por classe; vertical, a vantagem de prevencao do RL. " +
  "As dez classes cobrem recall de 0,20 a 0,998 - de quase cego a praticamente perfeito.\n\n" +
  "O resultado: nenhuma tendencia detectavel. Spearman 0,22 com p de 0,54. Pearson -0,02 com p de 0,95. " +
  "E o intervalo da inclinacao contem o zero.\n\n" +
  "A ressalva, que eu faco questao de dizer ANTES de qualquer pergunta: n = 10 classes. Isso e ausencia de tendencia " +
  "detectavel, nao prova de independencia. Mas o ponto logico permanece: a tendencia que a objecao EXIGE - negativa - " +
  "esta ausente.\n\n" +
  "E qual e o mecanismo, entao? Ele e estrutural, nao perceptual, e esta no ultimo item: o RF-Acting, mesmo classificando " +
  "corretamente, age de forma passiva em cerca de dois tercos dos passos; deixa o atacante alcancar o impacto e entao " +
  "bloqueia tarde. Ele MITIGA; ele nunca PREVINE. Um classificador de um disparo e estruturalmente incapaz de expressar " +
  "controle temporal.";

// ===========================================================================
// ACT V — CLOSING — slides 28-31 — 32:40 -> 40:00
// ===========================================================================

NOTES.limitations =
  "[32:40 -> 34:10 | alvo 90 s]\n\n" +
  "Limitacoes, declaradas - e eu prefiro apresenta-las eu mesmo, antes que a banca precise pedir.\n\n" +
  "Um: abstracoes de modelagem. Coerencia de sessao e aliasing sao construtos da camada de ambiente. O dataset nao traz " +
  "chave de sessao, e alfa e um botao controlado, nao uma propriedade medida em producao. Por isso a contribuicao e o " +
  "FORMATO da resposta ao longo de alfa, ancorado no empate em alfa igual a zero - e nao um numero absoluto de campo.\n\n" +
  "Dois: a vantagem e condicional. Em alfa igual a zero eu afirmo EMPATE, e nada alem disso.\n\n" +
  "Tres: o atacante e projetado, reativo, mas nao co-treinado. O cruzamento e condicional a essa classe de atacante. " +
  "Self-play fica como trabalho futuro.\n\n" +
  "Quatro: a baseline e sem memoria por construcao. O benchmark isola controle em janela contra controle por fluxo. " +
  "Uma baseline supervisionada COM janela e o fortalecimento mais direto, e esta nomeada como trabalho seguinte.\n\n" +
  "Cinco: um unico dataset. Replicacao no Bot-IoT e necessaria antes de qualquer alegacao de independencia.\n\n" +
  "Seis: o estudo OOD e um teste de estresse por injecao, com n = 10 - nao e evidencia de generalizacao a zero-day em " +
  "producao.\n\n" +
  "-- Este slide compra credibilidade. Nao o apresse, e nao peca desculpas por nenhum item.";

NOTES.conclusions =
  "[34:10 -> 35:30 | alvo 80 s]\n\n" +
  "Os tres achados, uma respiracao cada.\n\n" +
  "PRIMEIRO: o cruzamento e real e e controlado. Empate em alfa igual a zero, intervalos disjuntos a partir de 0,4, " +
  "e o RF chegando a ser prejudicial em alfa igual a 1 enquanto o PPO em janela permanece plano. " +
  "A contribuicao nao e 'o RL venceu' - e ter LOCALIZADO a fronteira onde a classificacao deixa de bastar.\n\n" +
  "SEGUNDO: nao e artefato da recompensa. O melhor agente lidera sob as duas, com diferencas de -63,1 e -63,0. " +
  "A moldagem muda qual algoritmo vence; nao muda quem lidera.\n\n" +
  "TERCEIRO: estende-se a classes nunca vistas. Vantagem de +0,70 a +0,78 em todas as dez classes retidas, " +
  "sem dependencia detectavel do recall. O mecanismo e controle temporal, e nao ponto cego do detector.\n\n" +
  "E o complemento operacional, na faixa: menos de 1% de perturbacao no trafego legitimo, uma politica pequena o " +
  "suficiente para um gateway de borda, e reprodutibilidade de ponta a ponta.\n\n" +
  "A linha que eu quero deixar: a contribuicao nao e 'RL vence' - e um relato controlado de QUANDO e POR QUE ele vence.\n\n" +
  "-- Feedback [28]: NAO detalhe os numeros de footprint aqui. Uma clausula basta. " +
  "Se perguntarem, va ao BACKUP B7.";

NOTES.future =
  "[35:30 -> 36:30 | alvo 60 s]\n\n" +
  "Trabalho futuro, em tres grupos - as tres colunas.\n\n" +
  "Coluna um, fortalecer a afirmacao: a baseline supervisionada com janela, que e o fortalecimento mais direto; " +
  "politicas recorrentes com estado de crenca - e eu registro que um ensaio recorrente preliminar NAO superou a janela sob " +
  "este orcamento, o que merece ser caracterizado; e um segundo dataset, o Bot-IoT.\n\n" +
  "Coluna dois, adversarios mais dificeis: atacante co-adaptativo em self-play; cadeias nao monotonicas com salto de " +
  "estagio; e defesa cooperativa multiagente.\n\n" +
  "Coluna tres, rumo a implantacao: quantificacao de custo em hardware de borda; treinamento federado multissitio; " +
  "e garantias formais de falso positivo via MDP com restricoes.\n\n" +
  "E na faixa inferior, disseminacao: um artigo condensando este trabalho foi submetido ao periodico Internet of Things, " +
  "da Elsevier. O codigo e os manifestos estao publicos no GitHub.";

NOTES.thanks =
  "[36:30 -> 37:00 | alvo 30 s]  --> deixa ~3 min de folga antes dos 40:00\n\n" +
  "Muito obrigado.\n\n" +
  "Agradeco ao Prof. Denis Fantinato pela orientacao ao longo de todo o percurso, e aos Profs. Alexandre Simoes e " +
  "Joao Kleinschmidt por aceitarem compor a banca e pelo tempo dedicado a leitura.\n\n" +
  "Agradeco tambem aos colegas da linha de pesquisa e a minha familia.\n\n" +
  "Estou a disposicao para as perguntas. Tenho slides de apoio com detalhes de recompensa, hiperparametros, mapeamento de " +
  "rotulos, desempenho do detector e footprint do modelo, caso sejam uteis.\n\n" +
  "-- Depois desta frase, PARE de falar. Nao preencha o silencio.";

// ===========================================================================
// BACKUP — not part of the timed talk
// ===========================================================================

NOTES.backupDivider =
  "[FORA DO TEMPO] Divisor. Material de apoio para perguntas.\n\n" +
  "MAPA RAPIDO DO BACKUP:\n" +
  "  B1 constantes de recompensa   -> 'de onde vem +250 / -200?'\n" +
  "  B2 hiperparametros            -> 'como escolheu os hiperparametros?'\n" +
  "  B3 mapeamento 34 -> 5         -> 'como voce mapeou os rotulos?'\n" +
  "  B4 detector RandomForest      -> 'sua baseline e fraca?'  (a mais provavel)\n" +
  "  B5 seguranca em trafego benigno -> 'e o falso positivo?'\n" +
  "  B6 reprodutibilidade          -> 'como sei que isso reproduz?'\n" +
  "  B7 footprint do modelo        -> 'cabe mesmo num gateway?'";

NOTES.backupReward =
  "[BACKUP B1] Tabela completa de constantes (Tabela 3.5 da metodologia).\n\n" +
  "A calibracao segue quatro principios. Os resultados terminais definem a escala: +250 defesa bem-sucedida, " +
  "-200 comprometimento. A moldagem por passo e uma ordem de grandeza menor, mais ou menos 5, para nunca dominar o desfecho. " +
  "Existem tetos nos componentes acumulaveis - 150 e 100 - para impedir farmar recompensa. " +
  "E as salvaguardas em benigno, -50 por exagero e -100 por bloqueio, garantem que bloquear demais nunca seja lucrativo.\n\n" +
  "A recompensa OUTCOME, esparsa, remove os cinco componentes condicionados ao estagio; restam custo de acao, " +
  "contabilidade terminal e prevencao.";

NOTES.backupHparams =
  "[BACKUP B2] Do Apendice C - todos obtidos por busca em grade por algoritmo.\n\n" +
  "Dois pontos merecem destaque. O A2C usa n_steps = 256, contra o padrao de 5 da biblioteca: rollouts longos sao " +
  "necessarios por causa do atraso de credito na recompensa esparsa. E o DQN usa buffer de 200 mil transicoes com " +
  "atualizacao lenta do alvo.\n\n" +
  "A rede e a mesma nos tres - MLP de duas camadas de 64 unidades - precisamente para que a comparacao isole a regra de " +
  "aprendizado.\n\n" +
  "-- Se a pergunta for 'o DQN nao recebeu mais dados por causa do replay?': sim, e proposital. O que se controla e o " +
  "orcamento de INTERACAO com o ambiente. O reaproveitamento por replay e parte da definicao do algoritmo, nao uma " +
  "vantagem injusta - e, sob recompensa esparsa, ele acabou sendo uma DESvantagem.";

NOTES.backupMapping =
  "[BACKUP B3] Do Apendice B - o mapeamento completo.\n\n" +
  "Sao 34 rotulos: 33 classes de ataque mais o trafego benigno. As estrelas marcam as dez classes reservadas, com pelo " +
  "menos duas por estagio nao-benigno - o que permite sondar a cadeia inteira, e nao so um estagio.\n\n" +
  "Uma decisao de engenharia relevante: rotulo desconhecido levanta erro rigido no codigo, em vez de cair num 'outros'. " +
  "Isso elimina a possibilidade de vazamento silencioso.";

NOTES.backupDetector =
  "[BACKUP B4] Evidencia de que a comparacao e justa. ESTE E O SLIDE MAIS PROVAVEL DE SER PEDIDO.\n\n" +
  "Busca em grade de 54 combinacoes sobre macro-F1 de validacao, com otimo interior: 200 arvores, profundidade 20, " +
  "pesos balanceados. O F1 de validacao e plano em 0,927 ao longo da contagem de arvores, o que mostra que ele nao esta " +
  "subajustado - mais capacidade nao ajudaria.\n\n" +
  "No teste balanceado retido: macro-F1 de 0,924; pior classe 0,87, justamente na ambiguidade recon/access; " +
  "e IMPACT praticamente perfeito. A unica confusao material e RECON classificado como ACCESS, em 13,2% - " +
  "que e exatamente a sobreposicao intrinseca do meio da cadeia.\n\n" +
  "-- Frase de fecho: 'a baseline nao perde por ser fraca; ela perde por ser sem memoria'.";

NOTES.backupBenign =
  "[BACKUP B5] O eixo de disponibilidade.\n\n" +
  "Aqui a taxa de falso positivo e a fracao de fluxos legitimos que recebem bloqueio ou isolamento. " +
  "Os agentes aprendidos: PPO 0,89%, A2C 0,66%, DQN 0,46% - todos abaixo do limiar operacional de 1%. " +
  "A politica aleatoria fica em 41,3%, e sempre-bloquear em 100% - que e precisamente por que a prevencao perfeita dela " +
  "e inadmissivel. Sempre-observar tem 0% de falso positivo, mas perde todos os episodios.\n\n" +
  "As triviais delimitam o compromisso; os agentes aprendidos sao as unicas politicas simultaneamente fortes e seguras.";

NOTES.backupRepro =
  "[BACKUP B6] Reprodutibilidade.\n\n" +
  "Cada figura vem acompanhada de um manifest.json com o SHA-256 de cada artefato de entrada, o commit do git que a " +
  "produziu e o comando exato. Uma rotina de verificacao repercorre a cadeia inteira num clone limpo.\n\n" +
  "Os numeros do texto sao gerados por macros a partir dos JSONs canonicos - nenhum numero foi digitado a mao, " +
  "o que elimina uma classe inteira de erro. Sao 462 testes.\n\n" +
  "O ponto pratico: clonar, rodar os testes e rodar a verificacao resulta em aprovacao SEM necessidade de retreinar nada.";

NOTES.backupFootprint =
  "[BACKUP B7] Footprint do modelo - movido para o backup a pedido do orientador.\n\n" +
  "A politica treinada tem cerca de 90 KB em fp32 e 23 mil parametros - um MLP de duas camadas de 64 unidades. " +
  "O RandomForest ajustado tem 181 MB e 1,7 milhao de nos. A razao e de aproximadamente 1956 vezes.\n\n" +
  "O ponto que importa nao e o numero, e a assimetria arquitetural: o artefato implantavel e a POLITICA, nao o detector. " +
  "O RandomForest so e necessario para a baseline. Um defensor que nunca consulta classificador nao tem nada grande " +
  "para embarcar - e isso e o que o coloca no orcamento de um gateway de borda.\n\n" +
  "-- Numeros gerados por script a partir dos checkpoints reais, nunca digitados a mao.";

module.exports = { NOTES };
