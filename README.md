# Trabalho 2 - Agente Neural com Colônia de Abelhas

## Identificação
- **Matrícula:** 2025130736 (termina em 6)
- **Metaheurística:** Colônia de Abelhas (ABC)
- **Classificador:** Rede Neural
- **Jogo:** Space Invaders Simplificado

## 📋 Conformidade com Requisitos

### Implementação conforme Item 2
✅ **Baseado nos códigos fornecidos:**
- `clear_game/` - Código base do jogo
- `game_with_sample_agent/` - Exemplo com algoritmo genético

✅ **Agente implementado em `agents.py`:**
```
clear_game/game/agents.py
```

✅ **Trabalho dentro de `clear_game/` (pasta raiz dos fontes):**
- `clear_game/abc_neural_training.py` - Sistema de treinamento
- `clear_game/evaluation_30_runs.py` - Avaliação com 30 execuções
- `clear_game/create_plots.py` - Geração de gráficos
- `clear_game/main_projeto.py` - Script principal

✅ **Arquivos do jogo não alterados:** `config.py` e `core.py` mantidos inalterados.

✅ **Sem bibliotecas externas:** 
- Rede Neural: implementada do zero
- ABC: implementado do zero
- Apenas numpy, matplotlib e scipy (para testes estatísticos)

## Estrutura do Projeto

```
ia-trab-2/
├── clear_game/                    # 📁 PASTA RAIZ DOS FONTES
│   ├── game/
│   │   ├── agents.py              # ✅ NeuralNetworkAgent implementado
│   │   ├── config.py              # ✅ Não alterado
│   │   └── core.py                # ✅ Não alterado
│   │
│   ├── abc_neural_training.py     # ✅ Treinamento ABC
│   ├── evaluation_30_runs.py      # ✅ Avaliação com 30 execuções
│   ├── create_plots.py            # ✅ Gráficos de evolução e boxplots
│   ├── watch_agent.py             # ✅ Visualização do agente jogando
│   ├── main_projeto.py            # ✅ Script principal
│   ├── *.npy                      # Pesos treinados salvos
│   ├── *.pkl                      # Histórico de treinamento
│   └── *.png                      # Gráficos gerados
│
├── game_with_sample_agent/        # ✅ Exemplo fornecido (não alterado)
├── artigo_trabalho2.tex           # ✅ Artigo acadêmico (Item 4)
├── ARTIGO_README.md               # ✅ Instruções para compilar artigo
└── README.md                      # ✅ Esta documentação
```

## Como Executar

### 1. Entrar na pasta raiz dos fontes
```bash
cd clear_game/
```

### 2. Teste Rápido (Verificar Implementação)
```bash
python3 main_projeto.py --quick-test
```

### 3. Execução Completa
```bash
python3 main_projeto.py
```

### 4. Executar Apenas Treinamento
```bash
python3 abc_neural_training.py
```

### 5. Executar Apenas Avaliação
```bash
python3 evaluation_30_runs.py
```

### 6. Executar Apenas Gráficos
```bash
python3 create_plots.py
```

### 6. Interface Visual (Opcional)

#### Jogar Manualmente
```bash
cd clear_game
python3 human_play.py
```
**Controles:**
- ↑ (seta para cima): mover para cima
- ↓ (seta para baixo): mover para baixo
- Fechar janela: sair do jogo

#### Ver Agente Neural Jogando
```bash
python3 watch_agent.py
```

Este script automaticamente:
- ✅ Carrega os pesos treinados (se disponíveis)
- ✅ Mostra a grade sensorial do agente
- ✅ Exibe o score em tempo real
- ✅ Funciona com pesos aleatórios se não houver treinamento

## Requisitos Atendidos

### ✅ Implementação Técnica
- **Agente Neural:** Implementado em `clear_game/game/agents.py`
- **Interface Agent:** Método `predict()` conforme especificado
- **Arquitetura:** 27 → 32 → 16 → 3 neurônios (1.475 parâmetros)

#### 🧮 Detalhamento dos 1.475 Parâmetros:
- **Entrada (27):** Grade 5×5 (25) + posição Y + velocidade (2)
- **Camada 1:** 27→32 = 864 pesos + 32 biases = **896 parâmetros**
- **Camada 2:** 32→16 = 512 pesos + 16 biases = **528 parâmetros**  
- **Camada 3:** 16→3 = 48 pesos + 3 biases = **51 parâmetros**
- **Total:** 896 + 528 + 51 = **1.475 parâmetros otimizados pelo ABC**
- **Estado:** 27 elementos (grade 5×5 + 2 variáveis internas)
- **Ações:** 0=noop, 1=cima, 2=baixo

### ✅ Algoritmo ABC
- **Baseado no artigo:** "Improved ABC Algorithm" (Kiran & Babalik, 2014)
- **População:** 100 abelhas (conforme requisitos)
- **Fases:** Inicialização, Employed Bees, Onlooker Bees, Scout Bees
- **Critérios de parada:** 1000 iterações OU 12 horas

### ✅ Avaliação
- **30 execuções** para cada agente
- **Testes estatísticos:** t-test e Wilcoxon conforme `scipy.stats`
- **Comparação:** Com baselines fornecidos nos requisitos
- **Nível de significância:** 95% (α = 0.05)

### ✅ Visualizações
- **Gráfico de evolução:** Iteração vs Melhor Pontuação
- **Boxplots:** Comparação entre agentes
- **Conforme exemplo:** Apêndice A dos requisitos

### ✅ Artigo Acadêmico (Item 4)
- **Arquivo:** `artigo_trabalho2.tex` (LaTeX)
- **Template:** Baseado em `elsarticle-template`
- **Extensão:** 12 páginas (expandido conforme novos requisitos)
- **Figuras:** Incluídas e explicadas (evolução + boxplots)
- **Tabelas:** Resultados experimentais detalhados
- **Análise:** Crítica aprofundada dos resultados e limitações
- **Compilação:** Instruções em `ARTIGO_README.md`

## Exemplo de Uso do Agente

```python
# Importar do local correto
import sys
sys.path.append('clear_game')

from game.core import SurvivalGame, GameConfig
from game.agents import NeuralNetworkAgent
import numpy as np

# Criar configuração do jogo
config = GameConfig(num_players=1)

# Criar agente neural
agent = NeuralNetworkAgent()

# Ou com pesos específicos
weights = np.load('best_neural_weights.npy')
agent = NeuralNetworkAgent(weights)

# Criar jogo
game = SurvivalGame(config=config, render=False)

# Loop de jogo conforme especificado nos requisitos
while not game.all_players_dead():
    actions = []
    for idx in range(config.num_players):
        if game.players[idx].alive:
            state = game.get_state(idx, include_internals=True)
            action = agent.predict(state)
            actions.append(action)
        else:
            actions.append(0)
    
    game.update(actions)
    if game.render:
        game.render_frame()

print(f"Score final: {game.players[0].score}")
```

## 💾 Arquivos de Resultados

O treinamento salva automaticamente todos os resultados:

### Treinamento
- `best_neural_weights_YYYYMMDD_HHMMSS.npy` - Pesos otimizados da rede neural
- `training_history_YYYYMMDD_HHMMSS.pkl` - Histórico completo do ABC

### Avaliação  
- `results_table.txt` - Tabela com 30 execuções e testes estatísticos

### Gráficos
- `training_evolution.png` - Evolução do fitness por iteração
- `boxplot_comparison.png` - Boxplot de comparação entre agentes

**💡 Importante:** Os resultados ficam salvos permanentemente. Não é necessário treinar novamente!

## 🔍 Validação da Implementação

Execute o teste rápido para verificar se tudo está funcionando:

```bash
python3 main_projeto.py --quick-test
```

**Saída esperada:**
```
✓ Agente implementado em clear_game/game/agents.py
✓ Uso do jogo como biblioteca  
✓ Trabalho realizado fora da pasta game
✓ ABC e Rede Neural sem bibliotecas externas
✓ NeuralNetworkAgent criado com sucesso
✓ Implementação no agents.py: CORRETA
```

---

**🎯 Trabalho implementado conforme todos os requisitos do Item 2**
