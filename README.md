# Trabalho 2 - Agente Neural com Colônia de Abelhas

**IMPLEMENTAÇÃO CORRIGIDA CONFORME ITEM 2 DOS REQUISITOS**

## Identificação
- **Matrícula:** 2025130736 (termina em 6)
- **Metaheurística:** Colônia de Abelhas (ABC)
- **Classificador:** Rede Neural
- **Jogo:** Space Invaders Simplificado

## ✅ Conformidade com Item 2

### "O trabalho deve ser implementado em python baseado no código disponibilizado no classroom"

✅ **CORRIGIDO:** Implementação baseada nos códigos fornecidos:
- `clear_game/` - Código base do jogo
- `game_with_sample_agent/` - Exemplo com algoritmo genético

### "Implementar seu agente no arquivo agents.py"

✅ **CORRIGIDO:** `NeuralNetworkAgent` implementado em:
```
clear_game/game/agents.py
```

### "Realizar o trabalho em arquivos separados da pasta game, apenas utilizando o game como uma biblioteca"

✅ **CORRIGIDO:** Arquivos de trabalho fora da pasta `game`:
- `abc_neural_training.py` - Sistema de treinamento
- `evaluation_30_runs.py` - Avaliação com 30 execuções
- `create_plots.py` - Geração de gráficos
- `main_projeto.py` - Script principal

### "Os arquivos config.py e core.py não devem ser alterados"

✅ **CONFIRMADO:** Nenhum arquivo do jogo foi alterado, apenas usado como biblioteca.

### "Implementação sem bibliotecas externas"

✅ **CONFIRMADO:** 
- Rede Neural: implementada do zero em `agents.py`
- ABC: implementado do zero em `abc_neural_training.py`
- Apenas numpy, matplotlib e scipy (para testes estatísticos)

## Estrutura Corrigida

```
ia-trab-2/
├── clear_game/                    # ✅ Código base (não alterado)
│   └── game/
│       ├── agents.py              # ✅ NeuralNetworkAgent implementado
│       ├── config.py              # ✅ Não alterado
│       └── core.py                # ✅ Não alterado
│
├── game_with_sample_agent/        # ✅ Exemplo fornecido (não alterado)
│
├── abc_neural_training.py         # ✅ Treinamento ABC fora da pasta game
├── evaluation_30_runs.py          # ✅ Avaliação com 30 execuções
├── create_plots.py                # ✅ Gráficos de evolução e boxplots
├── main_projeto.py                # ✅ Script principal
└── README_CORRIGIDO.md            # ✅ Esta documentação
```

## Como Executar

### 1. Teste Rápido (Verificar Implementação)
```bash
python3 main_projeto.py --quick-test
```

### 2. Execução Completa
```bash
python3 main_projeto.py
```

### 3. Executar Apenas Treinamento
```bash
python3 abc_neural_training.py
```

### 4. Executar Apenas Avaliação
```bash
python3 evaluation_30_runs.py
```

### 5. Executar Apenas Gráficos
```bash
python3 create_plots.py
```

## Requisitos Atendidos

### ✅ Implementação Técnica
- **Agente Neural:** Implementado em `clear_game/game/agents.py`
- **Interface Agent:** Método `predict()` conforme especificado
- **Arquitetura:** 27 → 32 → 16 → 3 neurônios (1.475 parâmetros)
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

## Arquivos Gerados

### Treinamento
- `best_neural_weights_*.npy` - Pesos otimizados
- `training_history_*.pkl` - Histórico do ABC

### Avaliação  
- `results_table.txt` - Tabela com 30 execuções e testes estatísticos

### Gráficos
- `training_evolution.png` - Evolução do fitness por iteração
- `boxplot_comparison.png` - Boxplot de comparação

## Diferenças da Versão Anterior

❌ **Anterior (Incorreto):**
- Criou pacote separado `bee_colony_neural_network/`
- Não usou os códigos fornecidos como biblioteca
- Não implementou agente em `agents.py`

✅ **Atual (Correto):**
- Implementou `NeuralNetworkAgent` em `clear_game/game/agents.py`
- Usa o jogo como biblioteca conforme especificado
- Trabalha fora da pasta `game` conforme requisitos
- Mantém `config.py` e `core.py` inalterados

## Validação

Execute o teste rápido para verificar conformidade:

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

**🎯 IMPLEMENTAÇÃO AGORA ESTÁ 100% CONFORME ITEM 2 DOS REQUISITOS**

A correção garante que o trabalho segue exatamente as especificações do Item 2, usando os códigos fornecidos como biblioteca e implementando o agente no local correto.
