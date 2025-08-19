"""
Sistema de Treinamento ABC + Rede Neural
Trabalho 2 - Inteligência Artificial e Sistemas Inteligentes
Bayron Thiengo Quinelato - 2025130736

Conforme requisitos:
- Utiliza o jogo como biblioteca
- Implementa ABC sem bibliotecas externas
- Trabalha fora da pasta game
- Usa o NeuralNetworkAgent implementado em agents.py
"""

import numpy as np
import time
import sys
import os
import pickle
from datetime import datetime

from game.core import SurvivalGame, GameConfig
from game.agents import NeuralNetworkAgent


class ArtificialBeeColony:
    """
    Implementação do Algoritmo de Colônia de Abelhas (ABC)
    Baseado no artigo: "Improved Artificial Bee Colony Algorithm for Continuous Optimization Problems"
    """
    
    def __init__(self, 
                 colony_size: int = 100,
                 dimension: int = None,
                 bounds: tuple = (-5.0, 5.0),
                 limit: int = 500,
                 max_iterations: int = 1000,
                 max_time_hours: float = 12.0):
        """
        Inicializa o algoritmo ABC conforme requisitos.
        
        Args:
            colony_size: 100 abelhas (conforme especificado)
            dimension: Número de parâmetros da rede neural
            bounds: Limites dos pesos
            limit: Limite para abandono
            max_iterations: 1000 iterações máximo
            max_time_hours: 12 horas máximo
        """
        self.colony_size = colony_size
        self.dimension = dimension
        self.bounds = bounds
        self.limit = limit
        self.max_iterations = max_iterations
        self.max_time_hours = max_time_hours
        
        # Variáveis do algoritmo
        self.population = None
        self.fitness_values = None
        self.trial_counters = None
        self.best_solution = None
        self.best_score = 0.0
        self.iteration = 0
        self.start_time = None
        
        # Histórico para gráficos
        self.best_score_history = []
        self.mean_score_history = []
    
    def initialize_population(self):
        """Fase de Inicialização: Equação (1) do artigo ABC"""
        self.population = np.random.uniform(
            self.bounds[0], 
            self.bounds[1], 
            (self.colony_size, self.dimension)
        )
        self.fitness_values = np.zeros(self.colony_size)
        self.trial_counters = np.zeros(self.colony_size, dtype=int)
        
        print(f"População ABC inicializada: {self.colony_size} abelhas, {self.dimension} parâmetros")
    
    def calculate_fitness(self, objective_value: float) -> float:
        """
        Calcula fitness para MAXIMIZAÇÃO de score
        """
        return max(1e-12, float(objective_value))
    
    def employed_bees_phase(self, fitness_function):
        """Fase das Abelhas Trabalhadoras: Equação (2) do artigo"""
        for i in range(self.colony_size):
            # Gerar solução candidata
            candidate = self.population[i].copy()
            
            # Selecionar dimensão e vizinho aleatórios
            j = np.random.randint(0, self.dimension)
            k = np.random.randint(0, self.colony_size)
            while k == i:
                k = np.random.randint(0, self.colony_size)
            
            # Aplicar modificação: v_ij = x_ij + φ(x_ij - x_kj)
            phi = np.random.uniform(-1, 1)
            candidate[j] = self.population[i][j] + phi * (
                self.population[i][j] - self.population[k][j]
            )
            
            # Manter dentro dos limites
            candidate[j] = np.clip(candidate[j], self.bounds[0], self.bounds[1])
            
            # Avaliar candidato
            candidate_objective = fitness_function(candidate)
            candidate_fitness = self.calculate_fitness(candidate_objective)
            
            # Seleção gulosa
            if candidate_fitness > self.fitness_values[i]:
                self.population[i] = candidate
                self.fitness_values[i] = candidate_fitness
                self.trial_counters[i] = 0
                
                # Atualizar melhor global
                if candidate_fitness > self.best_score:
                    self.best_score = candidate_fitness
                    self.best_solution = candidate.copy()
            else:
                self.trial_counters[i] += 1
    
    def onlooker_bees_phase(self, fitness_function):
        """Fase das Abelhas Observadoras: Equação (4) do artigo"""
        # Calcular probabilidades de seleção
        total_fitness = np.sum(self.fitness_values)
        if total_fitness == 0:
            probabilities = np.ones(self.colony_size) / self.colony_size
        else:
            probabilities = self.fitness_values / total_fitness
        
        for i in range(self.colony_size):
            # Seleção por roleta
            selected_idx = np.random.choice(self.colony_size, p=probabilities)
            
            # Gerar candidato baseado na fonte selecionada
            candidate = self.population[selected_idx].copy()
            
            j = np.random.randint(0, self.dimension)
            k = np.random.randint(0, self.colony_size)
            while k == selected_idx:
                k = np.random.randint(0, self.colony_size)
            
            phi = np.random.uniform(-1, 1)
            candidate[j] = self.population[selected_idx][j] + phi * (
                self.population[selected_idx][j] - self.population[k][j]
            )
            
            candidate[j] = np.clip(candidate[j], self.bounds[0], self.bounds[1])
            
            # Avaliar e selecionar
            candidate_objective = fitness_function(candidate)
            candidate_fitness = self.calculate_fitness(candidate_objective)
            
            if candidate_fitness > self.fitness_values[selected_idx]:
                self.population[selected_idx] = candidate
                self.fitness_values[selected_idx] = candidate_fitness
                self.trial_counters[selected_idx] = 0
                
                if candidate_fitness > self.best_score:
                    self.best_score = candidate_fitness
                    self.best_solution = candidate.copy()
            else:
                self.trial_counters[selected_idx] += 1
    
    def scout_bees_phase(self, fitness_function):
        """Fase das Abelhas Exploradoras"""
        for i in range(self.colony_size):
            if self.trial_counters[i] >= self.limit:
                # Gerar nova solução aleatória
                self.population[i] = np.random.uniform(
                    self.bounds[0], self.bounds[1], self.dimension
                )

                objective = fitness_function(self.population[i])
                self.fitness_values[i] = self.calculate_fitness(objective)
                self.trial_counters[i] = 0
                
                # Verificar se é novo melhor global
                if self.fitness_values[i] > self.best_score:
                    self.best_score = self.fitness_values[i]
                    self.best_solution = self.population[i].copy()
    
    def should_stop(self) -> bool:
        """Critérios de parada: 1000 iterações OU 12 horas"""
        if self.iteration >= self.max_iterations:
            return True
        
        if self.start_time is not None:
            elapsed_hours = (time.time() - self.start_time) / 3600
            if elapsed_hours >= self.max_time_hours:
                return True
        
        return False
    
    def optimize(self, fitness_function, verbose=True):
        """Executa o algoritmo ABC completo"""
        self.start_time = time.time()
        self.iteration = 0
        
        if verbose:
            print("="*60)
            print("INICIANDO TREINAMENTO ABC + REDE NEURAL")
            print("="*60)
            print(f"Colônia: {self.colony_size} abelhas")
            print(f"Parâmetros: {self.dimension}")
            print(f"Limites: {self.bounds}")
            print(f"Max iterações: {self.max_iterations}")
            print(f"Max tempo: {self.max_time_hours}h")
            print("-"*60)
        
        # Inicializar população
        self.initialize_population()
        
        # Avaliar população inicial
        for i in range(self.colony_size):
            objective = fitness_function(self.population[i])
            self.fitness_values[i] = self.calculate_fitness(objective)
            
            if self.fitness_values[i] > self.best_score:
                self.best_score = self.fitness_values[i]
                self.best_solution = self.population[i].copy()
        
        # Loop principal ABC
        while not self.should_stop():
            iteration_start = time.time()
            
            # Três fases do ABC
            self.employed_bees_phase(fitness_function)
            self.onlooker_bees_phase(fitness_function)
            self.scout_bees_phase(fitness_function)
            
            # Registrar histórico
            mean_score = np.mean(self.fitness_values)
            self.mean_score_history.append(mean_score)
            self.best_score_history.append(self.best_score)
            
            # Progresso
            if verbose and (self.iteration % 10 == 0 or self.iteration < 5):
                elapsed = (time.time() - self.start_time) / 3600
                iter_time = time.time() - iteration_start
                
                print(f"Iter {self.iteration:4d} | "
                      f"Melhor: {self.best_score:.6f} | "
                      f"Média: {mean_score:.6f} | "
                      f"Tempo: {elapsed:.2f}h | "
                      f"({iter_time:.1f}s/iter)")
            
            self.iteration += 1
        
        # Resultado final
        elapsed_total = (time.time() - self.start_time) / 3600
        if verbose:
            print("-"*60)
            print(f"TREINAMENTO CONCLUÍDO:")
            print(f"Iterações: {self.iteration}")
            print(f"Tempo total: {elapsed_total:.2f}h")
            print(f"Melhor pontuação: {self.best_score:.6f}")
            
            criterio = "Tempo" if elapsed_total >= self.max_time_hours else "Iterações"
            print(f"Critério de parada: {criterio}")
            print("="*60)
        
        return self.best_solution, self.best_score


class GameFitnessEvaluator:
    """
    Avaliador de fitness usando o jogo como biblioteca
    """
    
    def __init__(self, num_games: int = 3):
        """
        Args:
            num_games: Número de jogos para avaliar cada agente
        """
        self.num_games = num_games
        self.config = GameConfig(num_players=1, fps=60)
    
    def evaluate_agent_weights(self, weights: np.ndarray) -> float:
        """
        Avalia um conjunto de pesos da rede neural
        """
        try:
            # Criar agente com os pesos
            agent = NeuralNetworkAgent(weights)
            
            total_score = 0.0
            
            for game_idx in range(self.num_games):
                # Criar jogo
                game = SurvivalGame(config=self.config, render=False)
                
                # Jogar até morrer
                while not game.all_players_dead():
                    # Obter estado (include_internals=True para ter 27 elementos)
                    state = game.get_state(0, include_internals=True)
                    
                    # Agente decide ação
                    action = agent.predict(state)
                    
                    # Atualizar jogo
                    game.update([action])
                
                # Somar score
                total_score += game.players[0].score
            
            # Retornar score médio
            return total_score / self.num_games
            
        except Exception as e:
            # Em caso de erro, penalizar
            print(f"Erro na avaliação: {e}")
            return 0.0


def main():
    """Função principal do treinamento"""
    print("TRABALHO 2 - IA E SISTEMAS INTELIGENTES")
    print("Matrícula: 2025130736")
    print("Metaheurística: Colônia de Abelhas (ABC)")
    print("Classificador: Rede Neural")
    print()
    
    # Criar agente neural para obter dimensões
    temp_agent = NeuralNetworkAgent()
    dimension = temp_agent.total_params
    
    print(f"Rede Neural:")
    print(f"  Arquitetura: {temp_agent.layer_sizes}")
    print(f"  Total de parâmetros: {dimension}")
    print()
    
    # Criar avaliador de fitness
    evaluator = GameFitnessEvaluator(num_games=3)
    fitness_function = evaluator.evaluate_agent_weights
    
    # Criar otimizador ABC conforme requisitos
    abc = ArtificialBeeColony(
        colony_size=100,        # Conforme requisitos
        dimension=dimension,
        bounds=(-5.0, 5.0),
        limit=500,
        max_iterations=1000,    # Conforme requisitos
        max_time_hours=12.0     # Conforme requisitos
    )
    
    # Executar treinamento
    best_weights, best_score = abc.optimize(fitness_function)
    
    # Salvar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Salvar pesos
    weights_file = f"best_neural_weights_{timestamp}.npy"
    np.save(weights_file, best_weights)
    print(f"\nMelhores pesos salvos: {weights_file}")
    
    # Salvar histórico
    history = {
        'best_score_history': abc.best_score_history,
        'mean_score_history': abc.mean_score_history,
        'final_score': best_score,
        'iterations': abc.iteration,
        'total_time_hours': (time.time() - abc.start_time) / 3600,
        'config': {
            'colony_size': abc.colony_size,
            'dimension': dimension,
            'bounds': abc.bounds,
            'limit': abc.limit,
            'max_iterations': abc.max_iterations,
            'max_time_hours': abc.max_time_hours
        },
        'timestamp': timestamp
    }
    
    history_file = f"training_history_{timestamp}.pkl"
    with open(history_file, 'wb') as f:
        pickle.dump(history, f)
    print(f"Histórico salvo: {history_file}")
    
    # Testar melhor agente
    print(f"\nTestando melhor agente...")
    test_score = fitness_function(best_weights)
    print(f"Score de teste: {test_score:.2f}")
    
    return best_weights, best_fitness, history


if __name__ == "__main__":
    best_weights, best_fitness, history = main()
