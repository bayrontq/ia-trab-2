"""
Sistema de Avaliação com 30 Execuções e Análise Estatística
Trabalho 2 - Inteligência Artificial e Sistemas Inteligentes
Matrícula: 2025130736

Conforme requisitos:
- 30 execuções para cada agente
- Testes estatísticos (t-test e Wilcoxon)
- Comparação com baselines fornecidos
- Uso do jogo como biblioteca
"""

import numpy as np
import sys
import os
from typing import List, Dict
from scipy import stats

# Adicionar o diretório do jogo ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'clear_game'))

from game.core import SurvivalGame, GameConfig
from game.agents import NeuralNetworkAgent


class AgentEvaluator:
    """
    Avaliador rigoroso para agentes com 30 execuções
    """
    
    def __init__(self, num_trials: int = 30):
        """
        Args:
            num_trials: 30 execuções conforme requisitos
        """
        self.num_trials = num_trials
        self.config = GameConfig(num_players=1, fps=60)
    
    def evaluate_neural_agent(self, weights: np.ndarray, agent_name: str = "Neural ABC") -> Dict:
        """
        Avalia agente neural com 30 execuções
        """
        print(f"Avaliando {agent_name} com {self.num_trials} execuções...")
        
        scores = []
        
        for trial in range(self.num_trials):
            try:
                # Criar agente com pesos
                agent = NeuralNetworkAgent(weights)
                
                # Criar jogo
                game = SurvivalGame(config=self.config, render=False)
                
                # Jogar até morrer
                while not game.all_players_dead():
                    state = game.get_state(0, include_internals=True)
                    action = agent.predict(state)
                    game.update([action])
                
                score = game.players[0].score
                scores.append(score)
                
                if (trial + 1) % 5 == 0:
                    print(f"  Trial {trial+1}/{self.num_trials}: {score:.2f}")
                    
            except Exception as e:
                print(f"  Erro no trial {trial+1}: {e}")
                scores.append(0.0)
        
        # Calcular estatísticas
        scores = np.array(scores)
        results = {
            'agent_name': agent_name,
            'scores': scores,
            'mean': np.mean(scores),
            'std': np.std(scores, ddof=1),
            'min': np.min(scores),
            'max': np.max(scores),
            'median': np.median(scores),
            'num_trials': self.num_trials
        }
        
        print(f"  Resultados {agent_name}:")
        print(f"    Média: {results['mean']:.2f}")
        print(f"    Desvio: {results['std']:.2f}")
        print(f"    Min/Max: {results['min']:.2f}/{results['max']:.2f}")
        print()
        
        return results


class StatisticalAnalyzer:
    """
    Análise estatística conforme requisitos
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Args:
            significance_level: 95% de confiança (α = 0.05)
        """
        self.significance_level = significance_level
    
    def compare_agents(self, results1: Dict, results2: Dict) -> Dict:
        """
        Compara dois agentes com testes estatísticos
        """
        scores1 = results1['scores']
        scores2 = results2['scores']
        
        # Teste t para amostras independentes (conforme requisitos)
        t_stat, t_pvalue = stats.ttest_ind(scores1, scores2)
        
        # Teste não paramétrico de Wilcoxon (conforme requisitos)
        try:
            wilcoxon_stat, wilcoxon_pvalue = stats.mannwhitneyu(
                scores1, scores2, alternative='two-sided'
            )
        except ValueError:
            wilcoxon_stat, wilcoxon_pvalue = 0, 1.0
        
        # Determinar significância
        t_significant = t_pvalue < self.significance_level
        wilcoxon_significant = wilcoxon_pvalue < self.significance_level
        
        # Determinar superioridade
        mean_diff = results1['mean'] - results2['mean']
        if abs(mean_diff) < 1e-6:
            superiority = "Empate"
        elif mean_diff > 0:
            superiority = f"{results1['agent_name']} superior"
        else:
            superiority = f"{results2['agent_name']} superior"
        
        comparison = {
            'agent1': results1['agent_name'],
            'agent2': results2['agent_name'],
            'mean_difference': mean_diff,
            'superiority': superiority,
            't_test': {
                'statistic': t_stat,
                'p_value': t_pvalue,
                'significant': t_significant
            },
            'wilcoxon_test': {
                'statistic': wilcoxon_stat,
                'p_value': wilcoxon_pvalue,
                'significant': wilcoxon_significant
            }
        }
        
        return comparison
    
    def print_comparison(self, comparison: Dict):
        """Imprime comparação formatada"""
        print(f"Comparação: {comparison['agent1']} vs {comparison['agent2']}")
        print(f"Diferença de médias: {comparison['mean_difference']:.4f}")
        print(f"Resultado: {comparison['superiority']}")
        print()
        
        print("Teste t (paramétrico):")
        t_test = comparison['t_test']
        print(f"  Estatística t: {t_test['statistic']:.4f}")
        print(f"  P-valor: {t_test['p_value']:.6f}")
        print(f"  Significativo (α=0.05): {'Sim' if t_test['significant'] else 'Não'}")
        print()
        
        print("Teste Wilcoxon (não paramétrico):")
        w_test = comparison['wilcoxon_test']
        print(f"  Estatística U: {w_test['statistic']:.4f}")
        print(f"  P-valor: {w_test['p_value']:.6f}")
        print(f"  Significativo (α=0.05): {'Sim' if w_test['significant'] else 'Não'}")
        print()


def create_baseline_results():
    """
    Cria resultados dos agentes baseline usando dados fornecidos nos requisitos
    """
    # Dados fornecidos nos requisitos do trabalho
    rule_based_scores = np.array([
        12.69, 16.65, 6.97, 2.79, 15.94, 10.22, 21.90, 4.35, 6.22, 9.95,
        19.94, 20.56, 15.74, 17.68, 7.16, 15.68, 2.37, 15.43, 15.13, 22.50,
        25.82, 15.85, 17.02, 16.74, 14.69, 11.73, 13.80, 15.13, 12.35, 16.19
    ])
    
    neural_baseline_scores = np.array([
        38.32, 54.53, 61.16, 27.55, 16.08, 26.00, 25.33, 18.30, 39.76, 48.17,
        44.77, 47.54, 75.43, 23.68, 16.83, 15.81, 67.17, 53.54, 33.59, 49.24,
        52.65, 16.35, 44.05, 56.59, 63.23, 43.96, 43.82, 19.19, 28.36, 18.65
    ])
    
    human_scores = np.array([
        27.34, 17.63, 39.33, 17.44, 1.16, 24.04, 29.21, 18.92, 25.71, 20.05,
        31.88, 15.39, 22.50, 19.27, 26.33, 23.67, 16.82, 28.45, 12.59, 33.01,
        21.74, 14.23, 27.90, 24.80, 11.35, 30.12, 17.08, 22.96, 9.41, 35.22
    ])
    
    # Criar estruturas de resultados
    baselines = []
    
    for name, scores in [("Rule Based GA", rule_based_scores),
                        ("Neural GA Baseline", neural_baseline_scores), 
                        ("Human Player", human_scores)]:
        
        baseline = {
            'agent_name': name,
            'scores': scores,
            'mean': np.mean(scores),
            'std': np.std(scores, ddof=1),
            'min': np.min(scores),
            'max': np.max(scores),
            'median': np.median(scores),
            'num_trials': 30
        }
        baselines.append(baseline)
    
    return baselines


def save_results_table(all_results: List[Dict], comparisons: List[Dict]):
    """
    Salva tabela de resultados conforme requisitos
    """
    with open("results_table.txt", 'w') as f:
        f.write("="*80 + "\n")
        f.write("RESULTADOS DA AVALIAÇÃO - TRABALHO 2 IA\n")
        f.write("Matrícula: 2025130736 - ABC + Rede Neural\n")
        f.write("="*80 + "\n\n")
        
        # Tabela principal
        f.write("TABELA DE RESULTADOS (30 EXECUÇÕES POR AGENTE)\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Agente':<25} {'Média':<10} {'Desvio':<10} {'Min':<8} {'Max':<8}\n")
        f.write("-"*70 + "\n")
        
        for result in all_results:
            f.write(f"{result['agent_name']:<25} "
                   f"{result['mean']:<10.2f} "
                   f"{result['std']:<10.2f} "
                   f"{result['min']:<8.2f} "
                   f"{result['max']:<8.2f}\n")
        
        f.write("\n\n")
        
        # Dados brutos das 30 execuções
        f.write("DADOS BRUTOS (30 EXECUÇÕES)\n")
        f.write("-"*50 + "\n")
        
        for result in all_results:
            f.write(f"\n{result['agent_name']}:\n")
            scores = result['scores']
            # Quebrar em linhas de 10 valores
            for i in range(0, len(scores), 10):
                line_scores = scores[i:i+10]
                f.write("  " + ", ".join([f"{s:.2f}" for s in line_scores]) + "\n")
        
        f.write("\n\n")
        
        # Testes estatísticos
        f.write("TESTES ESTATÍSTICOS (α = 0.05)\n")
        f.write("-"*50 + "\n")
        
        for comp in comparisons:
            f.write(f"\n{comp['agent1']} vs {comp['agent2']}:\n")
            f.write(f"  Diferença de médias: {comp['mean_difference']:.4f}\n")
            
            t_sig = "Sim" if comp['t_test']['significant'] else "Não"
            f.write(f"  Teste t: p={comp['t_test']['p_value']:.6f}, Significativo: {t_sig}\n")
            
            w_sig = "Sim" if comp['wilcoxon_test']['significant'] else "Não"  
            f.write(f"  Wilcoxon: p={comp['wilcoxon_test']['p_value']:.6f}, Significativo: {w_sig}\n")
            
            f.write(f"  Resultado: {comp['superiority']}\n")
    
    print("Tabela de resultados salva: results_table.txt")


def main():
    """Função principal de avaliação"""
    print("="*80)
    print("AVALIAÇÃO COM 30 EXECUÇÕES - TRABALHO 2")
    print("Matrícula: 2025130736")
    print("="*80)
    
    # Verificar se existem pesos treinados
    possible_weights = [
        "best_neural_weights_*.npy",
        "best_neural_weights.npy"
    ]
    
    weights_file = None
    import glob
    for pattern in possible_weights:
        files = glob.glob(pattern)
        if files:
            weights_file = files[-1]  # Pegar o mais recente
            break
    
    # Inicializar avaliador e analisador
    evaluator = AgentEvaluator(num_trials=30)
    analyzer = StatisticalAnalyzer()
    
    all_results = []
    
    # Avaliar agente neural treinado (se disponível)
    if weights_file and os.path.exists(weights_file):
        print(f"Carregando pesos treinados: {weights_file}")
        weights = np.load(weights_file)
        neural_results = evaluator.evaluate_neural_agent(weights, "Neural ABC")
        all_results.append(neural_results)
    else:
        print("Pesos do agente neural não encontrados.")
        print("Execute primeiro: python abc_neural_training.py")
        print("\nAvaliando apenas agentes baseline...")
    
    # Adicionar baselines
    baselines = create_baseline_results()
    all_results.extend(baselines)
    
    # Imprimir resumo
    print("\n" + "="*70)
    print("RESUMO DOS RESULTADOS")
    print("="*70)
    for result in all_results:
        print(f"{result['agent_name']:<25}: "
              f"Média={result['mean']:.2f}, "
              f"Desvio={result['std']:.2f}")
    
    # Realizar comparações estatísticas
    print("\n" + "="*70)
    print("ANÁLISE ESTATÍSTICA COMPARATIVA")
    print("="*70)
    
    comparisons = []
    
    # Comparar todos os pares
    for i in range(len(all_results)):
        for j in range(i + 1, len(all_results)):
            comparison = analyzer.compare_agents(all_results[i], all_results[j])
            comparisons.append(comparison)
            analyzer.print_comparison(comparison)
            print("-" * 50)
    
    # Salvar resultados
    save_results_table(all_results, comparisons)
    
    print("\nAvaliação completa concluída!")
    return all_results, comparisons


if __name__ == "__main__":
    # Verificar se scipy está disponível
    try:
        import scipy.stats
    except ImportError:
        print("Erro: scipy não está instalado.")
        print("Execute: pip install scipy")
        sys.exit(1)
    
    all_results, comparisons = main()
