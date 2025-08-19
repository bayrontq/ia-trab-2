"""
Sistema de Treinamento ABC + Rede Neural
Trabalho 2 - Inteligência Artificial e Sistemas Inteligentes
Bayron Thiengo Quinelato - 2025130736

Conforme requisitos:
- Gráfico da evolução do agente (eixo x = iteração, eixo y = melhor pontuação)
- Boxplots para comparação
- Sem bibliotecas externas além das básicas
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import glob
from typing import List, Dict, Optional

def plot_training_evolution(history_file: str = None):
    """
    Cria gráfico da evolução do treinamento conforme requisitos
    """
    # Procurar arquivo de histórico
    if history_file is None:
        history_files = glob.glob("training_history_*.pkl")
        if not history_files:
            print("Nenhum arquivo de histórico encontrado.")
            return
        history_file = history_files[-1]  # Mais recente
    
    if not os.path.exists(history_file):
        print(f"Arquivo não encontrado: {history_file}")
        return
    
    # Carregar histórico
    with open(history_file, 'rb') as f:
        history = pickle.load(f)
    
    # Compatibilidade com ambos os formatos (antigo e novo)
    best_scores = history.get('best_score_history', history.get('best_fitness_history', []))
    iterations = list(range(len(best_scores)))
    
    # Criar gráfico
    plt.figure(figsize=(12, 8))
    
    # Gráfico principal
    plt.subplot(2, 1, 1)
    plt.plot(iterations, best_scores, 'b-', linewidth=2, label='Melhor Pontuação')
    
    # Adicionar pontuação média se disponível
    mean_scores = history.get('mean_score_history', history.get('fitness_history', []))
    if mean_scores:
        plt.plot(iterations, mean_scores, 'r--', 
                linewidth=1, alpha=0.7, label='Pontuação Média')
    
    plt.xlabel('Iteração')
    plt.ylabel('Pontuação')
    plt.title('Evolução do Agente Neural com ABC\n(Matrícula: 2025130736)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Zoom nas últimas iterações (se houver mais de 50)
    plt.subplot(2, 1, 2)
    if len(iterations) > 50:
        start_idx = max(0, len(iterations) - 50)
        zoom_iter = iterations[start_idx:]
        zoom_scores = best_scores[start_idx:]
        
        plt.plot(zoom_iter, zoom_scores, 'b-', linewidth=2)
        plt.xlabel('Iteração')
        plt.ylabel('Melhor Pontuação')
        plt.title('Zoom: Últimas 50 Iterações')
        plt.grid(True, alpha=0.3)
    else:
        plt.plot(iterations, best_scores, 'b-', linewidth=2)
        plt.xlabel('Iteração')
        plt.ylabel('Melhor Pontuação')
        plt.title('Evolução Completa')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_evolution.png', dpi=300, bbox_inches='tight')
    print("Gráfico de evolução salvo: training_evolution.png")
    plt.show()

def create_boxplot_comparison():
    """
    Cria boxplots para comparação conforme exemplo dos requisitos
    """
    # Dados dos agentes baseline (fornecidos nos requisitos)
    rule_based_data = [
        12.69, 16.65, 6.97, 2.79, 15.94, 10.22, 21.90, 4.35, 6.22, 9.95,
        19.94, 20.56, 15.74, 17.68, 7.16, 15.68, 2.37, 15.43, 15.13, 22.50,
        25.82, 15.85, 17.02, 16.74, 14.69, 11.73, 13.80, 15.13, 12.35, 16.19
    ]
    
    neural_baseline_data = [
        38.32, 54.53, 61.16, 27.55, 16.08, 26.00, 25.33, 18.30, 39.76, 48.17,
        44.77, 47.54, 75.43, 23.68, 16.83, 15.81, 67.17, 53.54, 33.59, 49.24,
        52.65, 16.35, 44.05, 56.59, 63.23, 43.96, 43.82, 19.19, 28.36, 18.65
    ]
    
    human_data = [
        27.34, 17.63, 39.33, 17.44, 1.16, 24.04, 29.21, 18.92, 25.71, 20.05,
        31.88, 15.39, 22.50, 19.27, 26.33, 23.67, 16.82, 28.45, 12.59, 33.01,
        21.74, 14.23, 27.90, 24.80, 11.35, 30.12, 17.08, 22.96, 9.41, 35.22
    ]
    
    # Tentar carregar dados do agente neural treinado
    neural_abc_data = None
    
    # Procurar arquivos de resultados
    result_files = glob.glob("*results*.txt")
    if result_files:
        # Tentar extrair dados do arquivo de resultados
        try:
            with open(result_files[0], 'r') as f:
                content = f.read()
                # Procurar linha com "Neural ABC:"
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if 'Neural ABC' in line and ':' in line:
                        # Próximas linhas devem ter os dados
                        data_lines = []
                        for j in range(i+1, min(i+4, len(lines))):
                            if lines[j].strip() and not lines[j].startswith(' '):
                                break
                            data_lines.append(lines[j])
                        
                        # Extrair números
                        numbers = []
                        for data_line in data_lines:
                            parts = data_line.strip().split()
                            for part in parts:
                                part = part.replace(',', '')
                                try:
                                    numbers.append(float(part))
                                except ValueError:
                                    continue
                        
                        if len(numbers) == 30:
                            neural_abc_data = numbers
                            break
        except Exception as e:
            print(f"Erro ao carregar dados do Neural ABC: {e}")
    
    # Preparar dados para boxplot
    data_list = [rule_based_data, neural_baseline_data, human_data]
    labels = ['Rule Based GA', 'Neural GA Baseline', 'Human Player']
    
    if neural_abc_data:
        data_list.insert(0, neural_abc_data)
        labels.insert(0, 'Neural ABC')
    
    # Criar boxplot
    plt.figure(figsize=(12, 8))
    
    bp = plt.boxplot(data_list, labels=labels, patch_artist=True)
    
    # Colorir caixas
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
    
    plt.title('Comparação de Performance dos Agentes\n(30 execuções por agente)', fontsize=14)
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Adicionar estatísticas
    for i, (data, label) in enumerate(zip(data_list, labels)):
        mean_val = np.mean(data)
        plt.text(i+1, mean_val, f'μ={mean_val:.1f}', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('boxplot_comparison.png', dpi=300, bbox_inches='tight')
    print("Boxplot salvo: boxplot_comparison.png")
    plt.show()

def create_simple_boxplot():
    """
    Versão simplificada usando apenas matplotlib (conforme exemplo do Apêndice A)
    """
    # Exemplo do Apêndice A adaptado
    mydata = [1, 2, 3, 4, 5, 6, 12]
    
    plt.figure(figsize=(8, 6))
    plt.boxplot(mydata, labels=['Exemplo'])
    plt.title('Exemplo de Boxplot (Apêndice A)')
    plt.ylabel('Valores')
    plt.grid(True, alpha=0.3)
    plt.show()

def main():
    """Função principal para criar todos os gráficos"""
    print("="*60)
    print("GERAÇÃO DE GRÁFICOS - TRABALHO 2")
    print("Matrícula: 2025130736")
    print("="*60)
    
    # Verificar arquivos necessários
    history_files = glob.glob("training_history_*.pkl")
    
    if history_files:
        print("1. Criando gráfico de evolução do treinamento...")
        plot_training_evolution()
        print()
    else:
        print("1. Arquivo de histórico não encontrado.")
        print("   Execute primeiro: python abc_neural_training.py")
        print()
    
    print("2. Criando boxplot de comparação...")
    create_boxplot_comparison()
    print()
    
    print("3. Exemplo simples de boxplot...")
    create_simple_boxplot()
    
    print("="*60)
    print("GRÁFICOS CONCLUÍDOS")
    print("="*60)
    print("Arquivos gerados:")
    
    generated_files = ['training_evolution.png', 'boxplot_comparison.png']
    for filename in generated_files:
        if os.path.exists(filename):
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} (não gerado)")
    
    print("\nOs gráficos podem ser usados no artigo científico.")

if __name__ == "__main__":
    main()
