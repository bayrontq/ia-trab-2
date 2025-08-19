#!/usr/bin/env python3
"""
Script para visualizar o agente neural jogando
Trabalho 2 - Inteligência Artificial e Sistemas Inteligentes
Matrícula: 2025130736
"""

import sys
import os
import glob
import numpy as np

# Adicionar path do jogo
sys.path.append('clear_game')

from game.core import SurvivalGame, GameConfig
from game.agents import NeuralNetworkAgent

def main():
    print("="*60)
    print("VISUALIZAÇÃO DO AGENTE NEURAL")
    print("Trabalho 2 - Matrícula: 2025130736")
    print("="*60)
    
    # Procurar pesos treinados
    weight_files = glob.glob('best_neural_weights_*.npy')
    
    if weight_files:
        # Usar o mais recente
        weight_file = max(weight_files)
        weights = np.load(weight_file)
        agent = NeuralNetworkAgent(weights)
        print(f"✓ Carregando pesos treinados: {weight_file}")
        
        # Informações sobre o agente
        print(f"✓ Arquitetura: {agent.layer_sizes}")
        print(f"✓ Total de parâmetros: {agent.total_params}")
    else:
        agent = NeuralNetworkAgent()
        print("⚠️ Pesos treinados não encontrados!")
        print("⚠️ Usando pesos aleatórios (execute o treinamento primeiro)")
        print(f"✓ Arquitetura: {agent.layer_sizes}")
    
    print()
    print("Configurações do jogo:")
    print("✓ Interface visual: ativada")
    print("✓ Grade sensorial: visível")
    print("✓ Resolução: 800x600")
    print()
    
    # Configurar jogo visual
    config = GameConfig(
        render_grid=True,    # Mostrar grade sensorial
        fps=60              # 60 FPS para visualização suave
    )
    
    game = SurvivalGame(config, render=True)
    
    print("🎮 Iniciando visualização...")
    print("📋 Controles:")
    print("   - Feche a janela para parar")
    print("   - O agente joga automaticamente")
    print()
    print("🧠 Agente neural em ação!")
    print("-" * 30)
    
    try:
        frame_count = 0
        
        while not game.all_players_dead():
            # Obter estado do jogo (27 elementos)
            state = game.get_state(0, include_internals=True)
            
            # Agente neural decide ação
            action = agent.predict(state)
            
            # Atualizar jogo
            game.update([action])
            game.render_frame()
            
            # Mostrar progresso ocasionalmente
            frame_count += 1
            if frame_count % 300 == 0:  # A cada 5 segundos (60fps)
                score = game.players[0].score
                print(f"Frame {frame_count:5d} | Score: {score:6.2f} | Ação: {action}")
        
        # Resultado final
        final_score = game.players[0].score
        print()
        print("="*60)
        print("GAME OVER!")
        print(f"Score Final: {final_score:.2f}")
        print(f"Frames Jogados: {frame_count}")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n🛑 Visualização interrompida pelo usuário")
    
    except Exception as e:
        print(f"\n❌ Erro durante visualização: {e}")
    
    finally:
        print("\n👋 Visualização encerrada")

if __name__ == "__main__":
    main()
