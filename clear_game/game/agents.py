import numpy as np
from abc import ABC, abstractmethod
from typing import List

class Agent(ABC):
    """Interface para todos os agentes."""
    @abstractmethod
    def predict(self, state: np.ndarray) -> int:
        """Faz uma previsão de ação com base no estado atual."""
        pass

class HumanAgent(Agent):
    """Agente controlado por um humano (para modo manual)"""
    def predict(self, state: np.ndarray) -> int:
        # O estado é ignorado - entrada vem do teclado
        return 0  # Padrão: não fazer nada (será sobrescrito pela entrada do usuário no manual_play.py)

class NeuralNetworkAgent(Agent):
    """
    Agente que usa rede neural para tomar decisões.
    Implementado conforme requisitos do Trabalho 2.
    Matrícula: 2025130736 -> Rede Neural + Colônia de Abelhas
    """
    
    def __init__(self, weights: np.ndarray = None):
        """
        Inicializa o agente neural.
        
        Args:
            weights: Pesos da rede neural (se None, usa pesos aleatórios)
        """
        # Arquitetura da rede: 27 -> 32 -> 16 -> 3
        # 27 = grade 5x5 (25) + 2 variáveis internas
        # 3 = ações (0=noop, 1=cima, 2=baixo)
        self.layer_sizes = [27, 32, 16, 3]
        self.num_layers = len(self.layer_sizes)
        
        # Inicializar estrutura da rede
        self.weights = []
        self.biases = []
        
        # Calcular número total de parâmetros
        total_params = 0
        for i in range(self.num_layers - 1):
            w_size = self.layer_sizes[i] * self.layer_sizes[i + 1]
            b_size = self.layer_sizes[i + 1]
            total_params += w_size + b_size
        
        self.total_params = total_params
        
        # Configurar pesos
        if weights is not None:
            self.set_weights(weights)
        else:
            self._initialize_random_weights()
    
    def _initialize_random_weights(self):
        """Inicializa pesos aleatórios usando Xavier initialization."""
        self.weights = []
        self.biases = []
        
        for i in range(self.num_layers - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            
            # Xavier initialization
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            w = np.random.uniform(-limit, limit, (fan_in, fan_out))
            b = np.zeros((fan_out,))
            
            self.weights.append(w)
            self.biases.append(b)
    
    def set_weights(self, weight_vector: np.ndarray):
        """
        Define os pesos da rede a partir de um vetor.
        Usado pela metaheurística ABC para otimizar.
        """
        if len(weight_vector) != self.total_params:
            raise ValueError(f"Weight vector deve ter {self.total_params} elementos, recebido {len(weight_vector)}")
        
        self.weights = []
        self.biases = []
        idx = 0
        
        for i in range(self.num_layers - 1):
            # Extrair pesos da matriz
            w_size = self.layer_sizes[i] * self.layer_sizes[i + 1]
            w_flat = weight_vector[idx:idx + w_size]
            w = w_flat.reshape((self.layer_sizes[i], self.layer_sizes[i + 1]))
            self.weights.append(w)
            idx += w_size
            
            # Extrair bias
            b_size = self.layer_sizes[i + 1]
            b = weight_vector[idx:idx + b_size]
            self.biases.append(b)
            idx += b_size
    
    def get_weights(self) -> np.ndarray:
        """Retorna todos os pesos como vetor."""
        vector = []
        for i in range(len(self.weights)):
            vector.extend(self.weights[i].flatten())
            vector.extend(self.biases[i].flatten())
        return np.array(vector)
    
    def _activation_tanh(self, x: np.ndarray) -> np.ndarray:
        """Função de ativação tangente hiperbólica."""
        return np.tanh(x)
    
    def _activation_softmax(self, x: np.ndarray) -> np.ndarray:
        """Função de ativação softmax."""
        # Softmax estável numericamente
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x)
    
    def _forward(self, x: np.ndarray) -> np.ndarray:
        """Propagação forward da rede neural."""
        current_input = x.copy()
        
        # Camadas ocultas com tanh
        for i in range(len(self.weights) - 1):
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            current_input = self._activation_tanh(z)
        
        # Camada de saída com softmax
        z = np.dot(current_input, self.weights[-1]) + self.biases[-1]
        output = self._activation_softmax(z)
        
        return output
    
    def predict(self, state: np.ndarray) -> int:
        """
        Faz uma previsão de ação com base no estado atual.
        Implementa a interface Agent conforme requisitos.
        
        Args:
            state: Estado do jogo (grade + variáveis internas)
            
        Returns:
            Ação a ser tomada (0=noop, 1=cima, 2=baixo)
        """
        # Verificar se o estado tem o tamanho correto (27 elementos)
        if len(state) != 27:
            raise ValueError(f"Estado deve ter 27 elementos (grade 5x5 + 2 internas), recebido: {len(state)}")
        
        # Fazer predição usando a rede neural
        output = self._forward(state)
        
        # Retornar ação com maior probabilidade
        action = np.argmax(output)
        
        # Garantir que a ação está no range válido
        return int(np.clip(action, 0, 2))
