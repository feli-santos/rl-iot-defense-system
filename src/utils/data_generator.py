import numpy as np
import networkx as nx
from typing import Tuple, List

class RealisticAttackDataGenerator:
    """Generate realistic attack sequences for training the LSTM model"""
    
    def __init__(self, num_nodes: int, num_patterns: int = 3):
        self.num_nodes = num_nodes
        self.num_patterns = num_patterns
        self.network_graph = self._create_network_topology()
        self.attack_patterns = self._generate_attack_patterns()
    
    def _create_network_topology(self) -> nx.Graph:
        """Create a realistic IoT network topology"""
        # Start with a small-world network (IoT networks often have this property)
        G = nx.watts_strogatz_graph(n=self.num_nodes, k=3, p=0.3)
        return G
    
    def _generate_attack_patterns(self) -> List[List[int]]:
        """Generate different attack patterns/paths through the network"""
        patterns = []
        
        # Pattern 1: Random walk through the network
        pattern = [np.random.randint(0, self.num_nodes)]
        for _ in range(5):
            neighbors = list(self.network_graph.neighbors(pattern[-1]))
            if neighbors:
                pattern.append(np.random.choice(neighbors))
            else:
                pattern.append(np.random.randint(0, self.num_nodes))
        patterns.append(pattern)
        
        # Pattern 2: Shortest path to high-value target (last node)
        start = np.random.randint(0, self.num_nodes // 2)
        target = self.num_nodes - 1
        try:
            path = nx.shortest_path(self.network_graph, start, target)
            patterns.append(path)
        except:
            patterns.append([start, target])
        
        # Pattern 3: Distributed attack
        pattern = list(np.random.choice(self.num_nodes, size=4, replace=False))
        patterns.append(pattern)
        
        return patterns
    
    def generate_sequence(self, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate attack sequence and corresponding labels"""
        # Choose a random attack pattern
        pattern_idx = np.random.randint(0, len(self.attack_patterns))
        pattern = self.attack_patterns[pattern_idx]
        
        # Create sequence
        X = np.zeros((seq_length, self.num_nodes))
        y = np.zeros((seq_length, self.num_nodes))
        
        # Apply pattern with some noise
        current_pos = 0
        for i in range(seq_length):
            if i > 0:
                X[i] = X[i-1].copy()  # Carry over previous state
            
            if current_pos < len(pattern):
                node = pattern[current_pos]
                X[i, node] = 1.0
                
                # Advance pattern with some randomness
                if np.random.random() > 0.3:  # 70% chance to follow pattern
                    current_pos += 1
            
            # Add label (next position to be attacked)
            if current_pos < len(pattern):
                next_node = pattern[current_pos]
                y[i, next_node] = 1.0
            
            # Add random noise occasionally
            if np.random.random() < 0.1:  # 10% noise
                random_node = np.random.randint(0, self.num_nodes)
                X[i, random_node] = 1.0
        
        return X, y
    
    def generate_batch(self, batch_size: int, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a batch of attack sequences"""
        X_batch = np.zeros((batch_size, seq_length, self.num_nodes))
        y_batch = np.zeros((batch_size, seq_length, self.num_nodes))
        
        for i in range(batch_size):
            X, y = self.generate_sequence(seq_length)
            X_batch[i] = X
            y_batch[i] = y
        
        return X_batch, y_batch
