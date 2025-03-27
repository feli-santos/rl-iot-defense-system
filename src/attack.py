import numpy as np
from typing import Dict, List, Tuple
import networkx as nx
import random

def generate_attack_graph(num_devices: int) -> nx.DiGraph:
    """
    Generate a directed acyclic attack dependency graph as described in the paper.
    The graph represents possible attack paths through IoT devices.
    
    Args:
        num_devices: Number of IoT devices in the environment
        
    Returns:
        A NetworkX directed acyclic graph representing attack dependencies
    """
    G = nx.DiGraph()
    
    # Add nodes (devices)
    for i in range(num_devices):
        G.add_node(i, device_id=f"device_{i}")
    
    # Create edges with dependencies (following monotonicity property from paper)
    # We'll create a hierarchical structure where attacks must progress through layers
    layers = _create_device_layers(num_devices)
    
    for i in range(len(layers)-1):
        current_layer = layers[i]
        next_layer = layers[i+1]
        
        # Create dependencies between layers (at least one edge per node)
        for node in current_layer:
            # Connect to 1-3 nodes in next layer (randomly)
            num_targets = random.randint(1, min(3, len(next_layer)))
            targets = random.sample(next_layer, num_targets)
            
            for target in targets:
                G.add_edge(node, target, 
                          exploit_id=f"exploit_{node}_{target}",
                          difficulty=random.uniform(0.1, 1.0))
    
    # Ensure it's acyclic (as per paper's attack dependency graph)
    assert nx.is_directed_acyclic_graph(G), "Attack graph must be acyclic"
    
    return G

def _create_device_layers(num_devices: int) -> List[List[int]]:
    """
    Create hierarchical layers of devices for the attack graph.
    This ensures the monotonicity property mentioned in the paper.
    
    Args:
        num_devices: Total number of devices
        
    Returns:
        List of layers where each layer contains device indices
    """
    # Determine number of layers (3-5 layers)
    num_layers = min(max(3, num_devices // 4), 5)
    
    # Distribute devices across layers
    layers = []
    devices_per_layer = num_devices // num_layers
    remaining = num_devices % num_layers
    
    for i in range(num_layers):
        layer_size = devices_per_layer + (1 if i < remaining else 0)
        start_idx = sum(len(l) for l in layers)
        layer = list(range(start_idx, start_idx + layer_size))
        layers.append(layer)
    
    return layers

def get_critical_paths(graph: nx.DiGraph, num_paths: int = 3) -> List[List[int]]:
    """
    Identify critical attack paths in the graph (used for attack simulation)
    as described in the paper's attack characterization section.
    
    Args:
        graph: The attack dependency graph
        num_paths: Number of critical paths to identify
        
    Returns:
        List of critical paths (each path is a list of nodes)
    """
    # Find all source nodes (nodes with no incoming edges)
    sources = [node for node in graph.nodes() if graph.in_degree(node) == 0]
    
    # Find all target nodes (nodes with no outgoing edges)
    targets = [node for node in graph.nodes() if graph.out_degree(node) == 0]
    
    critical_paths = []
    
    # Find top longest paths from sources to targets
    for source in sources[:num_paths]:
        for target in targets[:num_paths]:
            try:
                path = nx.shortest_path(graph, source, target)
                critical_paths.append(path)
            except nx.NetworkXNoPath:
                continue
    
    # Sort by path length and return top paths
    critical_paths.sort(key=len, reverse=True)
    return critical_paths[:num_paths]

def simulate_attack_sequence(graph: nx.DiGraph, start_node: int, max_steps: int = 10) -> List[int]:
    """
    Simulate an attack sequence starting from a given node,
    following the attack strategy described in the paper.
    
    Args:
        graph: The attack dependency graph
        start_node: Node where attack begins
        max_steps: Maximum length of attack sequence
        
    Returns:
        List of nodes representing the attack sequence
    """
    current_node = start_node
    sequence = [current_node]
    
    for _ in range(max_steps - 1):
        # Get possible next nodes (successors)
        successors = list(graph.successors(current_node))
        
        if not successors:
            break
        
        # Choose next node based on exploit difficulty (lower is easier)
        next_node = min(successors, key=lambda x: graph.edges[current_node, x]['difficulty'])
        sequence.append(next_node)
        current_node = next_node
    
    return sequence