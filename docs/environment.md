# IoT Environment Simulation

## Environment Design

The IoT environment is modeled as a Markov Decision Process (MDP) with:

- **State space**: Current security status of all network devices
- **Action space**: Available defensive measures
- **Transition function**: How defensive actions and attacks change the network
- **Reward function**: Feedback based on security outcomes

## Environment Implementation

The environment is implemented as a custom Gymnasium environment in `environment.py` that simulates an IoT network with multiple connected devices.

### State Representation

Each state $s_t$ represents the security status of the network at time $t$:

$$s_t = \{d_1, d_2, ..., d_n, a_1, a_2, ..., a_m\}$$

Where:
- $d_i$ represents the status of device $i$ (compromised, secure, etc.)
- $a_j$ represents the activity on connection $j$ (normal, suspicious, etc.)

### Action Space

The action space $A$ consists of defensive countermeasures:

$$A = \{a_1, a_2, a_3, a_4\}$$

Where:
- $a_1$: Patch vulnerable device
- $a_2$: Isolate compromised device
- $a_3$: Reset device
- $a_4$: Monitor device (no direct action)

### Attack Graph Generation

Attack paths are generated using a directed graph model. The function `generate_attack_graph()` creates a network topology where:

```python
def generate_attack_graph(num_nodes, edge_probability, critical_node_ratio=0.2):
    G = nx.erdos_renyi_graph(n=num_nodes, p=edge_probability, directed=True)
    
    # Designate critical nodes
    critical_nodes = random.sample(list(G.nodes()), int(num_nodes * critical_node_ratio))
    
    # Assign node attributes
    for node in G.nodes():
        G.nodes[node]['critical'] = node in critical_nodes
        G.nodes[node]['compromised'] = False
        G.nodes[node]['vulnerability'] = random.uniform(0.1, 0.9)
    
    return G
```

Critical nodes are high-value targets that provide larger rewards to the attacker if compromised.

### Critical Path Analysis

The function `get_critical_paths()` identifies the most vulnerable paths through the network:

```python
def get_critical_paths(G, source, targets, k=3):
    """Find k-shortest paths from source to critical targets"""
    paths = []
    
    for target in targets:
        try:
            # Find k shortest paths for each target
            shortest_paths = list(islice(nx.shortest_simple_paths(G, source, target), k))
            paths.extend(shortest_paths)
        except nx.NetworkXNoPath:
            continue
    
    # Sort paths by length and vulnerability
    return sorted(paths, key=lambda p: (len(p), sum(G.nodes[n]['vulnerability'] for n in p)))
```

### Adaptive Attacker

The `AdaptiveAttacker` class simulates an intelligent adversary that:

1. Identifies high-value targets
2. Adapts strategy based on defense actions
3. Follows vulnerability-weighted paths
4. Maintains persistence on the network

```python
class AdaptiveAttacker:
    def __init__(self, attack_graph, learning_rate=0.1):
        self.graph = attack_graph
        self.current_node = self._select_entry_point()
        self.target_nodes = self._identify_targets()
        self.attack_paths = get_critical_paths(self.graph, self.current_node, self.target_nodes)
        self.learning_rate = learning_rate
        self.node_values = {node: 1.0 for node in self.graph.nodes()}
        
    def select_next_target(self, defended_nodes):
        # Update node values based on defense actions
        for node in defended_nodes:
            self.node_values[node] *= (1 - self.learning_rate)
        
        # Select next target considering updated values
        # ...implementation details...
```

### Reward Function

The reward function $R(s, a, s')$ is designed to:

1. Reward successful defense actions
2. Penalize compromised devices
3. Provide larger penalties for critical node compromises
4. Consider the long-term network health

Mathematically:

$$R(s, a, s') = w_1 \cdot \text{DefenseSuccess} - w_2 \cdot \text{CompromisedCount} - w_3 \cdot \text{CriticalCompromised} + w_4 \cdot \text{NetworkHealth}$$

Where:
- $w_i$ are weight parameters
- $\text{DefenseSuccess}$ is 1 if the action prevented an attack, 0 otherwise
- $\text{CompromisedCount}$ is the number of compromised devices
- $\text{CriticalCompromised}$ is the number of critical compromised nodes
- $\text{NetworkHealth}$ is a measure of overall network integrity

## Environment-Agent Interaction

Each step in the environment follows this sequence:

1. Agent observes current state $s_t$
2. Agent selects defense action $a_t$
3. Action is applied to the environment
4. Attacker selects and executes next attack
5. New state $s_{t+1}$ and reward $r_t$ are calculated
6. Episode terminates if critical nodes are compromised or maximum steps reached

## Environment Configuration

Key parameters in the environment configuration include:

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `ENVIRONMENT_NUM_DEVICES` | Number of IoT devices | 20 |
| `ENVIRONMENT_NUM_STATES` | State space dimension | 23 |
| `ENVIRONMENT_NUM_ACTIONS` | Action space dimension | 4 |
| `ENVIRONMENT_HISTORY_LENGTH` | Number of previous states to track | 5 |
| `ENVIRONMENT_REWARD_WEIGHTS` | Weights for reward components | [1.0, 0.5, 2.0, 0.3] |

These parameters can be adjusted in the `config.py` file to simulate different network conditions and attack scenarios.