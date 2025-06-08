import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import hashlib

class TreeNode:
    """
    Class to represent a node of the tree containing data. 
    A node tracks an attribute of an equivalence class - what qualities they have and how many exist.
    An equivalence class can be defined by the sequence of TreeNodes in a tree
    """
    
    def __init__(self, attribute, parent=None):
        #print("New node!")
        self.child_nodes = {}  # Other TreeNodes formatted as {TreeNode.attribute: TreeNode}
        self.size = 0  # Size of this node
        
        # The attribute this node represents. Structured as (feature, value). 
        # Use the tuple to prevent confusion around the value if there are repeats across different features
        self.attribute = attribute
        self.parent = parent  # New attribute to track parent node
    
    def add_child(self, new_node):
        """Add a child node, and set the parent of the new_node."""
        new_node.parent = self  # Set parent reference
        self.child_nodes[new_node.attribute] = new_node
    
    def get_child(self, attribute):
        """Get child node by attribute."""
        return self.child_nodes.get(attribute, None)
    
    def increment(self):
        """Increment the size of the node."""
        self.size += 1
        
    def to_dict(self):
        """Convert the TreeNode to a dictionary."""
        return {
            'attribute': self.attribute,
            'size': self.size,
            'child_nodes': {k: v.to_dict() for k, v in self.child_nodes.items()}
        }
    
    @staticmethod
    def from_dict(data, parent=None):
        """Reconstruct a TreeNode from a dictionary."""
        node = TreeNode(data['attribute'], parent)
        node.size = data['size']
        for k, child_data in data['child_nodes'].items():
            child_node = TreeNode.from_dict(child_data, parent=node)
            node.add_child(child_node)
        return node

    def __repr__(self):
        return f"TreeNode({self.attribute}, size={self.size}, parent={self.parent.attribute if self.parent else None})"
    
    @property
    def label(self):
        if isinstance(self.attribute, (list, tuple)) and len(self.attribute) >= 2:
            return f"{self.attribute[0]}:{str(self.attribute[1])[:10]}:{self.size}\n{hashlib.md5(str(id(self)).encode()).hexdigest()[:6]}"
        elif isinstance(self.attribute, (list, tuple)):
            return f"{str(self.attribute[0])}:{self.size}\n{hashlib.md5(str(id(self)).encode()).hexdigest()[:6]}"
        else:
            return f"{str(self.attribute)}:{self.size}\n{hashlib.md5(str(id(self)).encode()).hexdigest()[:6]}"

    @property
    def num_siblings(self):
        return len(self.parent.child_nodes) - 1 # Subtract 1 to exclude the current node
    
    @property
    def upstream_sibling_counts(self):
        """
        Gets the number of siblings for this node and each parent node all the way up to the root.

        If this is a node in the 5th layer of a tree, this will return a list of len 5 like [1,2,3,4,5] if there is one sibling in the first, 2 for the parent, etc.
        """
        counts = []
        current = self
        while current.parent:
            counts.append(current.num_siblings)
            current = current.parent
        return counts

class EquivalenceClassTree:
    """
    A tree structure that acts as a sort of decision tree for categorizing data into equivalence classes 
    
    Each layer of the tree represents some attribute with each node being an option for that attribute. 
    
    The bottom layer of the tree represents the equivalence classes for the data. Intermediate layers represent
    the equivalence class formed by the current layer + all previous layers.
    """
    def __init__(self, features):
        """
        Initializes the root node
        """
        self.root = TreeNode(("root", "root"))
        self.layers = features
    
    def insert(self, data):
        """
        Insert a row into the tree.
        Each row should correspond to a unique path down the tree.
        """
        current_node = self.root
        for layer, value in zip(self.layers, data):
            # Check if the child exists for this exact combination (layer, value) under the current node.
            child_node = current_node.get_child((layer, value))

            if child_node:
                # If the node exists, increment the size of the equivalence class.
                child_node.increment()
            else:
                # Create a new node if none exists for this path.
                #print(f"Inserting at layer {layer}, creating node with attribute {layer}:{value}")
                child_node = TreeNode((layer, value), parent=current_node)
                current_node.add_child(child_node)
                child_node.increment()

            current_node = child_node  # Move to the child node for the next layer
                
    def to_networkx(self, max_depth=None, max_nodes_per_level=None):
        """
        Convert the tree to a NetworkX graph representation up to a specified depth,
        and limit the number of nodes displayed at each level.
        """
        G = nx.DiGraph()
        self._add_edges(self.root, G, max_depth=max_depth, max_nodes_per_level=max_nodes_per_level)
        return G

    def _add_edges(self, node, graph, parent=None, depth=0, max_depth=None, max_nodes_per_level=None):
        """
        Recursively adds edges from the current node to its children in the graph,
        with an optional depth limit and node limit per level.
        """
        if max_depth is not None and depth > max_depth:
            return  # Stop adding nodes beyond the max_depth

        node_label = node.label
        graph.add_node(node_label)

        if parent:
            graph.add_edge(parent, node_label)

        # Get child nodes
        child_nodes = list(node.child_nodes.values())

        # Limit the number of nodes per level if specified
        if max_nodes_per_level is not None and len(child_nodes) > max_nodes_per_level:
            child_nodes = child_nodes[:max_nodes_per_level]

        for child in child_nodes:
            self._add_edges(child, graph, node_label, depth=depth + 1, max_depth=max_depth, max_nodes_per_level=max_nodes_per_level)

    def draw(self, max_depth=None, max_nodes_per_level=None):
        """
        Draws the tree and shows it in Matplotlib with optional depth and node limits.
        """
        G = self.to_networkx(max_depth=max_depth, max_nodes_per_level=max_nodes_per_level)

        # Get layers to determine vertical positions
        layers = self._get_layers(self.root, max_depth=max_depth, max_nodes_per_level=max_nodes_per_level)

        pos = self._create_pyramid_layout(layers)

        plt.figure(figsize=(12, 8))
        nx.draw(
            G, pos, with_labels=True, node_size=3000, node_color="lightblue",
            font_size=10, font_weight="bold", arrows=True
        )
        plt.title("Equivalence Class Tree (Limited Nodes)")
        plt.show()

    def _get_layers(self, node, depth=0, layers=None, max_depth=None, max_nodes_per_level=None):
        """
        Helper function to retrieve the layers of the tree for pyramid positioning,
        with optional depth and node limits.
        """
        if layers is None:
            layers = {}
        if max_depth is not None and depth > max_depth:
            return layers  # Stop collecting layers beyond the max_depth

        if depth not in layers:
            layers[depth] = []

        # Add node to layer
        node_label = node.label
        if max_nodes_per_level is None or len(layers[depth]) < max_nodes_per_level:
            layers[depth].append(node_label)

        # Get child nodes
        child_nodes = list(node.child_nodes.values())

        # Limit the number of nodes per level if specified
        if max_nodes_per_level is not None and len(child_nodes) > max_nodes_per_level:
            child_nodes = child_nodes[:max_nodes_per_level]

        for child in child_nodes:
            self._get_layers(child, depth + 1, layers, max_depth=max_depth, max_nodes_per_level=max_nodes_per_level)

        return layers



    def _create_pyramid_layout(self, layers):
        """
        Creates a pyramid layout based on the tree layers using manual positioning
        """
        pos = {}
        max_width = max(len(layer) for layer in layers.values())
        
        for depth, layer in layers.items():
            width = len(layer)
            for i, node_label in enumerate(layer):
                # Calculate x position based on the layer's width and node index
                x = (i - (width - 1) / 2) * (12 / max_width)  # Adjust width 
                y = -depth * 2  # Space between layers
                pos[node_label] = (x, y)
        
        return pos
    
    def save_to_dict(self):
        """Save the tree and layers to a dictionary."""
        return {
            'layers': self.layers,
            'root': self.root.to_dict()
        }
    
    @staticmethod
    def load_from_dict(data):
        """Load the tree from a dictionary."""
        tree = EquivalenceClassTree(data['layers'])  # Define layers - not technically required but useful later prob
        tree.root = TreeNode.from_dict(data['root'])  # Load the root node from the dict -> this will recursively load all the children
        return tree
    
    @property
    def nodes(self):
        """
        Collect all nodes in the tree (including the root).
        """
        all_nodes = []
        self._collect_nodes(self.root, all_nodes)
        return all_nodes
    
    def _collect_nodes(self, node, all_nodes):
        """
        Helper method to recursively collect nodes.
        """
        all_nodes.append(node)
        for child in node.child_nodes.values():
            self._collect_nodes(child, all_nodes)

    def get_nodes_for_layer(self, target_layer):
        """
        Get all nodes that are exactly at the given layer.
        
        Parameters:
        - target_layer: The layer name to search for.
        
        Returns:
        - A list of nodes at the target layer.
        """
        def _get_nodes_at_layer(node, current_layer, target_layer, result):
            """
            Recursive helper to get nodes at a specific layer.
            """
            if current_layer == target_layer:
                result.append(node)
            else:
                for child in node.child_nodes.values():
                    _get_nodes_at_layer(child, child.attribute[0], target_layer, result)

        # Start recursive search from the root node
        nodes_at_layer = []
        _get_nodes_at_layer(self.root, self.root.attribute[0], target_layer, nodes_at_layer)
        
        if not nodes_at_layer:
            raise ValueError(f"Layer '{target_layer}' not found in the tree.")
        
        return nodes_at_layer

def df_to_tree(df, columns):
    """
    Converts a pandas DataFrame to an EquivalenceClassTree based on specified columns 
    
    Parameters:
    - df: A pandas DataFrame.
    - columns: A list of column names to build the tree with.
    
    Returns:
    - An EquivalenceClassTree instance.
    """
    tree = EquivalenceClassTree(columns)

    for row in tqdm(df[columns].values):
        tree.insert(row)
    
    print(f"Generated tree with {len(tree.nodes)} nodes.")
    
    return tree
