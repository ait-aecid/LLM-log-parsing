import re
from anytree import Node, RenderTree,PreOrderIter,LevelOrderIter


def extract_template(logs):
    delimiters = [' ', '=']
    pattern = '|'.join(map(re.escape, delimiters))
    logs_with_marked_equal = [re.sub('=', ' <EQUAL> ', log) for log in logs]
    all_words = [re.split(pattern, log) for log in logs_with_marked_equal]
    template = ['<*>' if len(set(word)) > 1 else word[0] for word in zip(*all_words)]
    template_str = ' '.join(template).replace(' <EQUAL> ', '=')

    return template_str

class TrieNode():
    def __init__(self, name, tag, is_end=False):
        self.name = name
        self.is_end = is_end
        self.tag = tag
        self.children = []

class VisualizeNode(Node):
    def __init__(self, name, tag, is_end=False, parent=None):
        super().__init__(name, parent=parent)
        self.is_end = is_end
        self.tag = tag

# Function to add a word to the Trie
def add_word(root, word, tag):
    node = root
    if str(word.split()[0]).find("<*>") != -1:
        word = " ".join(word.split()[1:])
    for char in word.split():
        child = next((n for n in node.children if n.name == char), None)
        if child is None:
            child = TrieNode(char, tag)
            node.children.append(child)
        node = child
    node.is_end = True


def visualize_trie(root):
    root_node = VisualizeNode(root.name,root.tag)
    for child in root.children:
        visualize_subtrie(child, root_node)
    return root_node


def visualize_subtrie(node, parent):
    child_node = VisualizeNode(node.name, node.tag, parent=parent)
    for child in node.children:
        visualize_subtrie(child, child_node)

def find_nodes_with_more_than_3_children(root):
    nodes = []

    for node in PreOrderIter(root):
        if isinstance(node, TrieNode) and len(node.children) > 100:
            nodes.append(node)

    return nodes
