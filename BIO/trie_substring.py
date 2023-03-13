import pydot
import graphviz


class TrieNode:
    def __init__(self):
        # A dictionary where keys represent characters and values represent nodes
        self.children = {} 
        # A boolean to indicate if the node represents the end of a word
        self.is_end_of_word = False 
        # An integer to store the index of the suffix if the node represents the end of a word
        self.suffix_index = -1 


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, depth, index):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            if depth > 0:
                node.is_end_of_word = True
                node.suffix_index = index
                depth -= 1

    def search(self, word):
        node = self.root
        for char in word: 
            if char not in node.children:
                return False
            node = node.children[char]
        return node

    def suffix_insert(self, word):
        for i in range(len(word)):
            self.insert(word[i:], len(word) - i - 1, i)

def find_substring_index(node, substring_index):
    if node.is_end_of_word:
        substring_index.append(node.suffix_index)
    for child_node in node.children.values():
        find_substring_index(child_node, substring_index)
    

def main():
    import sys
    # set the recursion limit to a high value
    sys.setrecursionlimit(10**6)
    string = "GATATATGCATATACTT"
    substring = "GAATTC"
    trie = Trie()
    trie.suffix_insert(string)
    search_result = trie.search(substring)
    if not search_result:
        print("substring not found")
    else:
        substring_index = []
        find_substring_index(search_result, substring_index)
        substring_index = list(set(substring_index))
        print("substring found at index:", substring_index)
        # Here we are extracting the substring with the index location to see if its right
        print("Substring:", string[substring_index[0]:substring_index[0] + len(substring)])

if __name__ == "__main__":
    main()