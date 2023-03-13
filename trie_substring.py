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
        # Initialize this trie (add a root node)
        self.root = TrieNode()

    """
    Insert a word into this trie

    This is spefic to inserting substrings

    The logic is similar to word insertion 
    except that we need to keep track of 
    the depth of the string

    Parameters:
    word (str): The string to insert
    depth (int): The depth of the string
    index (int): The index of the string
    """
    def insert(self, word, depth, index):
        # Start at the root
        node = self.root
        # For each character in the word
        for char in word:
            # If the character is not a child of the current node
            if char not in node.children:
                # Create a new node and add it as a child
                node.children[char] = TrieNode()
            # Move to the child node
            node = node.children[char]
            if depth > 0:
                node.is_end_of_word = True
                node.suffix_index = index
                depth -= 1

    def search(self, word):
        node = self.root
        # iterate over characters in word
        for char in word: 
            if char not in node.children:
                return False
            node = node.children[char]
        return node

    def suffix_insert(self, word):
        for i in range(len(word)):
            # Insert the suffix of the word
            self.insert(word[i:], len(word) - i - 1, i)

def find_substring_index(node, substring_index):
    # If the current node is the end of a word, add its index to the list
    if node.is_end_of_word:
        substring_index.append(node.suffix_index)
    # Recursively look through all the child nodes
    for child_node in node.children.values():
        find_substring_index(child_node, substring_index)
    

def main():
    string = "TTTATTTTTTTAACAGTGGATGCTATGGTATATTAAAAAGTCGTGCAAACGAAAATTTAAATAAATTTATATTTATTTAAAAAAACTTTATGGAAACACTTATTTTAAAAAAGTAATGAAATTAAATTATTTTAGTACATGAATATAATAACAATAATTTTTGAATTATTTGAAAACTAAAAAGGATTAAAGATCCGTAAAATAATTTTTATTTTAAGTAACATGTTGGTGAATTTAAAGATTCTCCTTTTATTAATTTTAGTATAACCAATAGTGAAAAAGTATTGACAAAGAAAGTAGCTTTATTTAATAAAGCATTTTGAAACTGTTTTATAATGATAATAAAATTTTTTTTATTCACTTTTTGTATAACGGTCTACAATAGTGAAAAAGTATTGACAAAGAAAGTAGCTTTATTTAATAAAGCATTTTGAAACTGTTTTATAATGATAATAAAATTTTTTTTATTCACTTTTTGTATAACGGTCTAGCAAGTTATAGACTTTAGTAAATGTGTACAAAATTGTTTTTTTGCTATTTAATTTTATTAATTACCCTTTTATTAATTTTAGTATAACCAATAGTGAAAAAGTATTGACAAAGAAAGTAGCTTTATTTAATAAAGCATTTTGAAACTGTTTTATAATGATAATAAAATTTTTTTTATTCACTTTTTGTATAACGGTCTAGCAAGTTATAGACTTTAGTAAATGTGTACATCTTAAAAACAAGCACTCTAGTCTAGAGAATAACAATTCTTATTGTTGGTTAAGAATTTTATTAAAAATTTAATATATTTTTTCAAAAGTGGGCTTAATTGCAGCTATCTATTGTTAAGTGCGTAATAGCTTAATTTTATTTTAAGGTGAATTTAAAGATTTAACTAAATAATAAATTGTTTTT"
    substring = "GAAAC"
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