import concurrent.futures

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
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


def search_in_parallel(trie, substring):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        search_result = list(executor.map(trie.search, substring))
    substring_index = []
    flag = False
    for node in search_result:
        if node:
            find_substring_index(node, substring_index, substring, flag)
    substring_index = list(set(substring_index))
    print("substring found at index:", substring_index)




def main():
    letters = "TTTATTTTTTTAACAGTGGATGCTATGGTATATTAAAAAGTCGTGCAAACGAAAATTTAAATAAATTTATATTTATTTAAAAAAACTTTATGGAAACACTTATTTTAAAAAAGTAATGAAATTAAATTATTTTAGTACATGAATATAATAACAATAATTTTTGAATTATTTGAAAACTAAAAAGGATTAAAGATCCGTAAAATAATTTTTATTTTAAGTAACATGTTGGTGAATTTAAAGATTCTCCTTTTATTAATTTTAGTATAACCAATAGTGAAAAAGTATTGACAAAGAAAGTAGCTTTATTTAATAAAGCATTTTGAAACTGTTTTATAATGATAATAAAATTTTTTTTATTCACTTTTTGTATAACGGTCTAGCAAGTTATAGACTTTAGTAAATGTGTACAAAATTGTTTTTTTGCTATTTAATTTTATTAATTACGTGAATTTAAAGATTCTCCTTTTATTAATTTTAGTATAACCAATAGTGAAAAAGTATTGACAAAGAAAGTAGCTTTATTTAATAAAGCATTTTGAAACTGTTTTATAATGATAATAAAATTTTTTTTATTCACTTTTTGTATAACGGTCTAGCAAGTTATAGACTTTAGTAAATGTGTACAAAATTGTTTTTTTGCTATTTAATTTTATTAATTACCTAAGTCAATAATAAGTATGAACCGGTGTTTGTTGCAAAAAGCTCGGATGAACTTTGTGTAGCCGCGATACATGAATCGCATTTTGTAATAGCTGGTTTTTCACGAAATTTTATTTTAGTATAATTATTTTAATTAAAATTGACTGTTATTTGTTTTTTGGTTTTATCTTAAAAACTAACTAAATAATGTTAATTTATATTTAAGGCAGACTATTTGTGGTTAAGCACTCTAGTCTAGAGAATAACAATTCTTATTGTTGGTTAAGAATTTTATTAAAAATTTAATATATTTTTTCAAAAGTGGGCTTAATTGCAGCTATCTATTGTTAAGTGCGTAATAGCTTAATTTTATTTTAAGGTGAATTTAAAGATTCTCCTTTTATTAATTTTAGTATAACCAATAGTGAAAAAGTATTGACAAAGAAAGTAGCTTTATTTAATAAAGCATTTTGAAACTGTTTTATAATGATAATAAAATTTTTTTTATTCACTTTTTGTATAACGGTCTAGCAAGTTATAGACTTTAGTAAATGTGTACAAAATTGTTTTTTTGCTATTTAATTTTATTAATTACTAACATGTTGTGAATTTAAAGATTCTGTGAATTTAAAGATTCTCCTTTTATTAATTTTAGTATAACCAATAGTGAAAAAGTATTGACAAAGAAAGTAGCTTTATTTAATAAAGCATTTTGAAACTGTTTTATAATGATAATAAAATTTTTTTTATTCACTTTTTGTATAACGGTCTAGCAAGTTATAGACTTTAGTAAATGTGTACAAAATTGTTTTTTTGCTATTTAATTTTATTAATTACCCTTTTATTAATTTTAGTATAACCAATAGTGAAAAAGTATTGACAAAGAAAGTAGCTTTATTTAATAAAGCATTTTGAAACTGTTTTATAATGATAATAAAATTTTTTTTATTCACTTTTTGTATAACGGTCTAGCAAGTTATAGACTTTAGTAAATGTGTACATCTTAAAAACAAGCACTCTAGTCTAGAGAATAACAATTCTTATTGTTGGTTAAGAATTTTATTAAAAATTTAATATATTTTTTCAAAAGTGGGCTTAATTGCAGCTATCTATTGTTAAGTGCGTAATAGCTTAATTTTATTTTAAGGTGAATTTAAAGATTTAACTAAATAATAAATTGTTTTT"
    substring = "ATCG"
    trie = Trie()
    trie.suffix_insert(letters)
    search_in_parallel(trie, substring)


def find_substring_index(node, substring_index, substring, flag):
    if node.is_end_of_word and not flag:
        substring_index.append(node.suffix_index - len(substring) + 1)
        flag = True
    for child_node in node.children.values():
        find_substring_index(child_node, substring_index, substring, flag)


if __name__ == "__main__":
    main()
