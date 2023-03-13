import java.util.HashMap;
import java.util.Vector;

class Trie {

    // Data members.
    HashMap<Character, Trie> next = new HashMap<Character, Trie>();
    Vector<Integer> indexes = new Vector<Integer>();

    // Trie Constructor to populate the suffix tree.
    Trie(String baseText) {
        int length = baseText.length();
        for (int currentIndex = 0; currentIndex < length; currentIndex++) {
            String suffix = baseText.substring(currentIndex);
            insertSuffix(suffix, currentIndex);
        }
    }

    Trie() {
        // Trie Constructor to create an empty Trie node.
    }

    void insertSuffix(String suffix, int currentIndex) {
        Trie currentNode = this;
        for (char letter : suffix.toCharArray()) {
            if (!currentNode.next.containsKey(letter)) {
                currentNode.next.put(letter, new Trie());
            }
            currentNode = currentNode.next.get(letter);
            currentNode.indexes.add(currentIndex);
        }
    }

    void searchPattern(String pattern) {
        Trie currentNode = this;
        for (char letter : pattern.toCharArray()) {
            if (!currentNode.next.containsKey(letter)) {
                System.out.println("Pattern not found\n");
                return;
            }
            currentNode = currentNode.next.get(letter);
        }
        print(pattern, currentNode);
    }

    void print(String pattern, Trie currentNode) {
        System.out.println(pattern + " found at: ");
        for (int index : currentNode.indexes) {
            System.out.print(index + " ");
        }
        System.out.println();
    }
}

public class Main {
    public static void main(String[] args) {
        String baseText;

        baseText = "Lorem Ipsum is simply dummy text of the printing and typesetting industry Lorem Ipsum has been the industrys standard dummy text ever since the when an unknown printer took a galley of type and scrambled it to make a type specimen book It has survived not only five centuries but also the leap into electronic typesetting remaining essentially unchanged It was popularised in the with the release of Letraset sheets containing Lorem Ipsum passages and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum";

        // Trie object with name suffixTree.
        Trie suffixTree = new Trie(baseText);

        suffixTree.searchPattern("peci");
    }
}
