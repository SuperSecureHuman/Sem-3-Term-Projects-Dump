import java.util.HashMap;
import java.util.Vector;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

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
        String string_path;
        // read the base text from the file dnatext.txt
        string_path = "/home/venom/repo/sem_proj/BIO/dnatext.txt";
        baseText = "";
        try {
            File file = new File(string_path);
            Scanner sc = new Scanner(file);
            while (sc.hasNextLine()) {
                baseText += sc.nextLine();
            }
            sc.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        // Trie object with name suffixTree.
        Trie suffixTree = new Trie(baseText);
        
        suffixTree.searchPattern("CCTC"); 
        System.out.println();
        suffixTree.searchPattern("GAATTC");
        System.out.println();
        suffixTree.searchPattern("CTTAAG");
        System.out.println();
        suffixTree.searchPattern("GAATTC");
    }
}
