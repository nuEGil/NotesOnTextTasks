/*This script can be compiled into an executable that 
reads a verry specific text file and looks for very specific 
words from a word list. then returns a json file with 
everythin saved. Just need to modify 
so that instead of a textfile load, it already has a loaded 
string. And instead of a hard coded word list, it 
gets a word list from the user..... hm. excecutable should 
just have like an arg parse or something. but will update 
later. */


#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <utility> // For pair
#include <algorithm> // For transform
#include <cctype>    // For tolower
#include <map>       // For map
#include <chrono>    // For timing functions

using namespace std;
using namespace chrono;

string loadFile(const string& filename) {
    auto start = high_resolution_clock::now();

    ifstream fin(filename);
    if (!fin.is_open()) {
        throw runtime_error("Could not open file: " + filename);
    }

    stringstream buffer;
    buffer << fin.rdbuf();
    string text = buffer.str();

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    cout << text << "Loaded " << filename << " in " 
         << duration.count() << " ms\n";

    return text;
}

// Define the number of characters in our alphabet
const int ALPHABET_SIZE = 26;

// TrieNode structure represents a single node in the trie.
struct TrieNode {
    // Array of pointers to child nodes. Each index corresponds to a letter 'a' through 'z'.
    TrieNode* children[ALPHABET_SIZE];
    
    // isEndOfWord is true if the node represents the end of a word.
    bool isEndOfWord;

    // Constructor to initialize a new TrieNode.
    TrieNode() {
        // Initialize all children pointers to nullptr.
        for (int i = 0; i < ALPHABET_SIZE; i++) {
            children[i] = nullptr;
        }
        // A new node is not the end of a word by default.
        isEndOfWord = false;
    }
};

// Trie class encapsulates the Trie data structure and its operations.
class Trie {
private:
    TrieNode* root;

public:
    // Constructor to initialize the Trie with an empty root node.
    Trie() {
        root = new TrieNode();
    }

    // Inserts a word into the Trie.
    void insert(string key) {
        TrieNode* current = root;
        for (char ch : key) {
            // Convert character to a 0-25 index. This code assumes lowercase inputs.
            int index = ch - 'a';
            // If the child node for the current character doesn't exist, create it.
            if (current->children[index] == nullptr) {
                current->children[index] = new TrieNode();
            }
            // Move to the next node in the path.
            current = current->children[index];
        }
        // Mark the last node as the end of a word.
        current->isEndOfWord = true;
    }

    // Searches for a word in the Trie. Returns true if the word is found, otherwise false.
    bool search(string key) {
        TrieNode* current = root;
        for (char ch : key) {
            int index = ch - 'a';
            // If the child node doesn't exist, the word is not in the Trie.
            if (current->children[index] == nullptr) {
                return false;
            }
            // Move to the next node.
            current = current->children[index];
        }
        // The word is found only if we reached the end of the string AND the last node is marked as the end of a word.
        return (current != nullptr && current->isEndOfWord);
    }

    // Checks if any word in the Trie starts with the given prefix.
    bool startsWith(string prefix) {
        TrieNode* current = root;
        for (char ch : prefix) {
            int index = ch - 'a';
            // If the child node doesn't exist, no word has this prefix.
            if (current->children[index] == nullptr) {
                return false;
            }
            // Move to the next node.
            current = current->children[index];
        }
        // We found a path for the entire prefix.
        return true;
    }

    // Finds all occurrences of words from the trie within a given text.
    // Returns a vector of pairs, where each pair contains the starting index and the length of a found word.
    vector<pair<int, int>> findWordsInText(const string& text) {
        vector<pair<int, int>> foundWords;
        int n = text.length();

        // Iterate through each character of the text, treating it as a potential start of a word.
        for (int i = 0; i < n; i++) {
            TrieNode* current = root;
            // From the current position, traverse the text and the trie simultaneously.
            for (int j = i; j < n; j++) {
                int index = text[j] - 'a';
                // If the character is not a lowercase letter or the path doesn't exist in the trie, break.
                if (index < 0 || index >= ALPHABET_SIZE || current->children[index] == nullptr) {
                    break;
                }
                
                // Move to the next node in the trie.
                current = current->children[index];

                // If the current trie node marks the end of a word, we've found a match.
                if (current->isEndOfWord) {
                    // Record the starting index (i) and the length of the found word (j - i + 1).
                    foundWords.push_back({i, j - i + 1});
                }
            }
        }
        return foundWords;
    }
    
    // A function to delete the Trie and free memory.
    void freeTrie(TrieNode* node) {
        if (!node) {
            return;
        }
        for (int i = 0; i < ALPHABET_SIZE; i++) {
            freeTrie(node->children[i]);
        }
        delete node;
    }
    
    // Destructor to clean up the allocated memory.
    ~Trie() {
        freeTrie(root);
    }
};


// Function to process a given text using a word list and a Trie.
map<string, vector<int>> processTextWithTrie(const vector<string>& wordlist, const string& contents) {
    Trie myTrie;
    map<string, vector<int>> results;

    cout << "Inserting words into Trie..." << endl;
    auto start_insert = high_resolution_clock::now();
    for (const string& word : wordlist) {
        myTrie.insert(word);
        cout << "  - Inserted '" << word << "'" << endl;
    }
    auto end_insert = high_resolution_clock::now();
    duration<double, milli> elapsed_insert = end_insert - start_insert;
    cout << "Trie building time: " << elapsed_insert.count() << " ms" << endl;

    cout << "\nScanning text for words from the list..." << endl;
    auto start_find = high_resolution_clock::now();
    vector<pair<int, int>> found = myTrie.findWordsInText(contents);
    auto end_find = high_resolution_clock::now();
    duration<double, milli> elapsed_find = end_find - start_find;
    cout << "Text search time: " << elapsed_find.count() << " ms" << endl;

    auto start_map = high_resolution_clock::now();
    if (found.empty()) {
        cout << "  - No words from the list found in the text." << endl;
    } else {
        cout << "\nFound the following words:" << endl;
        for (const auto& match : found) {
            string foundWord = contents.substr(match.first, match.second);
            cout << "  - Found word starting at index " << match.first
                      << " with length " << match.second << ": '" << foundWord << "'" << endl;
            // Add the starting index to the map for this word.
            results[foundWord].push_back(match.first);
        }
    }
    auto end_map = high_resolution_clock::now();
    duration<double, milli> elapsed_map = end_map - start_map;
    cout << "Result mapping time: " << elapsed_map.count() << " ms" << endl;
    
    return results;
}

// Function to save the word locations map to a JSON file.
void saveMapToJson(const map<string, vector<int>>& data, const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        return;
    }

    file << "{\n";
    bool firstEntry = true;
    for (const auto& pair : data) {
        if (!firstEntry) {
            file << ",\n";
        }
        file << "  \"" << pair.first << "\": [";
        bool firstIndex = true;
        for (int index : pair.second) {
            if (!firstIndex) {
                file << ", ";
            }
            file << index;
            firstIndex = false;
        }
        file << "]";
        firstEntry = false;
    }
    file << "\n}";
    file.close();
    cout << "\nSuccessfully wrote JSON data to " << filename << endl;
}

int main() {
    auto full_start = high_resolution_clock::now();
    
    string filename = "crime and punishment.txt";  // or any other file
    string contents = loadFile(filename); // not sure if variable collides. but whatever 
    
    // convert to lower case 
    
    transform(contents.begin(), contents.end(), contents.begin(),
    [](unsigned char c){ return tolower(c); });

    cout << "File size: " << contents.size() << " characters\n";

    // Words to insert into the Trie.
    vector<string> wordlist = { 
                                "rodion", "pulcheria", "alexandrovna", 
                                "dounia", "raskolnikov", "romanovitch", 
                                "porfiry", "pyotr", "petrovitch",
                                "dmitri", "prokofitch", "sofya", "semyonovna", 
                                "marmeladov","amalia", "fyodorovna",
                                "lebeziatnikov","darya","frantsovna", "katerina", "ivanovna"
                               };
    // I want to build a trie using these words. 
    map<string, vector<int>> wordLocations = processTextWithTrie(wordlist, contents);

    // And then save to json 
    saveMapToJson(wordLocations, "word match locs.json");
 
    auto full_end = high_resolution_clock::now();
    auto full_duration = duration_cast<milliseconds>(full_end - full_start);

    cout << "Full program computed in  " << full_duration.count() << " ms\n";

    return 0;
}