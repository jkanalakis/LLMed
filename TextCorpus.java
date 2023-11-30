// Copyright 2023 John Kanalakis
// LLMed | Large Language Model for Educational Understanding
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software 
// and associated documentation files (the "Software"), to deal in the Software without restriction, 
// including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do 
// so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial 
// portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT 
// NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

// The TextCorpus provides functionality to store textual data corpora for the model, including 
// utilities to access this training data, vocabulary metadata, individual words/samples, and 
// key attributes of the corpus
public class TextCorpus {

    private List<String> textData; // Stores the textual training data
    private List<String> vocabList; // Stores the vocabulary (unique words)

    // Cleans/normalizes a text sample
    private String preprocessText(String text) {
        System.out.println("   TextCorpus.preprocessText()");

        // Normalize whitespace
        text = text.replaceAll("\\s+", " ");

        // Convert to lower case
        text = text.toLowerCase();

        // Remove punctuation
        text = text.replaceAll("[,:;.?!-()]", "");

        // Expand common contractions
        text = text.replaceAll("won't", "will not");
        text = text.replaceAll("can't", "cannot");

        // Other contractions

        return text;
    }

    // Initializes the text and vocabulary storage
    public TextCorpus() {

        textData = new ArrayList<>();
        vocabList = new ArrayList<>();
    }

    // Processes text and tracks unique words
    public void updateVocabulary(String text) {
        System.out.println("   TextCorpus.updateVocabulary(" + text + ")");

        // Split the text into words
        String[] words = text.split("\\s+"); // Split by whitespace

        for (String word : words) {
            // Add each word to the vocabulary if it's not already present
            if (!vocabList.contains(word)) {
                vocabList.add(word);
            }
        }
    }

    // Loads text data from files into storage
    public void loadText(String file) {

        System.out.println("   TextCorpus.loadText()");

        try {
            BufferedReader reader = new BufferedReader(new FileReader(file));
            String line;
            while ((line = reader.readLine()) != null) {

                // Clean text
                String cleaned = preprocessText(line);

                // Update data and vocab
                textData.add(cleaned);
                updateVocabulary(cleaned);

            }
            reader.close();

        } catch (IOException e) {
            System.out.println("Error reading file.");
            e.printStackTrace();
        }

    }

    // Retrieves current vocabulary size
    public int getVocabSize() {

        return vocabList.size();
    }

    // Looks up index for a given word
    public int getVocabIndex(String word) {

        return vocabList.indexOf(word);
    }

    // Retrieves word for given index
    public String getVocabWord(int index) {

        if (index >= 0 && index < vocabList.size()) {
            return vocabList.get(index);
        }

        return null;
    }

    // Looks up index for given word
    public int getWordIndex(String word) {

        return vocabList.indexOf(word);
    }

    // Gets number of text samples loaded
    public int getTextSize() {

        return textData.size();
    }

    // Accesses stored text sample by index
    public String getTextSample(int index) {

        return textData.get(index);
    }

}