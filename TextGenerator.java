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

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;

// Responsible for integrating with the trained neural network to encode input text into vector 
// representations, make probabilistic predictions for the next words, decode the output vectors 
// back into readable text, and handle generating or completing text sequences
public class TextGenerator {

    private NeuralNetwork network; // Reference to the trained neural network model
    private TextCorpus corpus; // Reference to the text corpus
    private Random rng; // Random number generator
    private int embeddingSize; // Size of the word embeddings
    private double temperature;

    // Helper to locate maximum value index in a vector
    private int findMaxIndex(double[] vector) {

        double maxValue = Double.NEGATIVE_INFINITY;
        int maxIndex = -1;

        for (int i = 0; i < vector.length; i++) {

            if (vector[i] > maxValue) {
                maxValue = vector[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }

    // Samples next word index based on prediction probabilities
    private int sampleTopKFromDistribution(double[] distribution, int k) {

        // Create a list of word indices and their corresponding probabilities
        List<Map.Entry<Integer, Double>> candidates = new ArrayList<>();

        for (int i = 0; i < distribution.length; i++) {
            candidates.add(new AbstractMap.SimpleEntry<>(i, distribution[i]));
        }

        // Sort and retain top k candidates
        candidates.sort((e1, e2) -> Double.compare(e2.getValue(), e1.getValue()));
        candidates = candidates.subList(0, Math.min(k, candidates.size()));

        // Normalize probabilities of top k candidates
        double sum = candidates.stream().mapToDouble(Map.Entry::getValue).sum();

        for (Map.Entry<Integer, Double> entry : candidates) {
            entry.setValue(entry.getValue() / sum);
        }

        // Sample from top k candidates
        double randomValue = rng.nextDouble();
        double cumulativeProbability = 0.0;

        for (Map.Entry<Integer, Double> entry : candidates) {
            cumulativeProbability += entry.getValue();
            if (randomValue < cumulativeProbability) {
                return entry.getKey();
            }
        }

        // Fallback (should not happen, but just in case)
        return candidates.get(rng.nextInt(candidates.size())).getKey();
    }

    private double[] softmaxWithTemperature(double[] logits) {

        double[] softened = new double[logits.length];
        double sum = 0;

        for (int i = 0; i < logits.length; i++) {
            softened[i] = Math.exp(logits[i] / temperature);
            sum += softened[i];
        }

        for (int i = 0; i < softened.length; i++) {
            softened[i] /= sum;
        }

        return softened;
    }

    // Initializes key references and rng
    public TextGenerator(NeuralNetwork network, TextCorpus corpus, int embeddingSize, double temperature) {

        this.network = network;
        this.corpus = corpus;
        this.embeddingSize = embeddingSize;
        this.temperature = temperature;
        this.rng = new Random(); // For sampling
    }

    // Encodes text into an embedding vector
    public double[] encodeText(String text) {

        String[] words = text.split(" ");
        double[] embedding = new double[embeddingSize * words.length];

        for (int i = 0; i < words.length; i++) {

            int index = corpus.getWordIndex(words[i]);

            if (index != -1) {

                double[] wordEmbedding = network.getWordEmbedding(index);
                System.arraycopy(wordEmbedding, 0, embedding, i * embeddingSize, embeddingSize);
            }
        }

        return embedding;
    }

    // Encodes text into one-hot vector
    public double[] encodeTextAsVector(String text) {

        int vocabSize = corpus.getVocabSize(); // Get the size of the vocabulary
        double[] vector = new double[vocabSize];

        int index = corpus.getWordIndex(text); // Get the index of the word in the vocabulary

        if (index != -1) {

            vector[index] = 1.0; // Set the element at the index of the word to 1
        }

        return vector;
    }

    // Decodes output vector into words
    public String decodeVector(double[] vector) {

        String decoded = "";
        int maxIndex = findMaxIndex(vector);
        int iterations = 0;

        while (maxIndex != -1 && iterations < vector.length) { // Avoid infinite loops

            String word = corpus.getVocabWord(maxIndex);
            decoded += " " + word;
            vector[maxIndex] = Double.NEGATIVE_INFINITY; // Mark as used and ensure it's the lowest value
            maxIndex = findMaxIndex(vector);
            iterations++;
        }

        return decoded.trim(); // Trim to remove leading space
    }

    // Gets next word predictions from network
    public double[] predictNext(double[] encodedText) {

        double[] output = network.predict(encodedText);
        return softmaxWithTemperature(output);
    }

    // Generates text token by token using network
    public String generateText(String initialText, int length) {

        double[] encoded = encodeText(initialText);
        String generated = initialText;

        for (int i = 0; i < length; i++) {

            double[] output = predictNext(encoded);

            int sampleIndex = sampleTopKFromDistribution(output, 10);

            String word = corpus.getVocabWord(sampleIndex);
            generated += " " + word;
            encoded = encodeText(word); // Update encoding with the new word
        }

        return generated;
    }

    // Predicts completion for partial text
    public String completeText(String initialText, int maxLength) {

        // Encode seed text
        double[] encoded = encodeText(initialText);

        // Predict next words
        double[] output = predictNext(encoded);

        // Decode predictions
        String completion = decodeVector(output);

        // Combine initial text with completion
        String combinedText = initialText + " " + completion;

        // Split the text into words and limit to 25 words
        String[] words = combinedText.split("\\s+"); // Split on spaces
        StringBuilder limitedText = new StringBuilder();

        for (int i = 0; i < Math.min(words.length, maxLength); i++) {

            limitedText.append(words[i]).append(" ");
        }

        // Return the truncated or limited text
        return limitedText.toString().trim(); // Trim to remove the last space
    }

}