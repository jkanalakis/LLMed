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

import java.util.Random;

// Responsible for integrating with the trained neural network to encode input text into vector 
// representations, make probabilistic predictions for the next words, decode the output vectors 
// back into readable text, and handle generating or completing text sequences
public class TextGenerator {

    private NeuralNetwork network; // Reference to the trained neural network model
    private TextCorpus corpus; // Reference to the text corpus
    private Random rng; // Random number generator

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
    private int sampleFromDistribution(double[] distribution) {

        double sum = 0;
        double randomValue = rng.nextDouble();

        for (int i = 0; i < distribution.length; i++) {
            sum += distribution[i];
            if (randomValue < sum) {
                return i;
            }
        }

        // Should not reach here
        return -1;

    }

    // Initializes key references and rng
    public TextGenerator(NeuralNetwork network, TextCorpus corpus) {

        this.network = network;
        this.corpus = corpus;
        this.rng = new Random(); // For sampling
    }

    // Encodes text into one-hot vector
    public double[] encodeTextAsVector(String text) {

        int vocabSize = corpus.getVocabSize(); // Get the size of the vocabulary
        double[] vector = new double[vocabSize]; // Initialize a vector of size vocabSize with all 0s

        int index = corpus.getWordIndex(text); // Method to get the index of the word in the vocabulary
        if (index != -1) {
            vector[index] = 1.0; // Set the element at the index of the word to 1
        }

        return vector;
    }

    // Encodes text into bag-of-words vector
    public double[] encodeText(String text) {

        double[] encoded = new double[corpus.getVocabSize()];

        String[] words = text.split(" ");

        for (String word : words) {

            int index = corpus.getVocabIndex(word);

            // Ignore unknown words
            if (index != -1) {
                encoded[index] += 1;
            }

        }

        return encoded;

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

        // Add batch dimension
        double[][] input = new double[1][encodedText.length];
        input[0] = encodedText;

        // Get output prediction
        double[][] output = new double[input.length][];
        for (int i = 0; i < input.length; i++) {
            output[i] = network.predict(input[i]); // Call predict on each 1D sub-array
        }

        // Remove batch dimension
        return output[0];

    }

    // Generates text token by token using network
    public String generateText(String initialText, int length) {

        // Encode seed text
        double[] encoded = encodeText(initialText);

        // Output string
        String generated = initialText;

        // Generate text token-by-token
        for (int i = 0; i < length; i++) {

            // Predict next token
            double[] output = predictNext(encoded);

            // Sample from distribution
            int sample = sampleFromDistribution(output);

            // Decode and append to output
            String word = corpus.getVocabWord(sample);
            generated += " " + word;

            // Update encoded vector
            encoded = encodeText(word);

        }

        return generated;

    }

    // Predicts completion for partial text
    public String completeText(String initialText) {

        // Encode seed text
        double[] encoded = encodeText(initialText);

        // Predict next words
        double[] output = predictNext(encoded);

        // Decode predictions
        String completion = decodeVector(output);

        // Return completed text
        return initialText + " " + completion;

    }

}