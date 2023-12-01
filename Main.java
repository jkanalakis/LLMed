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

import java.util.Scanner;

public class Main {

    public static void main(String[] args) {

        System.out.println("==========================================================");
        System.out.println("LLMed | Large Language Model for Educational Understanding");
        System.out.println("==========================================================");

        // Create text corpus
        TextCorpus corpus = new TextCorpus();
        System.out.println("Load corpus text");
        corpus.loadText("A-Tale-of-Two-Cities-by-Charles-Dickens.txt");

        // Specify network configuration
        System.out.println("Create layers, vocabulary size is: " + corpus.getVocabSize());
        int[] layers = { corpus.getVocabSize(), 500, 250, corpus.getVocabSize() };

        // Construct network
        System.out.println("Create NeuralNetwork");
        int embeddingSize = 250;
        NeuralNetwork network = new NeuralNetwork(layers);
        network.initializeEmbeddingWeights(corpus.getVocabSize(), embeddingSize);

        // Create text generator
        System.out.println("Create TextGenerator");
        TextGenerator generator = new TextGenerator(network, corpus, embeddingSize);

        // Initialize trainer
        System.out.println("Create ModelTrainer");
        ModelTrainer trainer = new ModelTrainer(corpus, network, generator);

        // Train
        System.out.println("Train model");
        trainer.trainModel(10);

        // Test completion
        System.out.println("\n\nComplete text: In his expostulation he dropped his cleaner hand...");
        String completed = generator.completeText("In his expostulation he dropped his cleaner hand", 65);
        System.out.println(completed);

        // Test generation
        System.out.println("\n\nGenerate some text: This dialogue had been held in so very low a whisper...");
        String generated = generator.generateText("This dialogue had been held in so very low a whisper", 24);
        System.out.println(generated);

        // Prompt user to enter text for completion
        Scanner scanner = new Scanner(System.in);

        while (true) {

            System.out.print("\n\nEnter text to complete (or quit): ");
            String input = scanner.nextLine();

            if (input.equalsIgnoreCase("quit")) {
                break;
            }

            String response = generator.completeText(input, 25);

            System.out.println("\n" + response);
        }

        scanner.close();
    }

}