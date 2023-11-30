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

public class Main {

    public static void main(String[] args) {

        System.out.println("==========================================================");
        System.out.println("LLMed | Large Language Model for Educational Understanding");
        System.out.println("==========================================================");

        // Create text corpus
        TextCorpus corpus = new TextCorpus();
        System.out.println("Load text");
        corpus.loadText("data.txt");

        // Specify network configuration
        System.out.println("Create layers");
        int[] layers = { 200, 100, corpus.getVocabSize() };

        // Construct network
        System.out.println("Create NeuralNetwork");
        NeuralNetwork network = new NeuralNetwork(layers);

        // Initialize trainer
        System.out.println("Create ModelTrainer");
        ModelTrainer trainer = new ModelTrainer(corpus, network);

        // Train
        System.out.println("Train model");
        trainer.trainModel(10);

        // Create text generator
        System.out.println("Create TextGenerator");
        TextGenerator generator = new TextGenerator(network, corpus);

        // Test generation
        System.out.println("Generate some text");
        String generated = generator.generateText("The dog", 10);
        System.out.println(generated);

        // Test completion
        System.out.println("Complete some text");
        String completed = generator.completeText("The quick brown ");
        System.out.println(completed);
    }

}