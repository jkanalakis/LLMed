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

// handles fetching batch data examples from the TextCorpus data source, forwarding them through
// the NeuralNetwork to train the model weights via backpropagation based on the loss, and 
// tracking the model accuracy over training epochs
public class ModelTrainer {

    private TextCorpus corpus; // Stores reference to the TextCorpus data source
    private NeuralNetwork network; // Stores reference to the NeuralNetwork to train
    private int batchSize; // Batch size hyperparameter
    private Random rng; // Random number generator, used for sampling batches
    private TextGenerator textGenerator; // Used to encode text to vector representations

    // Retrieves next batch of training examples from corpus
    private double[][][] getNextBatch() {

        int vocabSize = corpus.getVocabSize();
        double[][][] batch = new double[batchSize][2][vocabSize]; // 3D array to hold pairs of input and target

        for (int i = 0; i < batchSize; i++) {
            int index = rng.nextInt(corpus.getTextSize());
            String text = corpus.getTextSample(index);
            batch[i][0] = textGenerator.encodeTextAsVector(text); // First element of the pair is input

            String nextWord = text.split(" ")[1];
            batch[i][1] = textGenerator.encodeTextAsVector(nextWord); // Second element of the pair is target
        }

        return batch;
    }

    // Initializes trainer with corpus and network references
    public ModelTrainer(TextCorpus corpus, NeuralNetwork network) {

        this.corpus = corpus;
        this.network = network;
    }

    // Main training loop - gets batches, trains network
    public void trainModel(int epochs) {

        for (int i = 0; i < epochs; i++) {
            double[][][] batch = getNextBatch();
            for (int j = 0; j < batchSize; j++) {
                double[] inputExample = batch[0][j]; // j-th example in input
                double[] targetExample = batch[1][j]; // j-th example in target
                network.train(inputExample, targetExample);
            }
        }
    }

}