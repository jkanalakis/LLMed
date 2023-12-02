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
    private int batchSize; // Stores reference to the batch size
    private Random rng; // Random number generator, used for sampling batches
    private TextGenerator textGenerator; // Used to encode text to vector representations

    // Retrieves next batch of training examples from corpus
    private double[][][] getNextBatch() {

        int vocabSize = corpus.getVocabSize();
        double[][][] batch = new double[batchSize][2][vocabSize]; // 3D array to hold pairs of input and target

        for (int i = 0; i < batchSize; i++) {

            int index = rng.nextInt(corpus.getTextSize() - 1); // Ensure there's a next word

            String[] words = corpus.getTextSample(index).split(" ");

            if (words.length >= 2) {

                String currentWord = words[0]; // Current word
                String nextWord = words[1]; // Next word

                batch[i][0] = textGenerator.encodeTextAsVector(currentWord); // Encode current word as input
                batch[i][1] = textGenerator.encodeTextAsVector(nextWord); // Encode next word as target

                // System.out.println("Input: " + currentWord + ", Target: " + nextWord);
            } else {

                // Handle edge case where there's only one word or empty text
                i--; // Redo this iteration with a different random index
            }
        }

        return batch;
    }

    // Calculate the mean error of the predictions for each batch
    private double computeMeanError(double[] errors) {

        double sum = 0;

        for (double error : errors) {

            sum += Math.abs(error); // Using absolute error; you could also square the errors for Mean Squared Error
        }

        return sum / errors.length;
    }

    // Initializes trainer with corpus and network references
    public ModelTrainer(TextCorpus corpus, NeuralNetwork network, TextGenerator textGenerator, int batchSize) {

        this.corpus = corpus;
        this.network = network;
        this.rng = new Random();
        this.textGenerator = textGenerator;
        this.batchSize = batchSize;
    }

    // Main training loop - gets batches, trains network
    public void trainModel(int epochs) {

        for (int epoch = 0; epoch < epochs; epoch++) {

            double totalError = 0;

            for (int batch = 0; batch < batchSize; batch++) {

                double[][][] batchData = getNextBatch();
                double batchError = 0;

                for (int j = 0; j < batchSize; j++) {

                    double[] input = batchData[j][0];
                    double[] target = batchData[j][1];
                    network.train(input, target);
                    double[] output = network.predict(input);
                    double[] error = network.calculateError(output, target);
                    batchError += computeMeanError(error); // Implement computeMeanError to calculate mean error
                }

                batchError /= batchSize;
                totalError += batchError;
                System.out.println("Epoch " + (epoch + 1) + ", Batch " + (batch + 1) +
                        ", Average Error: " + batchError);
            }

            double averageEpochError = totalError / batchSize;
            System.out.println("End of Epoch " + (epoch + 1) + ", Average Error: " + averageEpochError);
        }
    }

}