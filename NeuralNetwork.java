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

import java.util.ArrayList;
import java.util.List;

// The NeuralNetwork class defines the overall architecture and training logic for the neural 
// net, including initializing the layered topology, handling forward and backward passes, 
// making predictions, and encapsulating the learning related hyperparameters
public class NeuralNetwork {

    private List<DenseLayer> layers; // Stores the sequence of DenseLayer objects that defines the network topology
    private double learningRate = 0.01; // Learning rate hyperparameter for gradient updates
    private double[][] embeddingWeights; // Dimensionality of feature vectors representing words

    // Runs a forward pass, chaining layer propagations
    private double[] forwardPropagation(double[] input) {

        double[] output = input;

        for (DenseLayer layer : layers) {
            output = layer.forwardPropagation(output);

        }
        return output;

    }

    // Instantiates layers based on provided layer sizes
    public NeuralNetwork(int[] shape) {

        layers = new ArrayList<>(shape.length - 1);

        for (int i = 1; i < shape.length; i++) {

            int inputSize = shape[i - 1]; // Set input size to the number of nodes in the previous layer
            layers.add(new DenseLayer(shape[i], inputSize, learningRate)); // shape[i] is the number of nodes in the
                                                                           // current layer
        }
    }

    // Initialize embeddingWeights
    public void initializeEmbeddingWeights(int vocabSize, int embeddingSize) {

        this.embeddingWeights = new double[vocabSize][embeddingSize];

        for (int i = 0; i < vocabSize; i++) {
            for (int j = 0; j < embeddingSize; j++) {
                this.embeddingWeights[i][j] = Math.random(); // Or some other initialization
            }
        }
    }

    public double[] getWordEmbedding(int wordIndex) {

        if (embeddingWeights == null) {
            throw new IllegalStateException("Embedding weights have not been initialized.");
        }
        if (wordIndex < 0 || wordIndex >= embeddingWeights.length) {
            throw new IllegalArgumentException("Invalid word index");
        }

        return embeddingWeights[wordIndex];
    }

    // Compares output and target arrays to compute error signal
    public double[] calculateError(double[] output, double[] target) {

        if (output.length != target.length) {
            throw new IllegalArgumentException("Output and target arrays must have the same length");
        }

        double[] errors = new double[output.length];

        for (int i = 0; i < output.length; i++) {
            errors[i] = target[i] - output[i];
        }
        return errors;
    }

    // Performs a forward pass, backward pass, and layer update loop
    public void train(double[] input, double[] target) {

        // Forward pass
        double[] output = forwardPropagation(input);

        // Backward pass
        double[] error = calculateError(output, target);

        for (int i = layers.size() - 1; i >= 0; --i) {
            // Update each layer using backpropagation
            layers.get(i).backwardPropagation(error);

            // Calculate the error for the previous layer, if necessary
            if (i > 0) {
                error = layers.get(i - 1).calculatePreviousLayerError(error); // pass the current error
            }
        }
    }

    // Runs a forward pass to generate output for given input
    public double[] predict(double[] input) {

        return forwardPropagation(input);
    }

}