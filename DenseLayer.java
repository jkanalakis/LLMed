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

import java.io.Serializable;

// The DenseLayer defines a fully-connected neural network layer with input/output nodes, 
// weight parameters, bias terms, and methods for the forward and backpropagation computational
// logic during network operations
public class DenseLayer implements Serializable {

    private static final long serialVersionUID = 1L; // Add a serialVersionUID

    private int numNodes; // Stores the number of nodes (neurons) in this dense layer
    private double[][] weights; // The weight parameters between inputs and output nodes
    private double[] biases; // The bias terms associated with each output node
    private double[] inputs; // Cache of input values from last forward pass
    private double[] outputs; // Cache of output values from last forward pass
    private double learningRate; // The learning rate hyperparameter for gradient updates

    // Computes standard sigmoid activation function
    private double sigmoid(double x) {

        return 1 / (1 + Math.exp(-x));
    }

    // Computes derivative of sigmoid for backPropagation
    private double derivativeSigmoid(double x) {

        double sigmoid = sigmoid(x);
        return sigmoid * (1 - sigmoid);
    }

    // Randomly initializes weight and bias values
    private void initializeWeights() {

        for (int i = 0; i < numNodes; i++) {

            for (int j = 0; j < weights[i].length; j++) {

                // Random initialization to prevent each neuron in the network layer from
                // learning the same features during training
                weights[i][j] = Math.random();
            }

            biases[i] = Math.random();
        }
    }

    // Performs gradient update of parameters
    private void updateWeightsAndBiases(double[][] weightGradients, double[] biasGradients) {

        for (int i = 0; i < numNodes; i++) {

            for (int j = 0; j < inputs.length; j++) {

                weights[i][j] += learningRate * weightGradients[i][j];
            }

            biases[i] += learningRate * biasGradients[i];
        }
    }

    // Initializes dense layer state including input size, nodes
    public DenseLayer(int numNodes, int inputSize, double learningRate) {

        this.numNodes = numNodes;
        this.learningRate = learningRate;
        this.weights = new double[numNodes][inputSize];
        this.biases = new double[numNodes];

        // Initialize weights and biases
        initializeWeights();
    }

    // Forward propagation to compute layer outputs
    public double[] forwardPropagation(double[] inputs) {

        this.inputs = inputs;
        this.outputs = new double[numNodes];

        for (int i = 0; i < numNodes; i++) {

            outputs[i] = 0;

            for (int j = 0; j < inputs.length; j++) {

                outputs[i] += weights[i][j] * inputs[j];
            }

            outputs[i] += biases[i];
            outputs[i] = sigmoid(outputs[i]);
        }

        return outputs;
    }

    // Backward propagation to compute parameter gradients
    public void backwardPropagation(double[] expectedOutputs) {

        double[] errors = new double[numNodes];
        double[][] weightGradients = new double[numNodes][inputs.length];
        double[] biasGradients = new double[numNodes];

        for (int i = 0; i < numNodes; i++) {

            errors[i] = expectedOutputs[i] - outputs[i];

            for (int j = 0; j < inputs.length; j++) {

                weightGradients[i][j] = errors[i] * derivativeSigmoid(outputs[i]) * inputs[j];
            }

            biasGradients[i] = errors[i] * derivativeSigmoid(outputs[i]);
        }

        updateWeightsAndBiases(weightGradients, biasGradients);
    }

    // Compute error to propagate further back
    public double[] calculatePreviousLayerError(double[] nextLayerError) {

        double[] previousLayerError = new double[inputs.length];

        for (int i = 0; i < inputs.length; i++) {

            for (int j = 0; j < numNodes; j++) {

                // Propagate error back to the previous layer
                previousLayerError[i] += nextLayerError[j] * weights[j][i] * derivativeSigmoid(outputs[j]);
            }
        }

        return previousLayerError;
    }

}
