# LLMed | Large Language Model for Understanding

The purpose of this project is to promote understanding of LLMs and core components. This project implements a very basic character-level language model in Java using a feedforward neural network architecture. The model is trained to predict the next character in sequences of text.

## Overview

The language model comprises a neural network with dense, fully-connected layers. It is trained via backpropagation on a text data file samples in order to learn statistical patterns in sequences. The trained model can then be used to generate or complete texts, by repeatedly sampling predictions for the next character.

## Tuning the LLM

### Main.java

- Epochs: 10, Overfitting can occur if the model is trained for too long. Conversely, underfitting can happen if not trained enough.
- Neural Network Hidden Layers: 100 and 50, More complex texts or larger vocabularies might benefit from larger or more numerous hidden layers, but this also increases the risk of overfitting and the computational cost.
- Embedding Size: 50, Dimensionality of feature vectors representing words.

- Learning Rate:0.01, Learning rate hyperparameter for gradient updates

- Batch Size:8, Number of training samples processed together

## Notes

This project highlights a few important factors about LLMs.

- Preprocessing data is pretty important. The included public domain book by Charles Dickens includes a lot of special characters and symbols that caused issues with creating the vocabulary list of words.

The system consists of the following key classes:

## TextCorpus

Stores textual training data along with vocabulary metadata.

- Load text data
- Provide access to text samples
- Preprocess and normalize text
- Track vocabulary and statistics

## NeuralNetwork

Defines network architecture and handles training.

- Initialize network topology
- Forward and backward passes
- Updating layer weights
- Making next character inferences

## DenseLayer

Defines a fully-connected neural network layer.

- Making forward pass predictions
- Calculating parameter gradients in backpropagation
- Storing weights, biases, outputs
- Applying activation functions

## ModelTrainer

Handles training workflow.

- Retrieve data batches
- Feedforward/backpropagate batches through networks
- Track training metrics like loss

## TextGenerator

Uses trained model to handle text generation tasks.

- Encoding text samples to input vectors
- Feeding vectors through trained network
- Decoding predictions to texts
- Sampling from predictions

Let me know if you have any other questions!

Sample output:

```
==========================================================
LLMed | Large Language Model for Educational Understanding
==========================================================
Load text
   TextCorpus.loadText()
   TextCorpus.preprocessText()
   TextCorpus.updateVocabulary(the dog ran quickly across the park the park was full of families playing the weather was sunny with few clouds the dog continued running around happily many other dogs were also at the park playing joyfully everyone at the park was having a nice relaxing afternoon some children threw balls for the dogs to fetch the dogs brought the balls back dutifully each time it was an idyllic scene in the park that afternoon)
Create layers
Create NeuralNetwork
Create ModelTrainer
Train model
Create TextGenerator
Generate some text
   NeuralNetwork.predict()
   NeuralNetwork.predict()
   NeuralNetwork.predict()
   NeuralNetwork.predict()
   NeuralNetwork.predict()
   NeuralNetwork.predict()
   NeuralNetwork.predict()
   NeuralNetwork.predict()
   NeuralNetwork.predict()
   NeuralNetwork.predict()
The dog the the the the the the the the the the
Complete some text
   NeuralNetwork.predict()
The quick brown  happily scene a an around also some brought to dog relaxing time ran full running dogs afternoon at dutifully weather having of across park fetch many quickly were sunny clouds the that threw for each in joyfully continued with nice it families balls playing other few idyllic children back was everyone
```
