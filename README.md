# LLMed | Large Language Model for Understanding

The purpose of this project is to promote understanding of LLMs and core components and is not intended for production use.

## Overview

This project implements a very basic character-level language model in Java from scratch and applying a feedforward neural network architecture. The model is trained to predict the next character in sequences of text. The sample text provided is "A Tale of Two Cities" by Charles Dickens, a public domain publication with a 10,000 word vocabualry. Run Main.java and its 5 supporting classes in any Java VM to experiment with LLM tuning.

This language model comprises a basic neural network with dense, fully-connected layers. It is trained via backpropagation on a text data file samples in order to learn statistical patterns in sequences. The trained model can then be used to generate or complete texts, by repeatedly sampling predictions for the next character.

## Fine Tuning the LLM

### Main.java

- **Epochs**: 10 (5-20), The number of times the entire dataset is passed forward and backward through the neural network. By repeatedly exposing the neural network to the entire dataset, it learns to recognize patterns and adjust its weights accordingly. Overfitting can occur if the model is trained for too long. Conversely, underfitting can happen if not trained enough.
- **Neural Network Hidden Layers**: 100 and 50 (50-500 and 10-100), Number of nodes (neurons) in the hidden layers of the network. The neural network has two hidden layers: the first with 100 nodes and the second with 50 nodes. More complex texts or larger vocabularies might benefit from larger or more numerous hidden layers, but this also increases the risk of overfitting and the computational cost.
- **Embedding Size**: 50 (25-250), Dimensionality of feature vectors representing words. A larger embedding size allows for more detailed representations of words, capturing richer semantic and syntactic information.

### NeuralNetwork.java

- **Learning Rate**:0.01 (0.001-0.05), Controls how much the model's weights are adjusted during training. It determines the size of the steps taken in the weight space during the optimization process. A higher learning rate means larger steps, while a lower rate results in smaller steps. It's one of the most important hyperparameters to tune in a neural network.

### ModelTrainer.java

- **Batch Size**:8 (1-32), Training with batches impacts computational efficiency. Larger batches provide more accurate estimates of the gradient, but are computationally more expensive. Smaller batches are computationally less expensive per update but may require more iterations overall.

### TextGenerator.java

- **Temperature**: 0.7 (0.5-1.5), A lower temperature makes the distribution sharper (more confident), while a higher temperature makes it flatter (more diverse but less accurate).

The system consists of the following key classes:

## TextCorpus Class

Stores textual training data along with vocabulary metadata.

- Load text data
- Provide access to text samples
- Preprocess and normalize text
- Track vocabulary and statistics

## NeuralNetwork Class

Defines network architecture and handles training.

- Initialize network topology
- Forward and backward passes
- Updating layer weights
- Making next character inferences

## DenseLayer Class

Defines a fully-connected neural network layer.

- Making forward pass predictions
- Calculating parameter gradients in backpropagation
- Storing weights, biases, outputs
- Applying activation functions

## ModelTrainer Class

Handles training workflow.

- Retrieve data batches
- Feedforward/backpropagate batches through networks
- Track training metrics like loss

## TextGenerator Class

Uses trained model to handle text generation tasks.

- Encoding text samples to input vectors
- Feeding vectors through trained network
- Decoding predictions to texts
- Sampling from predictions

Let me know if you have any other questions!

Sample output:

````
==========================================================
LLMed | Large Language Model for Educational Understanding
==========================================================
Load corpus text
Create layers, vocabulary size is: 10029
Create NeuralNetwork
Create TextGenerator
Create ModelTrainer
Train model
Epoch 1, Batch 1, Average Error: 0.9999002891614318
Epoch 1, Batch 2, Average Error: 0.9999002891614318
Epoch 1, Batch 3, Average Error: 0.999900288044374
Epoch 1, Batch 4, Average Error: 0.9860869573718222
Epoch 1, Batch 5, Average Error: 0.9367324849997511
Epoch 1, Batch 6, Average Error: 0.9488376772632638
Epoch 1, Batch 7, Average Error: 0.9591138465226053
Epoch 1, Batch 8, Average Error: 0.9686865036312469
End of Epoch 1, Average Error: 0.9748947920194908
Epoch 2, Batch 1, Average Error: 0.9724575981132799
Epoch 2, Batch 2, Average Error: 0.9664251010665131
Epoch 2, Batch 3, Average Error: 0.9749355129932356
Epoch 2, Batch 4, Average Error: 0.9713063324024448
Epoch 2, Batch 5, Average Error: 0.97739366117978
Epoch 2, Batch 6, Average Error: 0.9700289043255153
Epoch 2, Batch 7, Average Error: 0.9596907590747116
Epoch 2, Batch 8, Average Error: 0.94452024586008
End of Epoch 2, Average Error: 0.967094764376945
Epoch 3, Batch 1, Average Error: 0.9288646353032536
Epoch 3, Batch 2, Average Error: 0.949984224870208
Epoch 3, Batch 3, Average Error: 0.9209969824072486
Epoch 3, Batch 4, Average Error: 0.8960226024862198
Epoch 3, Batch 5, Average Error: 0.8331732464311224
Epoch 3, Batch 6, Average Error: 0.7518584486170439
Epoch 3, Batch 7, Average Error: 0.6576114541662151
Epoch 3, Batch 8, Average Error: 0.5592711951743257
End of Epoch 3, Average Error: 0.8122228486819545
Epoch 4, Batch 1, Average Error: 0.46598585024018463
Epoch 4, Batch 2, Average Error: 0.48595694062575173
Epoch 4, Batch 3, Average Error: 0.5409241055595707
Epoch 4, Batch 4, Average Error: 0.683113484253913
Epoch 4, Batch 5, Average Error: 0.5973475182355109
Epoch 4, Batch 6, Average Error: 0.4680005717310653
Epoch 4, Batch 7, Average Error: 0.3617742871477053
Epoch 4, Batch 8, Average Error: 0.28308073664622635
End of Epoch 4, Average Error: 0.48577293680499095
Epoch 5, Batch 1, Average Error: 0.2271457019522432
Epoch 5, Batch 2, Average Error: 0.1873243756361241
Epoch 5, Batch 3, Average Error: 0.15829777443852888
Epoch 5, Batch 4, Average Error: 0.13651827342277914
Epoch 5, Batch 5, Average Error: 0.11983454333247001
Epoch 5, Batch 6, Average Error: 0.1337780384434886
Epoch 5, Batch 7, Average Error: 0.13733082241333983
Epoch 5, Batch 8, Average Error: 0.11954471079093239
End of Epoch 5, Average Error: 0.15247178005373827
Epoch 6, Batch 1, Average Error: 0.10565710628349181
Epoch 6, Batch 2, Average Error: 0.09456429960186616
Epoch 6, Batch 3, Average Error: 0.08552393788420301
Epoch 6, Batch 4, Average Error: 0.07801832873368793
Epoch 6, Batch 5, Average Error: 0.07169900543098577
Epoch 6, Batch 6, Average Error: 0.06630910315533192
Epoch 6, Batch 7, Average Error: 0.061657078360254226
Epoch 6, Batch 8, Average Error: 0.05760526736913019
End of Epoch 6, Average Error: 0.07762926585236887
Epoch 7, Batch 1, Average Error: 0.05404567821436524
Epoch 7, Batch 2, Average Error: 0.05089332719176844
Epoch 7, Batch 3, Average Error: 0.0481234756674845
Epoch 7, Batch 4, Average Error: 0.06199942825950882
Epoch 7, Batch 5, Average Error: 0.06535401151183386
Epoch 7, Batch 6, Average Error: 0.0606538873748359
Epoch 7, Batch 7, Average Error: 0.05648788816849921
Epoch 7, Batch 8, Average Error: 0.05283851436576063
End of Epoch 7, Average Error: 0.05629952634425708
Epoch 8, Batch 1, Average Error: 0.04963963460948709
Epoch 8, Batch 2, Average Error: 0.04680545119825764
Epoch 8, Batch 3, Average Error: 0.04427832544058079
Epoch 8, Batch 4, Average Error: 0.042013195779629545
Epoch 8, Batch 5, Average Error: 0.039969105959923645
Epoch 8, Batch 6, Average Error: 0.03811763753336765
Epoch 8, Batch 7, Average Error: 0.036427391285572326
Epoch 8, Batch 8, Average Error: 0.03488318799260627
End of Epoch 8, Average Error: 0.041516741224928114
Epoch 9, Batch 1, Average Error: 0.03346509126719783
Epoch 9, Batch 2, Average Error: 0.03215837141235568
Epoch 9, Batch 3, Average Error: 0.03094959374727372
Epoch 9, Batch 4, Average Error: 0.029832595484737164
Epoch 9, Batch 5, Average Error: 0.02896177667182073
Epoch 9, Batch 6, Average Error: 0.03956113896404992
Epoch 9, Batch 7, Average Error: 0.04122716278573183
Epoch 9, Batch 8, Average Error: 0.03911381273609482
End of Epoch 9, Average Error: 0.03440869288365771
Epoch 10, Batch 1, Average Error: 0.03722454080661865
Epoch 10, Batch 2, Average Error: 0.03550451536415816
Epoch 10, Batch 3, Average Error: 0.03395068733438006
Epoch 10, Batch 4, Average Error: 0.032524613302063354
Epoch 10, Batch 5, Average Error: 0.031215662449706038
Epoch 10, Batch 6, Average Error: 0.030009814696994607
Epoch 10, Batch 7, Average Error: 0.028898154089008466
Epoch 10, Batch 8, Average Error: 0.027864178204964393
End of Epoch 10, Average Error: 0.03214902078098671


Complete text: In his expostulation he dropped his cleaner hand...
In his expostulation he dropped his cleaner hand and was in the that drop inform mix sins balanced than spaces wwwgutenbergorgcontact era beholders employee by control it clerkenwell doubt test profoundly ninety needful thundered spent methodically wise flag fathers swearing hounsditch besieged employing teems strolled vegetable lifes plate range paltering delightful thirsty doomed quickest unavenged countersigned insinuation privileges wolf animated infallible plaiting scratch paper grains


Generate some text: This dialogue had been held in so very low a whisper...
This dialogue had been held in so very low a whisper and was mix was in inform sins inform and mix drop balanced and and that balanced the mix inform drop balanced the the drop


Enter text to complete (or quit): ```
````
