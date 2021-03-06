# Variational Quantum Classifier (using Pennylane Quantum language)

Variational circuits play a role in quantum machine learning akin to that of neural networks in classical machine learning. They typically have a layered structure and a set of tunable parameters that are learned through training with a classical optimization algorithm. Data is input to the circuit by embedding it into the state of a quantum device, and measurement results inform the classical optimizer how to adjust the parameters. The circuit structure, optimization method, and how data is encoded in the circuit varies heavily across algorithms, and there are often problem-speciﬁc considerations that aﬀect how they are trained.


The below algorithm is executed in pennylane quantum language. The input to the program is a image with reduced dimensions to 3 features using PCA and categorical encoding of -1,0 and 1 are used. 
Whereas the output is the string of predicted labels based on the test dimension.  

The Ansatz used for the classification is given below, 

<img src="Ansatz.JPG" height="150" width="500">

## Input
The ﬁle input.in consists of 3 parts:
1. A set of training data points with dimensions (250, 3)
2. The categorical labels for the training points with dimensions (250, )
3. A set of testing data points with dimensions (50, 3)
The data has all been concatenated into a single string. 

## Output

##### The accuracy of 100% was achieved. While training the model the best parameters were saved in a variable and the same were used for testing the model.

<img src="VQC_output_with_graph.JPG" height="300" width="450">

##### Visuzlization of Cost and Accuracy on every epoch is given below,

<img src="graph.JPG" height="300" width="450">

