from collections import Counter
import sys
import pennylane as qml
import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer, AdamOptimizer


def classify_data(X_train, Y_train, X_test):
    """Develop and train variational quantum classifier.

    Args:
        X_train (np.ndarray): An array of floats of size (250, 3) to be used as training data.
        Y_train (np.ndarray): An array of size (250,) which are the categorical labels
            associated to the training data. The categories are labeled by -1, 0, and 1.
        X_test (np.ndarray): An array of floats of (50, 3) to serve as testing data.

    Returns:
        str: The predicted categories of X_test, converted from a list of ints to a
            comma-separated string.
    """

    # Use this array to make a prediction for the labels of the data in X_test
    predictions = []

    NUM_WIRES=3
    NUM_LAYERS=2

    dev = qml.device("default.qubit",wires= NUM_WIRES)

    @qml.qnode(dev)
    def circuit(params,x):
        xEmbeded=[i*np.pi for i in x]
        for i in range(NUM_WIRES):
            qml.RX(xEmbeded[i],wires=i)
            qml.Rot(*params[0,i],wires=i)
        
        qml.CZ(wires=[1, 0])
        qml.CZ(wires=[1, 2])
        qml.CZ(wires=[0, 2])

        for i in range(NUM_WIRES):
            qml.Rot(*params[1,i],wires=i)

        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2))

    def prediction(weigths,x_train):
        predictions = [circuit(weigths, f) for f in x_train]
        for i,p in enumerate(predictions):
            maxi=p[0]
            indexMax=0
            for k in range(3):
                if p[k] > maxi:
                    maxi=p[k]
                    indexMax=k
            predictions[i]=indexMax-1
        
        return predictions

    
    def cost(weights, x_train, labels):
        predictions = [circuit(weights, f) for f in x_train]
        
        loss=0
        for i in range(len(predictions)):
            min=predictions[i][0]
            max=predictions[i][0]
            for k in range(3):
                if predictions[i][k]>max:
                    max=predictions[i][k]
                if predictions[i][k]<min:
                    min=predictions[i][k]
                    

            x=(predictions[i][labels[i]+1]-min)/(max-min)

            loss+= (1-x)**2       # [0.4,0.3,0.3]   

        return loss/len(predictions)

    def accuracy(weights,x_train,labels):
        predictions=prediction(weights,x_train)
        loss=0
        for i in range(len(predictions)):
            if predictions[i]==labels[i]:
                loss+=1
        loss=loss/len(predictions)
        return loss

    params = (0.01 * np.random.randn(2, NUM_WIRES, 3))
    bestparams=(0.01 * np.random.randn(2, NUM_WIRES, 3))
    bestcost=1
    opt = AdamOptimizer(0.425)
    batch_size = 10
    Hist=[]
    for it in range(30):
        
        # Update the weights by one optimizer step
        batch_index = np.random.randint(0, 250, (batch_size,))
        X_train_batch = X_train[batch_index]
        Y_train_batch = Y_train[batch_index]
        params = opt.step(lambda v: cost(v, X_train_batch, Y_train_batch), params)

        # Compute predictions on train and validation set
        #predictions_train = prediction(params,X_train)

        cosT = cost(params, X_train,Y_train)
        # Compute accuracy on train and validation set
        acc = accuracy(params, X_train,Y_train) 
        Hist.append(acc)
        if cosT < bestcost:
            bestcost = cosT
            bestparams = params
        
        
        print(
            "Iter: {:5d} | Cost: {:0.7f} | Accuracy: {:0.2f}% ".format(
            it + 1, cosT, acc*100
        ))

    predictions = prediction(bestparams,X_test)
    results =[1,0,-1,0,-1,1,-1,-1,0,-1,1,-1,0,1,0,-1,-1,0,0,1,1,0,-1,0,0,-1,0,-1,0,0,1,1,-1,-1,-1,0,-1,0,1,0,-1,1,1,0,-1,-1,-1,-1,0,0]

    accResult = accuracy(bestparams,X_test,results)
    print()
    print("FINAL ACCURACY: {:0.2f}%".format(accResult*100))
    circuit(bestparams, X_train[0])
    print()
    print(circuit.draw())
    return array_to_concatenated_string(predictions), accuracy


def array_to_concatenated_string(array):
    """
    Turns an array of integers into a concatenated string of integers
    separated by commas. (Inverse of concatenated_string_to_array).
    """
    return ",".join(str(x) for x in array)


def concatenated_string_to_array(string):
    """
    Turns a concatenated string of integers separated by commas into
    an array of integers. (Inverse of array_to_concatenated_string).
    """
    return np.array([int(x) for x in string.split(",")])


def parse_input(giant_string):
    """
    Parse the input data into 3 arrays: the training data, training labels,
    and testing data.

    Dimensions of the input data are:
      - X_train: (250, 3)
      - Y_train: (250,)
      - X_test:  (50, 3)
    """
    X_train_part, Y_train_part, X_test_part = giant_string.split("XXX")

    X_train_row_strings = X_train_part.split("S")
    X_train_rows = [[float(x) for x in row.split(",")] for row in X_train_row_strings]
    X_train = np.array(X_train_rows)

    Y_train = concatenated_string_to_array(Y_train_part)

    X_test_row_strings = X_test_part.split("S")
    X_test_rows = [[float(x) for x in row.split(",")] for row in X_test_row_strings]
    X_test = np.array(X_test_rows)

    return X_train, Y_train, X_test


if __name__ == "__main__":

    X_train, Y_train, X_test = parse_input(sys.stdin.read())
    output_string = classify_data(X_train, Y_train, X_test)
    print(f"{output_string}")
