# VQC-with-custom-Ansatz

The input of the program are the 3 dimension of the image and the labels they represent i.e. -1,0 and 1.
Whereas the output is the string of predicted labels based on the test dimension.  

## Input
The ﬁle input.in consists of 3 parts:
1. A set of training data points with dimensions (250, 3)
2. The categorical labels for the training points with dimensions (250, )
3. A set of testing data points with dimensions (50, 3)
The data has all been concatenated into a single string. 

## Output
The output of the program is a string similar to one in the ﬁle answer.ans containing the predicted labels for the testing data separated by commas.
The accuracy of 98% was achieved by using this model and custom built ansatz function for classifying the data.

![OUTPUT](VQC_output.PNG?raw=true "Title")
