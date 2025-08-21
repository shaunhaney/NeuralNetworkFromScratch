import random
import math
import matplotlib.pyplot as plt
import numpy as np
from HistoryList import HistoryList

class SimpleNN:

    # input
    @property
    def input(self):
        return self._input

    @input.setter
    def input(self, value):
        self._input = value

    @input.deleter
    def input(self):
        del self._input

    #hiddenLayer 
    """
    hiddenLayer is actually a matrix with the same number of rows as there are 
    samples (like input) the number of columns matches the number of hidden nodes.
    """
    @property
    def hiddenLayer(self):
        return self._hiddenLayer

    @hiddenLayer.setter
    def hiddenLayer(self, value):
        self._hiddenLayer = value

    @hiddenLayer.deleter
    def hiddenLayer(self):
        del self._hiddenLayer
    """ 
    expectedOutput:
    What does the expected output look like?
    Rows:  each row corresponds to a combination of input parameters.
    Let's say we're doing a very simple XOR Neural Network and we 
    have 2 inputs (2 1-bit numbers between 0 and 1)  and we have 1 bit 
    of output.  We would have 4 rows (00, 01, 10, 11) and 1 column. 
    Let's say we had 8 inputs and expected 4 outputs.  We would have 64 rows
    (0-63) and 4 columns 
    """
    @property
    def expectedOutput(self):
        return self._expectedOutput

    @expectedOutput.setter
    def expectedOutput(self, value):
        self._expectedOutput = value

    @expectedOutput.deleter
    def expectedOutput(self):
        del self._expectedOutput

    # actual output
    @property
    def actualOutput(self):
        return self._actualOutput

    @actualOutput.setter
    def actualOutput(self, value):
        self._actualOutput = value

    @actualOutput.deleter
    def actualOutput(self):
        del self._actualOutput

    # w1
    @property
    def w1(self):
        return self._w1

    @w1.setter
    def w1(self, value):
        self._w1 = value

    @w1.deleter
    def w1(self):
        del self._w1

    # b1
    @property
    def b1(self):
        return self._b1

    @b1.setter
    def b1(self, value):
        self._b1 = value

    @b1.deleter
    def b1(self):
        del self._b1

    # w2
    @property
    def w2(self):
        return self._w2

    @w2.setter
    def w2(self, value):
        self._w2 = value

    @w2.deleter
    def w2(self):
        del self._w2

    # b2
    @property
    def b2(self):
        return self._b2

    @b2.setter
    def b2(self, value):
        self._b2 = value

    @b2.deleter
    def b2(self):
        del self._b2

    # epochs
    @property
    def epochs(self):
        return self._epochs

    @epochs.setter
    def epochs(self, value):
        self._epochs = value

    @epochs.deleter
    def epochs(self):
        del self._epochs

    def createReporter(self, reporter, matrix):
        reporter.register(matrix)
        self._reporters.append(reporter)

    def report(self):
        for reporter in self._reporters: 
            reporter.record()


    def __init__(self, input, expectedOutput):
        self._reporters=[]
        self.actualOutput=[]

        try:
            self._inputRows=len(input)
            self._inputColumns=len(input[0])
        except IndexError:
            raise TypeError("Input needs to be a list")

        try: 
            self._outputRows=len(expectedOutput)
            self._outputColumns=len(expectedOutput[0])
        except: 
            raise TypeError("Input needs to be a list")
        
            
        #Input and expected output should be lists.  Get their dimensions.
        self._input=input
        self._expectedOutput=expectedOutput
        
        # Set the hidden layer size (input columns + output columns)
        self._hiddenLayerSize=math.ceil((self._inputColumns+self._outputColumns)/2) 
        self._hiddenLayerSize=max(2,self._hiddenLayerSize)
        
        # Create the initial set of weights
        # The matrix is going to be the number inputs (columns) by the number of
        # hidden nodes 
        self.w1=[]
        for ndx in range(self._inputColumns):
            self.w1.append([])
            for ndx2 in range(self._hiddenLayerSize):
                self.w1[ndx].append(random.random())

        # Create the initial set of weights
        # The matrix is going to be the number of hidden nodes (rows) by the 
        # number of outputs (columns) 
        self.w2=[]
        for ndx in range(self._hiddenLayerSize):
            self.w2.append([])
            for ndx2 in range(self._outputColumns):
                self.w2[ndx].append(random.random()) 
        
        #Create the vector of biases for the hidden nodes
        self.b1 = []
        for ndx in range(self._hiddenLayerSize):
            self.b1.append(random.random())

        #Create the vector of biases for the hidden nodes
        self.b2 = []
        for ndx in range(self._outputColumns):
            self.b2.append(random.random())

        
    
    def dotProduct(self, mX,mY):
        xNumRows=len(mX)
        xNumCols=len(mX[0])

        yNumRows=len(mY)
        yNumCols=len(mY[0])

        if (xNumCols!=yNumRows):
            return None
        
        mA=[]
        for i in range(0,xNumRows):
            mA.append([])

            for j in range(0,yNumCols):
                mA[i].append(0)
                for k in range(0,xNumCols):
                    mA[i][j]=mA[i][j]+mX[i][k]*mY[k][j]
        return mA
    
    def transpose(self,a):
        valid=(a is not None) and (a[0] is not None) and (a[0][0] is not None)
        if (not valid):
            return 
        o=[]
        # Go along a single column and compose a row in the output 
        # array
        for d in range(len(a[0])):
            o.append([])
            for r in range(len(a)):
                o[d].append(a[r][d])
        return o

    # keep our weight values between 0 and 1 noninclusive 
    def sigmoid(self,x):
        return 1/(1+math.exp(-x))
    
    def forwardPropagation(self):

        # Perform dot product between inputs and weights
        z1=self.dotProduct(self.input,self.w1)
        self._hiddenLayer=[] 
        for ndx in range(len(z1)):
            self._hiddenLayer.append([])
            for ndx2 in range(len(z1[0])):
                self._hiddenLayer[ndx].append(self.sigmoid(z1[ndx][ndx2]+self.b1[ndx2]))
        
        # Perform dot product between hidden layer and outputs
        z2=self.dotProduct(self._hiddenLayer,self.w2)
        self.actualOutput.clear()
        for ndx in range(len(z2)):
            self.actualOutput.append([])
            for ndx2 in range(len(z2[0])):
                self.actualOutput[ndx].append(self.sigmoid(z2[ndx][ndx2]+self.b2[ndx2]))

    """
    loss: 
    How far off the actual output is from the expected output.  Negative means 
    values are too high, and positive means values are too low.  Zero, which will
    never happen means expected values are being predicted by forward propagation
    """
    def loss(self):
        total_loss = 0.0
        self._error=[]
        for i in range(self._outputRows):
            self._error.append([])
            for j in range(self._outputColumns):
                self._error[i].append(self.expectedOutput[i][j] - self.actualOutput[i][j])
                total_loss += self._error[i][j] ** 2
        return total_loss / self._outputRows

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def backpropagation(self, learningRate=0.1):
        loss=self.loss()

        output_delta = []
        for i in range(self._outputRows):
            row = []
            for j in range(self._outputColumns):
                error = self._error[i][j]
                derivative = self.sigmoid_derivative(self.actualOutput[i][j])
                row.append(error * derivative)
            output_delta.append(row)

        # Get output delta dot transform of w2
        hidden_error_signal = self.dotProduct(output_delta,self.transpose(self.w2))

        # sigmoid derivative of 2D hidden layer 
        sigmoid_gradients = []
        for i in range(len(self._hiddenLayer)):
            row = []
            for j in range(len(self._hiddenLayer[0])):
                row.append(self.sigmoid_derivative(self._hiddenLayer[i][j]))
            sigmoid_gradients.append(row)

        hidden_delta=[]
        for i in range(len(hidden_error_signal)):
            row = []
            for j in range(len(hidden_error_signal[0])):
                row.append(hidden_error_signal[i][j] * sigmoid_gradients[i][j])
            hidden_delta.append(row)

        # Update w1 and b1
        for i in range(len(self.w1)):
            for j in range(len(self.w1[0])):
                grad = 0
                for k in range(self._inputRows):
                    grad += self._input[k][i] * hidden_delta[k][j]
                self.w1[i][j] += learningRate * grad

        for i in range(len(self.b1)):
            grad = sum([hidden_delta[k][i] for k in range(self._inputRows)])
            self.b1[i] += learningRate * grad

        
        # Update w2 and b2
        for i in range(len(self.w2)):
            for j in range(len(self.w2[0])):
                grad = 0
                """ Here, the number of samples in hidden rows should be the same as input rows"""
                for k in range(self._inputRows):
                    grad += self._hiddenLayer[k][i] * output_delta[k][j]
                self.w2[i][j] += learningRate * grad

        for i in range(len(self.b2)):
            grad = sum([output_delta[k][i] for k in range(self._outputRows)])
            self.b2[i] += learningRate * grad

    def train(self,learningRate=0.1,maxEpochs=None,targetLoss=0.00001):
        loss=1
        ctr=0
        
        self.report()

        keepGoing = lambda loss,ctr: (ctr<maxEpochs if maxEpochs is not None else True) and loss>targetLoss
        while (keepGoing(loss,ctr)):
            self.forwardPropagation()
            self.backpropagation(learningRate)
            loss=self.loss()
            ctr+=1
            if (ctr%10000==0):
                self.report()
        self.report()
        self.epochs=ctr 

def plot_history(histories, names):
    columnsInARow=int(len(histories)/2+0.5)
    fig, axes = plt.subplots(2, columnsInARow, figsize=(10, 4))

    historyCtr=0
    
    for history in histories:

        rowNdx=int(historyCtr/columnsInARow)
        colNdx=historyCtr%columnsInARow

        p=axes[rowNdx][colNdx]
        name=names[historyCtr]

        rows=len(history[0])
        if (isinstance(history[0][0],list)):
            cols = len(history[0][0])
            for r in range(rows):
                for c in range(cols):
                    p.plot([m[r][c] for m in history], label=f"{name}[{r}][{c}]")
        else:
            for r in range(rows):
                p.plot([m[r] for m in history], label=f"{name}[{r}]")
        p.set_title(f"{names[historyCtr]} parameter trajectories")
        p.legend()
        historyCtr+=1

    fig.supxlabel("Checkpoint index")
    fig.supylabel("value")
    plt.tight_layout()
    plt.show()                      

# Rows are samples in input and output, thus even though output, in this case, 
# has only one value per sample, each sample is still a row. 
n=SimpleNN([[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]],[[0.0],[1.0],[1.0],[0.0]])

w1History=HistoryList()
b1History=HistoryList()
w2History=HistoryList()
b2History=HistoryList()
actualOutputHistory=HistoryList()

n.createReporter(w1History,n.w1)
n.createReporter(b1History,n.b1)
n.createReporter(w2History,n.w2)
n.createReporter(b2History,n.b2)
n.createReporter(actualOutputHistory,n.actualOutput)

n.train()

print ("Final Result:")
print (f"Actual Output: {n.actualOutput}")
print (f"       Epochs: {n.epochs}") 
print (f"           w1: {n.w1}")
print (f"           w2: {n.w2}")
print (f"           b1: {n.b1}")
print (f"           b2: {n.b2}")

plot_history([x.records for x in [w1History,b1History,w2History,b2History,actualOutputHistory]],["w1","b1","w2","b2","actualOutput"])