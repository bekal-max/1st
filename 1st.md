import numpy as np
import matplotlib.pyplot as plt
X=np.array([[0.6,0.2,0.1],[0.35,0.1,0.1],[0.8,0.4,0.2]])
Y=np.array([[0.75,0.25],[0.3,0.8],[0.3,0.65]])
lr =0.65
# scale units
class NeuralNetwork(object):
    def __init__(self):
        
      #parameters
        self.inputSize = 3 
        self.outputSize = 2
        self.hiddenSize = 3
        
        #Weights and Biases Initialization
        self.W1 = np.random.uniform(size=(self.inputSize,self.hiddenSize)) #W1[3][3] size
        self.W2 = np.random.uniform(size=(self.hiddenSize,self.outputSize))#W2[3][2] size
        self.B1 = np.random.uniform(size=(1,self.hiddenSize))#B1[1][3]for bias weight1
        self.B2 = np.random.uniform(size=(1,self.outputSize))#B2[1][2]for bias weight2
        
    def feedForward(self, X):
        #forward propogation through the network
        self.z = np.dot(X, self.W1)+self.B1
        self.z2 = self.sigmoid(self.z) #activation function
        self.z3 = np.dot(self.z2, self.W2)+self.B2 
        output = self.sigmoid(self.z3)
        return output

     # Sigmoid function   
    def sigmoid(self, s, deriv=False):
        if (deriv == True):
            return s * (1 - s)
        return 1/(1 + np.exp(-s))
     
      # Backward propagation    
    def backward(self, X, Y, output):
        #backward propogate through the network
        self.output_error = Y - output # error in output
        self.output_delta = self.output_error * self.sigmoid(output, deriv=True)#error to hl
        self.z2_error = self.output_delta.dot(self.W2.T) #z2 error: how much our hidden laye
        self.z2_delta = self.z2_error * self.sigmoid(self.z2, deriv=True) #error from hidden 

          # Updating Weights and Biases
        self.W1 += X.T.dot(self.z2_delta)         
        self.W2 += self.z2.T.dot(self.output_delta)
        self.B1 +=np.sum(self.z2_delta,axis=0,keepdims=True) * lr
        self.B2 +=np.sum(self.output_delta,axis=0,keepdims=True) * lr 

      # Plot error graph
    def plot_errors(self, x_axis, y_axis):
      plt.plot(x_axis[:100],y_axis[:100], color='b',linewidth=3)
      plt.xlabel('Iteration')
      plt.ylabel('Error')
      plt.title('Error graph!')
      plt.show()
    def train(self, X, y):
        output = self.feedForward(X)
        self.backward(X, Y, output)
        if(i==0):
         print("------------------------------------------")
         print(" initial_weights:")
         print(""+str(self.W1)+str(self.W2))
         print("initial_loss: i=" +str(i) +" is "+ str(sum_error))# initial loss
        if(i==10):
         print("------------------------------------------")
         print(" 10th_iteration_weights:")
         print(""+str(self.W1)+str(self.W2))
         print("------------------------------------------")
        if(i==1998):
         print("------------------------------------------")
         print("final_weights:")
         print(""+str(self.W1)+str(self.W2))
NN = NeuralNetwork()
errors = list();
iteration = list()
sum_error=np.mean(np.square(Y - NN.feedForward(X)))
errors.append(sum_error)
for i in range(2000): #trains the NN 10000 times
    errors.append(np.mean(np.square(Y - NN.feedForward(X))))
    iteration.append(i)
    if(sum_error<=0.0001):
         print("sum_error and iteration:" +str(i)+str(sum_error))
    if(i>=1999):
          print("the network is not convergent")
          break;    
    NN.train(X, Y) 
NN.plot_errors(iteration,errors)    
print("------------------------------------------")
print("Input: ")
print(X)
print("------------------------------------------")
print("Actual_output: ")
print(Y)
print("------------------------------------------")
print("final_loss: ")
print("i=" +str(i)+" is " +str(np.mean(np.square(Y - NN.feedForward(X)))))
print("------------------------------------------")
print("Predicted Output: ")
print((NN.feedForward(X)))
print("------------------------------------------")
