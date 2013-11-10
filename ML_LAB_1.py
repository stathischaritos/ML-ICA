import scipy
import numpy as np
from numpy import *
from numpy.linalg import *
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from functools import partial
import math
import scipy.io.wavfile

# Signal generators
def sawtooth(x, period=0.2, amp=1.0, phase=0.):
    return (((x / period - phase - 0.5) % 1) - 0.5) * 2 * amp

def sine_wave(x, period=0.2, amp=1.0, phase=0.):
    return np.sin((x / period - phase) * 2 * np.pi) * amp

def square_wave(x, period=0.2, amp=1.0, phase=0.):
    return ((np.floor(2 * x / period - 2 * phase - 1) % 2 == 0).astype(float) - 0.5) * 2 * amp

def triangle_wave(x, period=0.2, amp=1.0, phase=0.):
    return (sawtooth(x, period, 1., phase) * square_wave(x, period, 1., phase) + 0.5) * 2 * amp

def random_nonsingular_matrix(d=2):
    """
    Generates a random nonsingular (invertible) matrix if shape d*d
    """
    epsilon = 0.1
    A = np.random.rand(d, d)
    while abs(np.linalg.det(A)) < epsilon:
        A = np.random.rand(d, d)
    return A

def plot_signals(X):
    """
    Plot the signals contained in the rows of X.
    """
    figure()
    for i in range(X.shape[0]):
        ax = plt.subplot(X.shape[0], 1, i + 1)
        plot(X[i, :])
        ax.set_xticks([])
        ax.set_yticks([])



def make_mixtures(S):
    A = random_nonsingular_matrix(len(S))
    X = np.dot(A,S)
    return X


def plot_histogram(X , bins = 50):
    hist, bins = np.histogram(X ,  bins)
    width =  0.9 *(bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()
    
def plot_histograms(X):
    for x in X:
        plot_histogram(x)

def fi_0(x):
    y = []
    for i in x:
       y += [ 2*math.cosh(i)/(math.cosh(2*i) + 1) ] 
    return y

def ac_0(x):
    y = []
    for i in x:
       y += [ -math.tanh(i)] 
    return y 

def fi_1(x,c = 1):
    y = []
    for i in x:
       y += [ math.exp( -(i**2)/2 ) * math.cosh(i)] 
    return y

def ac_1(x):
    y = []
    for i in x:
       y += [ -i + math.tanh(i) ] 
    return y

def fi_2(x,c = 1):
    y = []
    for i in x:
       y += [ c*math.exp( -(i**4)/4 ) ] 
    return y

def ac_2(x):
    y = []
    for i in x:
       y += [ -i**3 ] 
    return y

def fi_3(x,c = 1):
    y = []
    for i in x:
       y += [ 1/(i**2 + 5)**3 ] 
    return y

def ac_3(x):
    y = []
    for i in x:
       y += [ -(6*i)/(i**2 + 5) ] 
    return y

def plot_function(f,x_min = -3,x_max = 3, signal_length = 100):
    x = linspace(x_min, x_max, signal_length)
    y = f(x)
    plot(x,y)


def whiten(X): 
    Xcov = np.cov(X) 
    d,V = eigh(Xcov) 
    D = diag(1./sqrt(d+0)) 
    W = dot(dot(V,D),V.T) 
    Xwhiten = dot(W,X) 
    return Xwhiten

def scatter_plot_mat(M,C,V):
    figure()
    k = 1;
    for i in range(0,1):
        for j in range(i+1,2):
            subplot(10,3,k)
            scatter(M[i],M[j],s=0.1)
            subplot(10,3,k+1)
            scatter(C[i],C[j],s=0.1)
            subplot(10,3,k+2)
            scatter(V[i],V[j],s=0.1)
            k += 3
 
a0 = lambda x : -tanh(x)
a1 = lambda x : -x + tanh(x)
a2 = lambda x : -power(x,3)
a3 = lambda x : -(6*x)/(power(x,2) + 5)
a4 = lambda x : -2*x

def ICA(X, activation_function, learning_rate = 1.0 , threshold = 0.5 , print_interval = 2000):
    
    #Algorithm Variables
    N = X.shape[1]                           # Size of Observation Set.
    I = X.shape[0]                           # Size of each observation vector x.
    W = random_nonsingular_matrix(I)         # Starting Weight matrix.
    average_dW = zeros((I,I))                # Natural Gradient average
    
    #Helper Variables
    iterations = 0;
    threshold_check = 100;
    
    #Normalise the Input
    #maxrow=amax(X)
    #X=X/maxrow    
 
    while threshold_check > threshold:
        
        # While the average dW is more than a set threshold do:
        ################################################################################################
        a = dot(W,X)                              # Put x through a linear mapping
        z = activation_function(a)                # Put a through a nonlinear map:
        xp = dot(W.T,a)                           # Put a back through W
        average_dW = (N*W + dot(z,xp.T)) / N      # Calculate the average gradient for the whole dataset
        W += learning_rate*average_dW             # Adjust the weight table W
        ################################################################################################
        
        # Recalculate threshold and print at a set interval.
        iterations +=1
        threshold_check = sum(np.absolute(average_dW))
        if iterations%print_interval == 0 : print iterations, threshold_check
            
    return W

def onlineICA(X, activation_function, learning_rate = 1.0 , iterations = 1):
    # Algorithm Variables
    N = X.shape[1]                           # Size of Observation Set.
    I = X.shape[0]                           # Size of each observation vector x.
    W = random_nonsingular_matrix(I)         # Starting Weight matrix.
    dW = zeros((I,I))                        # Natural Gradient average
    
    # Helper variables
    count = 1.0
    check = 1;
    for m in range(iterations):
        for x in X.T:
            a = dot(W,x)
            z = activation_function(a)
            xp = dot(W.T,a)
            for i in range(W.shape[0]):
                for j in range(W.shape[0]):
                    dW[i][j] = W[i][j] + xp[j]*z[i]
            W = W + learning_rate*dW
    
    return W


def save_wav(data, out_file, rate):
    scaled = np.int16(data / np.max(np.abs(data)) * 32767)
    scipy.io.wavfile.write(out_file, rate, scaled)


def plotICA(X, activation_function, learning_rate = 1.0 , threshold = 0.5 , print_interval = 2000):
    
    #Algorithm Variables
    N = X.shape[1]                           # Size of Observation Set.
    I = X.shape[0]                           # Size of each observation vector x.
    W = random_nonsingular_matrix(I)         # Starting Weight matrix.
    average_dW = zeros((I,I))                # Natural Gradient average
    
    #Helper Variables
    iterations = 0;
    threshold_check = 100;
    
    fig=plt.figure()
    x=list()
    y=list()
    plt.ion()
    plt.show()

    while threshold_check > threshold:
        
        # While the average dW is more than a set threshold do:
        ################################################################################################
        a = dot(W,X)                              # Put x through a linear mapping
        z = activation_function(a)                # Put a through a nonlinear map:
        xp = dot(W.T,a)                           # Put a back through W
        average_dW = (N*W + dot(z,xp.T)) / N      # Calculate the average gradient for the whole dataset
        W += learning_rate*average_dW             # Adjust the weight table W
        ################################################################################################
        
        # Recalculate threshold and print at a set interval.
        iterations +=1
        threshold_check = sum(np.absolute(average_dW))
        if iterations%print_interval == 0 : 
            x.append(iterations)
            y.append(threshold_check)
            plt.scatter(iterations,threshold_check)
            plt.draw()
            
    return W