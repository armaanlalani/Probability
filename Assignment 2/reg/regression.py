import numpy as np
import matplotlib.pyplot as plt
from util import *

def priorDistribution(beta):
    """
    Plot the contours of the prior distribution p(a)
    
    Inputs:
    ------
    beta: hyperparameter in the proir distribution
    
    Outputs: None
    -----
    """
    A0, A1 = np.meshgrid(np.linspace(-1,1,200), np.linspace(-1,1,200))
    a0 = A0[0].reshape(200,1)
    contour = []
    for i in range(200):
        data = np.concatenate((a0,A1[i].reshape(200,1)),1)
        contour.append(density_Gaussian([0,0],[[beta,0],[0,beta]],data))
    plt.contour(A0,A1,contour,colors='r')
    plt.plot([-0.1],[-0.5], marker='.', markersize=20, color='black')
    plt.xlabel('a0')
    plt.ylabel('a1')
    plt.title('Prior Distribution')
    plt.show()    
    return 
    
def posteriorDistribution(x,z,beta,sigma2):
    """
    Plot the contours of the posterior distribution p(a|x,z)
    
    Inputs:
    ------
    x: inputs from training set
    z: targets from traninng set
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    
    Outputs: 
    -----
    mu: mean of the posterior distribution p(a|x,z)
    Cov: covariance of the posterior distribution p(a|x,z)
    """
    A = np.append(np.ones((len(x),1)),x,axis=1)
    cov_A_inv = np.array([[1/beta,0], [0,1/beta]])
    cov_w_inv = 1/sigma2
    mu = np.squeeze(np.matmul(np.linalg.inv(cov_A_inv + cov_w_inv*np.matmul(A.T,A)), cov_w_inv*np.matmul(A.T,z)))
    Cov = np.linalg.inv(cov_A_inv + cov_w_inv*np.matmul(A.T,A))
    
    A0, A1 = np.meshgrid(np.linspace(-1,1,200), np.linspace(-1,1,200))
    a0 = A0[0].reshape(200,1)
    contour = []
    for i in range(200):
        data = np.concatenate((a0,A1[i].reshape(200,1)),1)
        contour.append(density_Gaussian(mu.T,Cov,data))
    plt.contour(A0,A1,contour,colors='r')
    plt.plot([-0.1],[-0.5], marker='.', markersize=20, color='black')
    plt.xlabel('a0')
    plt.ylabel('a1')
    plt.title('Posterior Distribution with %d Samples' %(len(x)))
    plt.show()
    return (mu,Cov)

def predictionDistribution(x,beta,sigma2,mu,Cov,x_train,z_train):
    """
    Make predictions for the inputs in x, and plot the predicted results 
    
    Inputs:
    ------
    x: new inputs
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    mu: output of posteriorDistribution()
    Cov: output of posteriorDistribution()
    x_train,z_train: training samples, used for scatter plot
    
    Outputs: None
    -----
    """
    A = np.append(np.ones((len(x),1)),np.expand_dims(x,1),axis=1)
    cov_A_inv = np.array([[1/beta,0], [0,1/beta]])
    cov_w = sigma2
    mu_z = np.matmul(A,mu)
    cov_z = cov_w + np.matmul(A, np.matmul(Cov,A.T))

    plt.scatter(x_train, z_train, color='red')
    plt.errorbar(x, mu_z, yerr=np.sqrt(np.diag(cov_z)), marker='x')
    plt.xlim([-4,4])
    plt.ylim([-4,4])
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title('Prediction using %d Samples' %(len(x_train)))
    plt.show()
    return 

if __name__ == '__main__':
    
    # training data
    x_train, z_train = get_data_in_file('training.txt')
    # new inputs for prediction 
    x_test = [x for x in np.arange(-4,4.01,0.2)]
    
    # known parameters 
    sigma2 = 0.1
    beta = 1
    
    # number of training samples used to compute posterior
    NS = [1,5,100]
    
    for ns in NS:
        # used samples
        x = x_train[0:ns]
        z = z_train[0:ns]
    
        # prior distribution p(a)
        priorDistribution(beta)
    
        # posterior distribution p(a|x,z)
        mu, Cov = posteriorDistribution(x,z,beta,sigma2)
    
        # distribution of the prediction
        predictionDistribution(x_test,beta,sigma2,mu,Cov,x,z)