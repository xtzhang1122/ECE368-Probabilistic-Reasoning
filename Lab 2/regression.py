#from statistics import covariance
#from importlib_metadata import distribution
import numpy as np
import matplotlib.pyplot as plt
import util

def priorDistribution(beta):
    """
    Plot the contours of the prior distribution p(a)
    
    Inputs:
    ------
    beta: hyperparameter in the proir distribution
    
    Outputs: None
    -----
    """
    ### TODO: Write your code here

    mean_vector = np.zeros(2)
    covariance_matrix = np.zeros((2, 2), int)
    np.fill_diagonal(covariance_matrix, beta)
    
    delta = 1 / 50
    x = np.arange(-1.0, 1.0, delta)
    y = np.arange(-1.0, 1.0, delta)
    X, Y = np.meshgrid(x, y)
    
    y = np.array([y]).T
    
    Z = []
    for val in x:
        x_entry = np.ones((100, 1)) * val
        sample = np.hstack((x_entry, y))
        Z.append(util.density_Gaussian(mean_vector, covariance_matrix, sample))

    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z)
    # ax.clabel(CS, inline=1, fontsize=10)
    ax.set_title('Prior Distribution')
  #  plt.show()
    plt.xlabel("x")
    plt.ylabel("z")
    plt.savefig("prior.pdf")
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
    # x is a column vector
    one = np.ones((x.shape[0], 1), int)
    x = np.hstack((one, x))

    convariance_inverse = np.array([[1 / beta, 0], [0, 1 / beta]])
    Cov = np.linalg.inv(np.matmul(x.T, x) * 1 / sigma2 + convariance_inverse) # (2, 2)
    mu = 1 / sigma2 * np.matmul(Cov, np.matmul(x.T, z)) # (2, 1)
    
    delta = 1 / 50
    x_axis = np.arange(-1.0, 1.0, delta)
    y_axis = np.arange(-1.0, 1.0, delta)
    X, Y = np.meshgrid(x_axis, y_axis)
    
    x_temp = X.flatten().reshape(-1, 1)
    y_temp = Y.flatten().reshape(-1, 1)
    input = np.concatenate((x_temp, y_temp), axis=1)

    Z = util.density_Gaussian(mu.T[0], Cov, input).reshape(len(x_axis), len(y_axis))
    
    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z)
    ax.scatter(-0.1, -0.5, color = 'red')
    ax.set_title('Posterior Distribution for 5 training example(s)')
   # plt.show()
    plt.xlabel("x")
    plt.ylabel("z")
    plt.savefig("posterior5.pdf")

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
    # x is a row vector
    one = np.ones((1, len(x)), int)
    x_design = np.vstack((one, x))
    
    mu = np.matmul(x_design.T, mu)
    Cov = np.matmul(x_design.T, np.matmul(Cov, x_design)) + sigma2
    variance = np.sqrt(np.diag(Cov))
    
    z = np.dot(-0.5, x) - 0.1

    fig, ax = plt.subplots()
    ax.scatter(x_train, z_train, color="red")
    ax.scatter(x, z, color="green")
    ax.errorbar(x, z, yerr=variance.flatten())
    ax.set_title('Predicted Distribution based on 5 training example(s)')
    plt.xlabel("x")
    plt.ylabel("z")
    plt.savefig("predict5.pdf")
    plt.show()
    
    return 
    

if __name__ == '__main__':
    
    # training data
    x_train, z_train = util.get_data_in_file('training.txt')
    # new inputs for prediction 
    x_test = [x for x in np.arange(-4,4.01,0.2)]
    
    # known parameters 
    sigma2 = 0.1
    beta = 1
    
    # number of training samples used to compute posterior
    ns  = 5
    
    # used samples
    x = x_train[0:ns]
    z = z_train[0:ns]
    
    # prior distribution p(a)
    priorDistribution(beta)
    
    # posterior distribution p(a|x,z)
    mu, Cov = posteriorDistribution(x,z,beta,sigma2)
    
    # distribution of the prediction
    predictionDistribution(x_test,beta,sigma2,mu,Cov,x,z)