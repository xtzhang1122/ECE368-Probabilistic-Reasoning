import numpy as np
import matplotlib.pyplot as plt
import util

def discrimAnalysis(x, y):
    """
    Estimate the parameters in LDA/QDA and visualize the LDA/QDA models
    
    Inputs
    ------
    x: a N-by-2 2D array contains the height/weight data of the N samples
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of five elments: mu_male,mu_female,cov,cov_male,cov_female
    in which mu_male, mu_female are mean vectors (as 1D arrays)
             cov, cov_male, cov_female are covariance matrices (as 2D arrays)
    Besides producing the five outputs, you need also to plot 1 figure for LDA 
    and 1 figure for QDA in this function         
    """

    # mean & covariance calculations
    male = x[np.where(y == 1)]
    female = x[np.where(y == 2)]
    
    mu_male = np.mean(male, 0)
    mu_female = np.mean(female, 0)

    cov = np.zeros((2, 2))
    cov_male = np.zeros((2, 2))
    cov_female = np.zeros((2, 2))

    for i in male:
        cov_male += np.matmul((i.reshape(2,1) - mu_male.reshape(2,1)), np.transpose(i.reshape(2,1) - mu_male.reshape(2,1)))

    for i in female:
        cov_female += np.matmul((i.reshape(2, 1) - mu_female.reshape(2, 1)), np.transpose(i.reshape(2, 1) - mu_female.reshape(2, 1) ))
    
    cov = 1 / len(x) * (cov_male + cov_female)
    cov_male = cov_male / len(male)
    cov_female = cov_female / len(female)

    # linear plot
    plt.scatter(male[:, 0], male[:, 1], c= "blue")
    plt.scatter(female[:, 0], female[:, 1], c= "red")
    x_grid = np.arange(50, 80, 0.15)
    y_grid = np.arange(80, 280, 1)
    X, Y = np.meshgrid(x_grid, y_grid)
    tmp = np.dstack((X, Y))
    tmp = np.reshape(tmp, (len(X)*len(Y), 2))
    maleDensity = util.density_Gaussian(mu_male, cov, tmp).reshape(len(X), X.shape[1])
    femaleDensity = util.density_Gaussian(mu_female, cov, tmp).reshape(len(Y), Y.shape[1])
    plt.contour(X, Y, maleDensity, colors= "blue")
    plt.contour(X, Y, femaleDensity, colors= "red")

    # LDA
    male_LDA = mu_male.T @ np.linalg.inv(cov) @ tmp.T - 0.5 * mu_male.T @ np.linalg.inv(cov) @ mu_male
    female_LDA = mu_female.T @ np.linalg.inv(cov) @ tmp.T - 0.5 * mu_female.T @ np.linalg.inv(cov) @ mu_female
    pred_LDA = male_LDA.reshape(len(X), X.shape[1]) - female_LDA.reshape(len(Y), Y.shape[1])
    plt.contour(X, Y, pred_LDA, 0)
    axes = plt.gca()
    axes.set_xlim([50, 80])
    axes.set_ylim([80, 280])
    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.title('Height vs. Weight for Linear Data Analysis')
    plt.savefig("lda.pdf")
    plt.show()


    # quadratic plot
    plt.scatter(male[:, 0], male[:, 1], c="blue")
    plt.scatter(female[:, 0], female[:, 1], c="red")
    maleDensity = util.density_Gaussian(mu_male, cov_male, tmp).reshape(len(X), X.shape[1])
    femaleDensity = util.density_Gaussian(mu_female, cov_female, tmp).reshape(len(Y), Y.shape[1])
    plt.contour(X, Y, maleDensity, colors="blue")
    plt.contour(X, Y, femaleDensity, colors="red")

    # QDA
    male_QDA = np.asarray(maleDensity)
    female_QDA = np.asarray(femaleDensity)
    pred_QDA = male_QDA - female_QDA
    plt.contour(X, Y, pred_QDA, 0)
    axes = plt.gca()
    axes.set_xlim([50, 80])
    axes.set_ylim([80, 280])
    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.title('Height vs. Weight for Quadratic Data Analysis')
    plt.savefig("qda.pdf")
    plt.show()

    return (mu_male,mu_female,cov,cov_male,cov_female)
    

def misRate(mu_male,mu_female,cov,cov_male,cov_female,x,y):
    """
    Use LDA/QDA on the testing set and compute the misclassification rate
    
    Inputs
    ------
    mu_male,mu_female,cov,cov_male,mu_female: parameters from discrimAnalysis
    
    x: a N-by-2 2D array contains the height/weight data of the N samples  
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of two elements: (mis rate in LDA, mis rate in QDA )
    """
    #linear
    male_LDA = mu_male.T@np.linalg.inv(cov)@x.T - 0.5 * mu_male.T@ np.linalg.inv(cov) @ mu_male
    female_LDA = mu_female.T@np.linalg.inv(cov)@x.T - 0.5 * mu_female.T @ np.linalg.inv(cov) @ mu_female
    pred_LDA = male_LDA - female_LDA
    pred_LDA[np.where(pred_LDA >= 0)] = 1
    pred_LDA[np.where(pred_LDA < 0)] = 2

    mis_lda = np.sum((pred_LDA != y))/len(x)

    #quadratic
    male_QDA = list()
    female_QDA = list()

    for i in range(len(x)):
        # male_QDA.append(-0.5*np.log(np.linalg.det(cov_male)) - 0.5*x[i]@np.linalg.inv(cov_male)@x[i].T + mu_male.T@np.linalg.inv(cov_male)@x[i].T - 
        #     0.5*mu_male.T@np.linalg.inv(cov_male)@mu_male)

        # female_QDA.append(-0.5 * np.log(np.linalg.det(cov_female)) - 0.5 * x[i] @ np.linalg.inv(cov_female) @ x[i].T + 
        #     mu_female.T @ np.linalg.inv(cov_female) @ x[i].T - 0.5 * mu_female.T @ np.linalg.inv(cov_female) @ mu_female)

        male_QDA.append(-0.5*np.log(np.linalg.det(cov_male)) - 0.5* np.matmul(np.matmul(x[i].T, np.linalg.inv(cov_male)), x[i]) + np.matmul(np.matmul(x[i].T, np.linalg.inv(cov_male)), mu_male.T) - 
            0.5* np.matmul(np.matmul(mu_male, np.linalg.inv(cov_male)), mu_male.T))

        female_QDA.append(-0.5 * np.log(np.linalg.det(cov_female)) - 0.5 * np.matmul(np.matmul(x[i].T, np.linalg.inv(cov_female)), x[i]) + 
            np.matmul(np.matmul(x[i].T, np.linalg.inv(cov_female)), mu_female.T)  - 0.5 * np.matmul(np.matmul(mu_female, np.linalg.inv(cov_female)), mu_female.T))

    male_QDA = np.asarray(male_QDA)
    female_QDA = np.asarray(female_QDA)

    pred_QDA = male_QDA - female_QDA
    pred_QDA[np.where(pred_QDA >= 0)] = 1
    pred_QDA[np.where(pred_QDA < 0)] = 2

    mis_qda = np.sum((pred_QDA != y)) / len(x)
   
    return (mis_lda, mis_qda)


if __name__ == '__main__':
    
    # load training data and testing data
    x_train, y_train = util.get_data_in_file('trainHeightWeight.txt')
    x_test, y_test = util.get_data_in_file('testHeightWeight.txt')
    
    # parameter estimation and visualization in LDA/QDA
    mu_male,mu_female,cov,cov_male,cov_female = discrimAnalysis(x_train,y_train)
    
    # misclassification rate computation
    mis_lda, mis_qda = mis_LDA,mis_QDA = misRate(mu_male,mu_female,cov,cov_male,cov_female,x_test,y_test)
    
    # print("mis_lda", round(mis_lda, 4))
    # print('mis_qda', round(mis_qda, 4))