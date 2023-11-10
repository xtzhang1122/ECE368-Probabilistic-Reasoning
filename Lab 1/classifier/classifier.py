import os.path
import numpy as np
import matplotlib.pyplot as plt
import util

def learn_distributions(file_lists_by_category):
    """
    Estimate the parameters p_d, and q_d from the training set
    
    Input
    -----
    file_lists_by_category: A two-element list. The first element is a list of 
    spam files, and the second element is a list of ham files.

    Output
    ------
    probabilities_by_category: A two-element tuple. The first element is a dict 
    whose keys are words, and whose values are the smoothed estimates of p_d;
    the second element is a dict whose keys are words, and whose values are the 
    smoothed estimates of q_d 
    """

    # frequency - dictionary {key - word : value : frequency}
    # words - total frequency

    spam_frequency = util.get_word_freq(file_lists_by_category[0])
    spam_words = sum(spam_frequency.values())
    ham_frequency = util.get_word_freq(file_lists_by_category[1])
    ham_words = sum(ham_frequency.values())

    unique_words = len(util.get_word_freq(file_lists_by_category[0] + file_lists_by_category[1]))

    for i in spam_frequency:
        spam_frequency[i] = (spam_frequency[i] + 1) / (spam_words + unique_words)

    for i in ham_frequency:
        ham_frequency[i] = (ham_frequency[i] + 1) / (ham_words + unique_words)
    
    smooth_spam = 1 / (spam_words + unique_words)
    smooth_ham  = 1 / (ham_words + unique_words)

    probabilities_by_category = (spam_frequency, ham_frequency, smooth_spam, smooth_ham)
    
    return probabilities_by_category

def classify_new_email(filename,probabilities_by_category,prior_by_category):
    """
    Use Naive Bayes classification to classify the email in the given file.

    Inputs
    ------
    filename: name of the file to be classified
    probabilities_by_category: output of function learn_distributions
    prior_by_category: A two-element list as [\pi, 1-\pi], where \pi is the 
    parameter in the prior class distribution

    Output
    ------
    classify_result: A two-element tuple. The first element is a string whose value
    is either 'spam' or 'ham' depending on the classification result, and the 
    second element is a two-element list as [log p(y=1|x), log p(y=0|x)], 
    representing the log posterior probabilities
    """

    spam_prob = 0
    ham_prob = 0

    dictionary = util.get_word_freq([filename])
    
    for i in dictionary:
        if i in probabilities_by_category[0]:
            spam_prob += np.log(probabilities_by_category[0][i]) * dictionary[i]
        elif i in probabilities_by_category[1]:
            spam_prob += np.log(probabilities_by_category[2]) * dictionary[i]

        if i in probabilities_by_category[1]:
            ham_prob += np.log(probabilities_by_category[1][i]) * dictionary[i]
        elif i in probabilities_by_category[0]:
            ham_prob += np.log(probabilities_by_category[3]) * dictionary[i]

    if spam_prob + np.log(prior_by_category[0]) > ham_prob + np.log(prior_by_category[1]):
        classify_result = ("spam", [spam_prob, ham_prob])
    else:
        classify_result = ("ham", [spam_prob, ham_prob])

    return classify_result

if __name__ == '__main__':
    
    # folder for training and testing 
    spam_folder = "data/spam"
    ham_folder = "data/ham"
    test_folder = "data/testing"

    # generate the file lists for training
    file_lists = []
    for folder in (spam_folder, ham_folder):
        file_lists.append(util.get_files_in_folder(folder))
        
    # Learn the distributions    
    probabilities_by_category = learn_distributions(file_lists)
    
    # prior class distribution
    priors_by_category = [0.5, 0.5]
    
    # Store the classification results
    performance_measures = np.zeros([2,2])
    # explanation of performance_measures:
    # columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    # to be more clear, performance_measures = [[p1 p2]
    #                                           [p3 p4]]
    # p1 = Number of emails whose true label is 'spam' and classified as 'spam' 
    # p2 = Number of emails whose true label is 'spam' and classified as 'ham' 
    # p3 = Number of emails whose true label is 'ham' and classified as 'spam' 
    # p4 = Number of emails whose true label is 'ham' and classified as 'ham' 

    # Classify emails from testing set and measure the performance
    for filename in (util.get_files_in_folder(test_folder)):
        # Classify
        label,log_posterior = classify_new_email(filename,
                                                 probabilities_by_category,
                                                 priors_by_category)
        
        # Measure performance (the filename indicates the true label)
        base = os.path.basename(filename)
        true_index = ('ham' in base) 
        guessed_index = (label == 'ham')
        performance_measures[int(true_index), int(guessed_index)] += 1

    template="You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
    # Correct counts are on the diagonal
    correct = np.diag(performance_measures)
    # totals are obtained by summing across guessed labels
    totals = np.sum(performance_measures, 1)
    print(template % (correct[0],totals[0],correct[1],totals[1]))
    
    
    ### TODO: Write your code here to modify the decision rule such that
    ### Type 1 and Type 2 errors can be traded off, plot the trade-off curve
   
    type_one = list()
    type_two = list()
    gamma_list = [1E-5, 1E-4, 1E-3, 1E-2, 1E-1, 0.5, 1, 10, 100, 1000, 10000]

    for gamma in gamma_list:
        performance_measures = np.zeros([2, 2])

        for filename in (util.get_files_in_folder(test_folder)):
            label, log_posterior = classify_new_email(filename, probabilities_by_category, priors_by_category)

            if log_posterior[0] - log_posterior[1] > gamma:
                label = 'spam'
            else:
                label = 'ham'

            # check prediction
            b = os.path.basename(filename)
            true_index = ('ham' in b)
            guessed_index = (label == 'ham')
            performance_measures[int(true_index), int(guessed_index)] += 1

        type_one.append(performance_measures[0][1])
        type_two.append(performance_measures[1][0])
    
    # print(type_one)
    # print(type_two)
    plt.plot(type_one, type_two)
    plt.xlabel('Type 1 Error')
    plt.ylabel('Type 2 Error')
    plt.title('Type 1 Error and Type 2 Error Trade-off')
    plt.savefig("nbc.pdf")
    plt.show()