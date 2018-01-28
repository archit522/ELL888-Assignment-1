from mnist import MNIST #New Library to load MNIST and EMNIST data
import numpy as np

def load_data():
    mndata = MNIST('/Users/architaggarwal/Desktop/python-mnist/emnist_data') #Loacation of directory containing EMNIST files
    mndata.select_emnist('balanced') #Choose which dataset to use
    images_train, labels_train = mndata.load_training()  #Loading training data
    images_test, labels_test = mndata.load_testing() #Loading testing data
    i_train = []
    l_train = []
    for x, y in zip(labels_train, images_train):
        if x >=10 and x<= 15:  #See the emnist-balanced-mapping.txt file in the dataset (Choosing only letters a-f)
            i_train.append(np.reshape(y, (784, 1))) #Reshaping to (784, 1)
            l_train.append(form_vector(x))  #Converting labels to vectors
    i_test = []
    l_test = []
    for x, y in zip(labels_test, images_test):
        if x>=10 and x<=15:   #See the emnist-balanced-mapping.txt file in the dataset (Choosing only letters a-f)
            i_test.append(np.reshape(y, (784, 1)))#Reshaping to (784, 1)
            l_test.append(x-10)#Converting labels to vectors
    training_data = zip(i_train, l_train) #Zipping training images and labels
    test_data = zip(i_test, l_test) #Zipping testing images and labels
    return (training_data, test_data)

def form_vector(label):
    a = np.zeros((6, 1))
    a[label - 10] = 1
    return a;

#tr_d, te_d = load_data();
#print tr_d[0]


#Refer to this link for python-mnist module
#https://github.com/sorki/python-mnist
