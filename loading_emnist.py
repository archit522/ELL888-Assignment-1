from mnist import MNIST #New Library to load MNIST and EMNIST data
import numpy as np

def load_data():
    mndata = MNIST('/Users/architaggarwal/Desktop/python-mnist/emnist_data') #Loacation of directory containing EMNIST files
    mndata.select_emnist('letters') #Choose which dataset to use
    images_train, labels_train = mndata.load_training()  #Loading training data
    images_test, labels_test = mndata.load_testing() #Loading testing data
    images_train = [np.reshape(x, (784, 1)) for x in images_train]  #Reshaping images
    labels_train = [form_vector(x) for x in labels_train]  #Converting labels to vectors
    images_test = [np.reshape(x, (784, 1)) for x in images_test]
    labels_test = [form_vector(x) for x in labels_test]
    training_data = zip(images_train, labels_train) #Zipping training images and labels
    test_data = zip(images_test, labels_test) #Zipping testing images and labels
    return (training_data, test_data)

def form_vector(label):
    a = np.zeros((26, 1))
    a[label - 1] = 1
    return a;

tr_d, te_d = load_data()
print tr_d[0]

#Refer to this link for python-mnist module
#https://github.com/sorki/python-mnist
