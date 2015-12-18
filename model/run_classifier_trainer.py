from config import get_config
import load_data
from HiddenLayer import HiddenLayer
import Updates
import theano
import theano.tensor as T
import numpy as np
import cPickle

import time

from ConvolutionalLayer import ConvPoolLayer

import sys

sys.setrecursionlimit(99999999)

config = get_config()

'''
Responsible for training classifier on dataset.  Saves classifier to file and loads from file.  

'''

classifier_learning_rate = 0.001

class Classifier:




    def __init__(self, config, mean, std):
        assert "use_convolutional_classifier" in config
        assert (not config["use_convolutional_classifier"])

        self.x = T.tensor4()

        self.y = T.ivector()

        x_normed = load_data.normalizeMatrix(self.x, mean, std)

        c1 = ConvPoolLayer(input=x_normed, in_channels = 3, out_channels = 96, kernel_len = 5, in_rows = 32, in_columns = 32, batch_size = 100,
                                        convstride=1, padsize=4,
                                        poolsize=3, poolstride=2,
                                        bias_init=0.0, name = "h1"
                                        )

        c2 = ConvPoolLayer(input=c1.output, in_channels = 96, out_channels = 128, kernel_len = 3, in_rows = 17, in_columns = 17, batch_size = 100,
                                        convstride=1, padsize=3,
                                        poolsize=3, poolstride=2,
                                        bias_init=0.0, name = "h2"
                                        )

        c3 = ConvPoolLayer(input=c2.output, in_channels = 128, out_channels = 256, kernel_len = 3, in_rows = 10, in_columns = 10, batch_size = 100,
                                        convstride=1, padsize=0,
                                        poolsize=3, poolstride=2,
                                        bias_init=0.0, name = "h3"
                                        )

        h1 = HiddenLayer(c3.output.flatten(2), num_in = 2304, num_out = 2048, initialization = 'xavier', name = "h1", activation = 'relu')

        h2 = HiddenLayer(h1.output, num_in = 2048, num_out = 2048, initialization = 'xavier', name = "h2", activation = 'relu')

        h3 = HiddenLayer(h2.output + h1.output, num_in = 2048, num_out = 2048, initialization = 'xavier', name = "h3", activation = 'relu')

        h4 = HiddenLayer(h3.output + h2.output, num_in = 2048, num_out = 2048, initialization = 'xavier', name = "h4", activation = 'relu')

        y_out = HiddenLayer(h4.output + h3.output, num_in = 2048, num_out = config["num_output"], initialization = 'xavier', name = "h_out", activation = None)

        py_x = T.nnet.softmax(y_out.output)

        individual_cost = -1.0 * (T.log(py_x)[T.arange(self.y.shape[0]), self.y])

        acc = T.eq(self.y, T.argmax(py_x, axis = 1))

        params = {}


        self.hidden_map = {'c1' : c1, 'c2' : c2, 'c3' : c3, 'h1' : h1, 'h2' : h2, 'h3' : h3, 'h4' : h4, 'y' : y_out}
        #self.hidden_map = {'h1' : h1.output, 'h2' : h2.output, 'h3' : h3.output, 'h4' : h4.output, 'y' : y_out.output}

        layers = self.hidden_map.values()

        self.doTrain = T.scalar()

        for layer in layers:
            layerParams = layer.getParams()
            for paramKey in layerParams:
                params[paramKey] = layerParams[paramKey]

        print "params", params

        updateObj = Updates.Updates(params, self.doTrain * T.sum(individual_cost), classifier_learning_rate)

        updates = updateObj.getUpdates()

        self.train_method = theano.function(inputs = [self.x, self.y, self.doTrain], outputs = {'cost' : T.mean(individual_cost), 'acc' : T.mean(acc), 'py_x' : py_x}, updates = updates)


    #Returns error rate
    def train(self, x, y, doTrain):
        return self.train_method(x,y, doTrain)

    #Returns error rate
    def validate(x,y):
        pass


    def tofile():
        pass


    @classmethod
    def fromfile():
        pass

if __name__ == "__main__":

    config = get_config()

    data = load_data.load_data_svhn(config)

    classifier = Classifier(config, mean = data['mean'], std = data['std'])

    totalCost = 0.0
    totalAcc = 0.0
    numSeen = 0.0

    for iteration in range(0, 1000 * 50):

        index = (iteration * 100) % data["train"][0].shape[0]
        index_test = (iteration * 100) % data["test"][0].shape[0]

        x = data["train"][0][index : index + 100]
        y = data["train"][1][index : index + 100]

        x_test = data['test'][0][index_test : index_test + 100]
        y_test = data['test'][1][index_test : index_test + 100]

        if x.shape[0] != 100 or x_test.shape[0] != 100:
            continue

        if config["dataset"] == 'svhn':
            x = np.swapaxes(x, 2, 3).swapaxes(1,2)
            x_test = np.swapaxes(x_test, 2, 3).swapaxes(1,2)

        #x = x.reshape(100, config['num_input'])
        #x_test = x_test.reshape(100, config['num_input'])

        results = classifier.train(x,y,1)
        results_test = classifier.train(x_test,y_test,0)
        cost = results_test['cost']
        acc = results_test['acc']
        totalCost += cost
        totalAcc += acc
        numSeen += 1

        #Evaluate on test data
        if iteration % 500 == 0:
            print "total cost", totalCost / numSeen
            print "total acc", totalAcc / numSeen
            print "Thousands of iterations", iteration / 1000
            totalCost = 0.0
            totalAcc = 0.0
            numSeen = 0.0

            #save hidden method

    cPickle.dump(classifier, open("model_files/get_hidden_svhn_" + str(int(time.time())) + ".pkl", "w"))


