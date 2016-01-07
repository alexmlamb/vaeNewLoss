import numpy as np
import gzip
import cPickle


class SvhnData:

    def __init__(self):
        pass

    def normalize(self, x):
        pass

    def denormalize(self, x):
        pass


    def getBatch(self):
        pass

#Returns list of tuples containing training, validation, and test instances.
def load_data_svhn(config):

    np.random.seed(config["seed"])

    import scipy.io as sio
    train_file = config["svhn_file_train"]
    extra_file = config["svhn_file_extra"]
    test_file = config["svhn_file_test"]

    train_object = sio.loadmat(train_file)
    extra_object = sio.loadmat(extra_file)
    test_object = sio.loadmat(test_file)

    print "objects loaded"

    # Note from Guillaume : This is NOT what we agreed on.
    # This is loading the data and converting it into an 8 GB array.
    # We had agreed to store this as uint8 temporarily and do the
    # conversion on a mini-batch basis.
    train_X = np.asarray(train_object["X"], dtype = 'uint8')
    extra_X = np.asarray(extra_object["X"], dtype = 'uint8')
    test_X = np.asarray(test_object["X"], dtype = 'uint8')

    train_Y = np.asarray(train_object["y"], dtype = 'uint8')
    extra_Y = np.asarray(extra_object["y"], dtype = 'uint8')
    test_Y = np.asarray(test_object["y"], dtype = 'uint8')

    print "converted to np arrays"

    del train_object
    del extra_object
    del test_object

    #By default SVHN labels are from 1 to 10.
    #This shifts them to be between 0 and 9.
    train_Y -= 1
    extra_Y -= 1
    test_Y -= 1

    assert train_Y.min() == 0
    assert train_Y.max() == 9

    train_X = np.swapaxes(train_X, 0,3)

    extra_X = np.swapaxes(extra_X, 0,3)

    test_X = np.swapaxes(test_X, 0,3)

    print "axes swapped"

    train_X = np.vstack((train_X, extra_X))
    train_Y = np.vstack((train_Y, extra_Y))

    print "vstacked"

    # It's super important that all the workers/master use the same shuffling scheme
    # because otherwise we can't talk about the "importance weight of training example 17"
    # if nobody agrees on which training example has index 17.
    old_seed = np.random.randint(low=0, high=np.iinfo(np.uint32).max)
    np.random.seed(42)
    train_indices = np.random.choice(train_X.shape[0], int(train_X.shape[0] * (1.0 - config["fraction_validation"])), replace = False)
    valid_indices = np.setdiff1d(range(0,train_X.shape[0]), train_indices)


    valid_X = train_X[valid_indices]
    valid_Y = train_Y[valid_indices]

    train_X = train_X[train_indices]
    train_Y = train_Y[train_indices]

    if 'noise' in config.keys() and config['noise'] != 'no_noise':
        noise_indices = np.random.choice(train_X.shape[0], int(train_X.shape[0] * (config["fraction_noise"])), replace = False)
        print np.amax(noise_indices)
        train_X[noise_indices] = noisify(train_X[noise_indices], config)
    else :
        print "Not adding any noise"

    np.random.seed(old_seed)

    assert not (config["load_svhn_normalization_from_file"] and config["save_svhn_normalization_to_file"])

    #get mean and std for each filter and each pixel.
    if not config["load_svhn_normalization_from_file"]:

        # The commented-out alternative did not work properly.
        #import safe_mean_std_var
        #x_mean, x_std, _ = safe_mean_std_var.mean_std_var(train_X, axis=0)
        #assert np.all(np.isfinite(x_mean))
        #assert np.all(np.isfinite(x_std))
        x_mean = train_X.mean(axis = (0))
        x_std = train_X.std(axis = (0))

        if config["save_svhn_normalization_to_file"]:
            cPickle.dump({"mean" : x_mean, "std" : x_std}, open(config["svhn_normalization_value_file"], "w"), protocol = cPickle.HIGHEST_PROTOCOL)
    else:
        svhn_normalization_values = cPickle.load(open(config["svhn_normalization_value_file"]))
        x_mean = svhn_normalization_values["mean"]
        x_std = svhn_normalization_values["std"]

    print "computed mean and var"

    print "Training Set", train_X.shape, train_Y.shape
    print "Validation Set", valid_X.shape, valid_Y.shape
    print "Test Set", test_X.shape, test_Y.shape

    return {"train": (train_X, train_Y.flatten()), "valid" : (valid_X, valid_Y.flatten()), "test" : (test_X, test_Y.flatten()), "mean" : x_mean, "std" : x_std}





