

def get_config():

    config = {}

    config["mb_size"] = 50

    #config['dataset'] = 'cifar'
    #config["dataset"] = "svhn"
    config['dataset'] = 'imagenet'
    #config['dataset'] = 'celeb'

    if config["dataset"] == "svhn":
        config["image_width"] = 32
    elif config['dataset'] == 'imagenet':
        config["image_width"] = 128
        config["num_labels"] = 1
    elif config['dataset'] == 'cifar':
        config['image_width'] = 32
        config['num_labels'] = 10
    elif config['dataset'] == 'celeb':
        config['image_width'] = 128

    config["imagenet_location"] = "/u/lambalex/data/imagenet/"
    config["cifar_location"] = "/u/lambalex/data/cifar/cifar-10-batches-py/"

    config["plot_output_directory"] = "/u/lambalex/plots/"

    config["mnist_file"] = "/data/lisatmp4/lambalex/mnist/mnist.pkl.gz"
    config["svhn_file_train"] = "/data/lisatmp4/lambalex/svhn/train_32x32.mat"
    config["svhn_file_extra"] = "/data/lisatmp4/lambalex/svhn/extra_32x32.mat"
    config["svhn_file_test"] = "/data/lisatmp4/lambalex/svhn/test_32x32.mat"

    config['vgg19_file'] = '/u/lambalex/trained_models/vgg-19/vgg19_normalized.pkl'

    config["seed"] = 9999494

    #config["use_convolutional_classifier"] = False

    #config["save_model_file_name"] = 'model_file'

    #config["save_svhn_normalization_to_file"] = False
    #config["load_svhn_normalization_from_file"] = True

    #config["svhn_normalization_value_file"] = "/data/lisatmp4/lambalex/svhn/svhn_normalization_values.pkl"

    #config["hidden_sizes"] = [2048, 2048, 2048, 2048]

    #Weights are initialized to N(0,1) * initial_weight_size
    #config["initial_weight_size"] = 0.01

    #Hold this fraction of the instances in the validation dataset
    #config["fraction_validation"] = 0.05
    

    return config


