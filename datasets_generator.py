import numpy as np

def datasets_generator() :
    dataset_sizes = np.logspace(0,6,7)

    Xs_train = [np.random.randn(nb_sample, nb_pred) for nb_sample, nb_pred in zip(dataset_sizes, dataset_sizes)]
    ys_train = [np.random.randint(0,2,nb_sample) for nb_sample in dataset_sizes]
    return Xs_train, ys_train