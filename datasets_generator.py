import numpy as np

def datasets_generator(D_sup_N = True) :
    dataset_sizes = np.around(np.logspace(1,5,5)).astype(int)
    if D_sup_N :
        Xs_train = [np.random.randn(nb_sample, nb_pred) for nb_sample, nb_pred in zip(dataset_sizes, dataset_sizes)]
    else : 
        Xs_train = [np.random.randn(nb_sample, nb_pred) for nb_sample, nb_pred in zip(dataset_sizes, dataset_sizes) if nb_pred <= nb_sample]
    ys_train = [np.random.randint(0,2,nb_sample) for nb_sample in dataset_sizes]
    return Xs_train, ys_train