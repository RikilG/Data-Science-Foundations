import numpy as np

def normal(dataset):
    """perform normalization based on normal distribution
    """

    temp    = dataset.iloc[:, 0 ]
    dataset.iloc[:, 0 ] = (temp - np.mean(temp))/np.std(temp)
    temp    = dataset.iloc[:, 1 ]
    dataset.iloc[:, 1 ] = (temp - np.mean(temp))/np.std(temp)
    return dataset


def min_max(dataset):
    """perform normalization based on min and max values
    """

    temp    = dataset.iloc[:, 0 ]
    dataset.iloc[:, 0 ] = (temp - np.min(temp))/(np.max(temp) - np.min(temp))
    temp    = dataset.iloc[:, 1 ]
    dataset.iloc[:, 1 ] = (temp - np.min(temp))/(np.max(temp) - np.min(temp))
    return dataset


def normalize(dataset, type="min-max"):
    """Normalize the given dataset
    """
    if type=="min-max": return min_max(dataset)
    else: # type=="normal"
        return normal(dataset)

def data_split(dataset, split_at=0.8, random_state=42):
    """split data into partitions randomly
    """
    train   = dataset.sample(frac=split_at, random_state=random_state)
    test    = dataset.drop(train.index)
    return train, test

def xy_split(dataset, target):
    """split dataset into factors and output index
    """
    x = dataset.drop(columns=target)
    temp = list(dataset.columns)
    temp.remove(target)
    y = dataset.drop(columns=temp)
    return x, y