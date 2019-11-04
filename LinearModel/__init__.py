from Optimizers import StdGradDesc
from Optimizers import StocasticGradDesc
from Optimizers import NormEqns
from Optimizers import RegGradDesc
from LinearModel.LinearModelStats import f, error, test


def fit(x_train, y_train, alpha, epsilion, method="GD"):
    """fits the linear model to the given data
    """

    module = StdGradDesc
    if method=="NE": module=NormEqns
    elif method=="GD": module=StdGradDesc
    elif method=="SGD": module=StocasticGradDesc

    w = module.run(x_train, y_train, function=f, error=error, alpha=alpha, epsilion=epsilion)
    return w


def reg_fit(train, alpha, epsilion, method="L2GD"):
    """
    """

    module = L2GradDesc

    if method=="L1GD": module=L1GradDesc
    elif method=="L2GD": module=L2GradDesc

    w = module.run(train, function=f, error=error, alpha=alpha, epsilion=epsilion, reg_type=method)
    return w