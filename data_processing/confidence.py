import numpy as np
from scipy.linalg import svd
from scipy.optimize import OptimizeResult, OptimizeWarning
from typing import Callable
import scipy.linalg as LA
from scipy.stats.distributions import t, f
import warnings


def confint(n: int, pars: np.ndarray, pcov: np.ndarray, confidence: float = 0.95,
            **kwargs):
    """
    This function returns the confidence interval for each parameter

    Note:
        Adapted from
        http://kitchingroup.cheme.cmu.edu/blog/2013/02/12/Nonlinear-curve-fitting-with-parameter-confidence-intervals/
        Copyright (C) 2013 by John Kitchin.
        https://kite.com/python/examples/702/scipy-compute-a-confidence-interval-from-a-dataset

    Parameters
    ----------
    n: int
        The number of data points
    pars: np.ndarray
        The array with the fitted parameters
    pcov: np.ndarray
        The covariance matrix
    confidence: float
        The confidence interval
    
    Returns
    -------
    np.ndarray:
         The matrix with the confindence intervals for the parameters
    """
    is_log = kwargs.get('is_log', False)
    from scipy.stats.distributions import t

    p = len(pars)  # number of data points
    dof = max(0, n - p)  # number of degrees of freedom

    if is_log:
        p = np.power(10, pars)
        pcov = np.power(10, pcov)

    # Quantile of Student's t distribution for p=(1 - alpha/2)
    # tval = t.ppf((1.0 + confidence)/2.0, dof) 
    alpha = 1.0 - confidence
    tval = t.ppf(1.0 - alpha / 2.0, dof)

    ci = np.zeros((p, 2), dtype=np.float64)

    for i, p, var in zip(range(n), pars, np.diag(pcov)):
        sigma = var ** 0.5
        ci[i, :] = [p - sigma * tval, p + sigma * tval]

    return ci


def confidence_interval(res: OptimizeResult, **kwargs):
    """
    This function estimates the confidence interval for the optimized parameters
    from the fit.
    
    Parameters
    ----------
    res: OptimizeResult
        The optimized result from least_squares minimization
    **kwargs
        confidence: float
            The confidence level (default 0.95)
    Returns
    -------
    np.ndarray
        ci: The confidence interval
    """
    if not isinstance(res, OptimizeResult):
        raise ValueError('Argument \'res\' should be an instance of \'scipy.optimize.OptimizeResult\'')

    level = kwargs.get('confidence', 0.95)
    absolute_sigma = kwargs.get('absolute_sigma', False)

    pcov = ls_covariance(res, absolute_sigma=absolute_sigma)
    # The vector of residuals at the solution
    residuals = res.fun
    # The number of data points
    n = len(residuals)
    # The number of parameters
    p = len(res.x)
    # The degrees of freedom
    dof = n - p

    # Quantile of Student's t distribution for p=(1 - alpha/2)
    # tval = t.ppf((1.0 + confidence)/2.0, dof)
    alpha = 1.0 - level
    tval = t.ppf(1.0 - alpha / 2.0, dof)

    ci = np.zeros((p, 2), dtype=np.float64)

    for i, p, var in zip(range(n), res.x, np.diag(pcov)):
        sigma = var ** 0.5
        ci[i, :] = [p - sigma * tval, p + sigma * tval]

    return ci


def ls_covariance(ls_res: OptimizeResult, absolute_sigma=False):
    """
    Estimates the covariance matrix for a `scipy.optimize.least_squares` result
    :param ls_res: The object returned by `scipy.optimize.least_squares`
    :type ls_res: OptimizeResult
    :param absolute_sigma: If True, `sigma` is used in an absolute sense and the estimated parameter
        covariance `pcov` reflects these absolute values.

        If False (default), only the relative magnitudes of the `sigma` values matter.
        The returned parameter covariance matrix `pcov` is based on scaling
        `sigma` by a constant factor. This constant is set by demanding that the
        reduced `chisq` for the optimal parameters `popt` when using the
        *scaled* `sigma` equals unity. In other words, `sigma` is scaled to
        match the sample variance of the residuals after the fit. Default is False.
        Mathematically,
        ``pcov(absolute_sigma=False) = pcov(absolute_sigma=True) * chisq(popt)/(M-N)``
    :type absolute_sigma: bool
    :return: The covariance matrix of the fit
    :rtype np.ndarray
    """
    popt = ls_res.x
    ysize = len(ls_res.fun)
    cost = 2. * ls_res.cost  # res.cost is half sum of squares!

    # Do Moore-Penrose inverse discarding zero singular values.
    _, s, VT = svd(ls_res.jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(ls_res.jac.shape) * s[0]
    s = s[s > threshold]
    VT = VT[:s.size]
    pcov = np.dot(VT.T / s ** 2, VT)

    if pcov is None or np.isnan(pcov).any():
        # indeterminate covariance
        pcov = np.zeros((len(popt), len(popt)), dtype=float)
        pcov.fill(np.inf)
        warnings.warn('Covariance of the parameters could not be estimated.',
                      category=OptimizeWarning)
    elif not absolute_sigma:
        if ysize > len(popt):
            s_sq = cost / (ysize - len(popt))
            pcov = pcov * s_sq
    return pcov

def get_rsquared(x: np.ndarray, y: np.ndarray, popt: np.ndarray, func: Callable[[np.ndarray, np.ndarray], np.ndarray]):
    """
    This function estimates R\ :sup:`2` \ for the fitting

    Reference:
        http://bagrow.info/dsv/LEC10_notes_2014-02-13.html

    Parameters
    ----------
    x: np.ndarray
        The experimetnal x points
    y: np.ndarray
        The experimental y points
    popt: np.ndarray
        The best fit parameters
    func: Callable[[np.ndarray, np.ndarray]
        The fitted function
    
    Returns
    -------
    float:
        The value of R\ :sup:`2`
    """

    # Get the sum of the residuals from the linear function
    slin = np.sum((y - func(x, *popt)) ** 2)
    # Get the sum of the residuals from the constant function
    scon = np.sum((y - y.mean()) ** 2)
    # Get r-squared
    return 1.0 - slin / scon


def predband(x: np.ndarray, xd: np.ndarray, yd: np.ndarray, p: np.ndarray,
             func: Callable[[np.ndarray, np.ndarray], np.ndarray], conf: float = 0.95):
    """
    This function estimates the prediction bands for the specified function without using the jacobian of the fit
    https://codereview.stackexchange.com/questions/84414/obtaining-prediction-bands-for-regression-model

    Parameters
    ----------
    x: np.ndarray
        The requested data points for the prediction bands
    xd: np.ndarray
        The experimental values for x
    yd: np.ndarray
        The experimental values for y
    p: np.ndarray
        The fitted parameters
    func: Callable[[np.ndarray, np.ndarray], np.ndarray]
        The optimized function
    conf: float
        The confidence level

    Returns
    -------
    np.ndarray:
        The value of the function at the requested points (x)
    np.ndarray:
        The lower prediction band
    np.ndarray:
        The upper prediction band
    """
    alpha = 1.0 - conf  # significance
    npoints = len(xd)  # data sample size
    var_n = len(p)  # number of parameters
    # Quantile of Student's t distribution for p=(1-alpha/2)
    from scipy.stats.distributions import t
    q = t.ppf(1.0 - alpha / 2.0, npoints - var_n)
    # Stdev of an individual measurement
    se = np.sqrt(1. / (npoints - var_n) * np.sum((yd - func(xd, *p)) ** 2))
    # Auxiliary definitions
    sx = (x - xd.mean()) ** 2
    sxd = np.sum((xd - xd.mean()) ** 2)
    # Predicted values (best-fit model)
    yp = func(x, *p)
    # Prediction band
    dy = q * se * np.sqrt(1.0 + (1.0 / npoints) + (sx / sxd))
    # Upper & lower prediction bands.
    lpb, upb = yp - dy, yp + dy
    return yp, lpb, upb


def mean_squared_error(yd: np.ndarray, ym: np.ndarray):
    """
    This function estimates the mean squared error of a fitting.

    Parameters
    ----------
    yd: np.ndarray
        The observed data points
    ym: np.ndarray
        The datapoints from the model

    Returns
    -------
    float:
        The mean squared error
    """
    if len(yd) != len(ym):
        raise ValueError('The length of the observations should be the same ' +
                         'as the length of the predictions.')
    if len(yd) <= 1:
        raise ValueError('Too few datapoints')
    n = len(yd)
    mse = np.sum((yd - ym) ** 2) / n
    return mse


def predint(x: np.ndarray, xd: np.ndarray, yd: np.ndarray, func: Callable[[np.ndarray, np.ndarray], np.ndarray],
            res: OptimizeResult, weights:np.ndarray=None, **kwargs):
    """
    This function estimates the prediction bands for the fit
    (see https://www.mathworks.com/help/curvefit/confidence-and-prediction-bounds.html)

    Parameters 
    ----------
    x: np.ndarray
        The requested x points for the bands
    xd: np.ndarray
        The x datapoints
    yd: np.ndarray
        The y datapoints
    func: Callable[[np.ndarray, np.ndarray]
        The fitted function
    res: OptimizeResult
        The optimzied result from least_squares minimization
    kwargs: dict
        confidence: float
            The confidence level (default 0.95)
        simulateneous: bool
            True if the bound type is simultaneous, false otherwise
        mode: [functional, observation]
            Default observation        
    """

    if len(yd) != len(xd):
        raise ValueError('The length of the observations should be the same ' +
                         'as the length of the predictions.')
    if len(yd) <= 1:
        raise ValueError('Too few datapoints')
    # from scipy.optimize import optimize

    if not isinstance(res, OptimizeResult):
        raise ValueError('Argument \'res\' should be an instance of \'scipy.optimize.OptimizeResult\'')

    simultaneous = kwargs.get('simultaneous', True)
    mode = kwargs.get('mode', 'observation')
    confidence = kwargs.get('confidence', 0.95)
    if mode == 'new_observation':
        mode = 'observation'

    p = len(res.x)

    # Needs to estimate the jacobian at the predictor point!!!
    ypred = func(x, res.x)

    beta = res.x

    if callable(res.jac):
        delta = res.jac(x)
    else:
        delta = np.zeros((len(ypred), p))
        # fdiffstep = np.spacing(np.abs(res.x)) ** (1 / 3)
        fdiffstep = np.finfo(beta.dtype).eps ** (1. / 3.)
        #    print('diff_step = {0}'.format(fdiffstep))
        #    print('popt = {0}'.format(res.x))
        for i in range(p):
            change = np.zeros(p)
            if res.x[i] == 0:
                nb = np.sqrt(LA.norm(beta))
                change[i] = fdiffstep[i] * (nb + (nb == 0))
            else:
                change[i] = fdiffstep[i] * res.x[i]

            predplus = func(x, beta + change)
            delta[:, i] = (predplus - ypred) / change[i]
    #    print('delta = {0}'.format(delta))

    # Find R to get the variance
    _, R = LA.qr(res.jac)
    # Get the rank of jac_pnp
    rankJ = res.jac.shape[1]
    Rinv = LA.pinv(R)
    pinvJTJ = np.dot(Rinv, Rinv.T)

    # The residual
    resid = res.fun
    n = len(resid)
    # Get MSE. The degrees of freedom when J is full rank is v = n-p and n-rank(J) otherwise
    mse = (LA.norm(resid)) ** 2 / (n - rankJ)
    # Calculate Sigma if usingJ 
    Sigma = mse * pinvJTJ

    # Compute varpred
    varpred = np.sum(np.dot(delta, Sigma) * delta, axis=1)
    #    print('varpred = {0}, len: '.format(varpred,len(varpred)))
    alpha = 1.0 - confidence
    if mode == 'observation':
        if not weights is None:
            error_var = mse / weights
        else:
            error_var = mse * np.ones(delta.shape[0])
        varpred += error_var
    # The significance
    if simultaneous:
        from scipy.stats.distributions import f
        if mode == 'observation':
            sch = [rankJ + 1]
        else:
            sch = rankJ
        crit = f.ppf(1.0 - alpha, sch, n - rankJ)
    else:
        from scipy.stats.distributions import t
        crit = t.ppf(1.0 - alpha / 2.0, n - rankJ)

    delta = np.sqrt(varpred) * crit

    lpb = ypred - delta
    upb = ypred + delta

    return ypred, lpb, upb


def prediction_intervals(model: Callable, x_pred, ls_res: OptimizeResult, level=0.95,
                         jac: Callable = None, weights: np.ndarray = None, **kwargs):
    """
    Estimates the prediction interval for a least` squares fit result obtained by
    scipy.optimize.least_squares.

    :param model: The model used to fit the data
    :type model: Callable
    :param x_pred: The values of X at which the model will be evaluated.
    :type x_pred: np.ndarray
    :param ls_res: The result object returned by scipy.optimize.least_squares.
    :type ls_res: OptimizeResult
    :param level: The confidence level used to determine the prediction intervals.
    :type level: float
    :param jac: The Jacobian of the model at the parameters. If not provided,
        it will be estimated from the model. Default None.
    :type jac: Callable
    :param weights: The weights of the datapoints used for the fitting.
    :type weights: np.ndarray
    :param kwargs:
    :return: The predicted values at the given x and the deltas for each prediction
        [y_predicction, delta]
    :rtype: List[np.ndarray, np.ndarray]
    """

    simultaneous = kwargs.get('simultaneous', False)
    new_observation = kwargs.get('new_observation', False)

    # The vector of residuals at the solution
    residuals = ls_res.fun
    beta = ls_res.x
    # The number of data points
    n = len(residuals)
    # The number of parameters
    p = len(beta)
    if n <= p:
        raise ValueError('Not enough data to compute the prediction intervals.')
    # The degrees of freedom
    dof = n - p
    # Quantile of Student's t distribution for p=(1 - alpha/2)
    # tval = t.ppf((1.0 + confidence)/2.0, dof)
    alpha = 1.0 - level

    # Compute the predicted values at the new x_pred
    y_pred = model(x_pred, ls_res.x)
    delta = np.empty((len(y_pred), p), dtype=np.float64)
    fdiffstep = np.finfo(beta.dtype).eps ** (1. / 3.)
    if jac is None:
        for i in range(p):
            change = np.zeros(p)
            if beta[i] == 0:
                nb = np.linalg.norm(beta)
                change[i] = fdiffstep * (nb + float(nb == 0))
            else:
                change[i] = fdiffstep * beta[i]
            predplus = model(x_pred, beta + change)
            delta[:, i] = (predplus - y_pred) / change[i]
    else:
        delta = jac(beta, x_pred, y_pred)

    J = ls_res.jac

    # Find R to get the variance
    _, R = np.linalg.qr(J)
    # Get the rank of jac_pnp
    rankJ = J.shape[1]
    Rinv = np.linalg.pinv(R)
    pinvJTJ = np.dot(Rinv, Rinv.T)

    # Get MSE. The degrees of freedom when J is full rank is v = n-p and n-rank(J) otherwise
    mse = (np.linalg.norm(residuals)) ** 2. / (n - rankJ)

    # Calculate Sigma if usingJ
    sigma = mse * pinvJTJ

    # Compute varpred
    varpred = np.sum(np.dot(delta, sigma) * delta, axis=1)

    if new_observation:
        if not weights is None:
            error_var = mse / weights
        else:
            error_var = mse * np.ones(delta.shape[0])
        varpred += error_var

    if simultaneous:
        if new_observation:
            sch = [rankJ + 1]
        else:
            sch = rankJ
        crit = np.sqrt(sch * (f.ppf(1.0 - alpha, sch, n - rankJ)))
    else:
        from scipy.stats.distributions import t
        crit = t.ppf(1.0 - alpha / 2.0, n - rankJ)

    delta = np.sqrt(varpred) * crit

    return y_pred, delta


def get_pcov(res: OptimizeResult, absolute_sigma:bool = False) -> np.ndarray:
    popt = res.x
    ysize = len(res.fun)
    cost = 2 * res.cost  # res.cost is half sum of squares!
    s_sq = cost / (ysize - popt.size)

    # Do Moore-Penrose inverse discarding zero singular values.
    _, s, VT = svd(res.jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
    s = s[s > threshold]
    VT = VT[:s.size]
    pcov = np.dot(VT.T / s ** 2, VT)
    pcov = pcov * s_sq

    if pcov is None or np.isnan(pcov).any():
        # indeterminate covariance
        pcov = np.zeros((len(popt), len(popt)), dtype=float)
        pcov.fill(np.inf)
        warnings.warn('Covariance of the parameters could not be estimated.',
                      category=OptimizeWarning)
    elif not absolute_sigma:
        if ysize > len(popt):
            s_sq = cost / (ysize - len(popt))
            pcov = pcov * s_sq
    return pcov



# References:
# - Statistics in Geography by David Ebdon (ISBN: 978-0631136880)
# - Reliability Engineering Resource Website:
# - http://www.weibull.com/DOEWeb/confidence_intervals_in_simple_linear_regression.htm
# - http://reliawiki.com/index.php/Simple_Linear_Regression_Analysis#Confidence_Intervals_in_Simple_Linear_Regression
# - University of Glascow, Department of Statistics:
# - http://www.stats.gla.ac.uk/steps/glossary/confidence_intervals.html#conflim
