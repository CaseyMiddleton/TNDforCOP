from sklearn.linear_model import LogisticRegression
from pygam import LinearGAM, LogisticGAM, s
from scipy.optimize import curve_fit, minimize
import numpy as np
import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)
import math

def fit_logistic_regression(x_data,y_data,x_vals_to_predict):
    ''' Fit logistic regression model using sklearn
    Inputs:
        x_data, y_data: column vectors of predictor (x) and outcome (y) variables
        x_vals_to_predict: predcitor variables for which we will generate predictions
    Returns 
        probs: array of predicted output for all x_vals_to_predict
    '''
    # need to reshape these arrays for LogisticRegression
    x_data = np.array(x_data).reshape(-1,1); y_data = np.array(y_data).reshape(-1,1); 
    x_vals_to_predict=np.array(x_vals_to_predict).reshape(-1,1)

    # Perform logistic regression to get P(y|x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            log_reg = LogisticRegression(penalty='none', solver="lbfgs", max_iter=1000)
        except:
            log_reg = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        log_reg.fit(x_data, y_data)
        # get probabilities for all x_vals_to_predict
        probs = log_reg.predict_proba(x_vals_to_predict)[:,1] # column 1 contains probability of infection

    return probs

def fit_logistic_regression_with_params(x_data,y_data,x_vals_to_predict):
    ''' Fit logistic regression model using sklearn
    Inputs:
        x_data, y_data: column vectors of predictor (x) and outcome (y) variables
        x_vals_to_predict: predcitor variables for which we will generate predictions
    Returns 
        probs: array of predicted output for all x_vals_to_predict
        log_reg.intercept_: fitted intercept term
        log_reg.coef_: fitted coefficient
    '''
    # need to reshape these arrays for LogisticRegression
    x_data = np.array(x_data).reshape(-1,1); y_data = np.array(y_data).reshape(-1,1); 
    x_vals_to_predict=np.array(x_vals_to_predict).reshape(-1,1)

    # Perform logistic regression to get P(y|x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            log_reg = LogisticRegression(penalty='none', solver="lbfgs", max_iter=1000)
        except:
            log_reg = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        log_reg.fit(x_data, y_data)
        # get probabilities for all x_vals_to_predict
        probs = log_reg.predict_proba(x_vals_to_predict)[:,1] # column 1 contains probability of infection

    return probs, log_reg.intercept_[0], log_reg.coef_[0]

def logistic(beta_0,beta_1,x):
    ''' 
    Logistic function
        beta_0: intercept term
        beta_1: slope term
        x: scalar or vector of independent variable data
    '''
    return np.exp(beta_0 + beta_1 * x)/(1 + np.exp(beta_0 + beta_1 * x))

def get_NLL_logistic(params, data):
    ''' 
    Negative log likelihood of logistic regression
    '''
    beta_0 = params[0]; beta_1 = params[1]
    Abs = np.array(data[0]); infected = np.array(data[1])
    preds = logistic(beta_0,beta_1,Abs)

    epsilon = 1e-10  # To prevent log(0)
    prob_pos = infected*np.log(np.clip(preds,epsilon,1 - epsilon)) 
    prob_neg = (1-infected)*np.log(np.clip(1-preds,epsilon,1-epsilon))
    return -1*np.sum(prob_pos + prob_neg)

# Define the scaled logit model
def scaled_logit(x, k, beta_0, beta_1):
    """
    Scaled logistic function
    
    Parameters:
    x (array-like): Input values.
    k (float): Maximum value (scale).
    beta_0 (float): Intercept parameter for linear regression
    beta_1 (float): Slope parameter.
    
    Returns:
    array-like: Scaled logistic function values.
    """
    return k / (1 + np.exp(beta_0 + beta_1*x))

def neg_log_likelihood_scaled_logit(params, data, lambda_reg=0.1):
    # unpack data
    k, beta_0, beta_1 = params
    Abs = np.array(data[0]); infected = np.array(data[1])

    # Get likelihood
    epsilon = 1e-10  # To prevent log(0)
    prob_pos = infected * np.log(np.clip(scaled_logit(Abs, k, beta_0, beta_1), epsilon, 1 - epsilon))
    prob_neg = (1 - infected) * np.log(np.clip(1 - scaled_logit(Abs, k, beta_0, beta_1), epsilon, 1 - epsilon))
    
    # Add L2 regularization penalty on beta_1 to discourage steep slopes
    regularization_penalty = lambda_reg * beta_1**2
    
    return -1*np.sum(prob_pos + prob_neg) + regularization_penalty

# Modified function to fit the scaled logit model with regularization
def fit_scaled_logit(x_data, y_data, lambda_reg=0.1, initial_guess=(0.5, -1, 1)):
    """
    Fit the scaled logistic regression model with regularization to avoid steep slopes.
    
    Parameters:
    x_data (array-like): Independent variable values.
    y_data (array-like): Dependent variable values.
    lambda_reg (float): Regularization parameter. Higher values penalize steeper slopes more.
    initial_guess (tuple): Initial guesses for k, beta_0, and beta_1.
    
    Returns:
    tuple: Fitted parameters (k, beta_0, beta_1).
    """
    data = [x_data, y_data]

    result = minimize(neg_log_likelihood_scaled_logit, initial_guess, 
                     method='Nelder-Mead', args=(data, lambda_reg),
                     options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 10000, 'maxfev': 20000})
    
    if not result.success:
        print("Did not find optimal parameter fit to minimize likelihood")
    return result.x


def one_minus_OR(predicted_probabiliies):
    ''' 
    1 - odds ratio
    '''
    baseline_odds = predicted_probabiliies[0] / (1-predicted_probabiliies[0])
    VE = [1 - (pred/(1-pred))/baseline_odds for pred in predicted_probabiliies]
    return VE 

def get_L2_norm_error(estimate,true):
    '''Calculate L2 norm '''
    diff = np.array(true) - np.array(estimate)
    sum_squares = np.sum(diff**2)
    return math.sqrt(sum_squares)