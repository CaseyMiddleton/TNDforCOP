from sklearn.linear_model import LogisticRegression
from pygam import LinearGAM, LogisticGAM, s
from scipy.optimize import curve_fit, minimize
import numpy as np
import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)
import math

def fit_logistic_regression(x_data,y_data,x_vals_to_predict):
    ''' Fit logistic regression model using sklearn and return prediction
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
    ''' Fit logistic regression model using sklearn and return fitted parameters
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



# ---------------------------------------------------------------
# Adding covariate ------------------------------------------
# ---------------------------------------------------------------

def rescale_to_unit_interval(x):
    """
    Rescale array to [0,1] using min-max scaling.
    """
    x = np.array(x, dtype=float)
    x_min, x_max = np.min(x), np.max(x)
    if x_max == x_min:  # edge case: constant array
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min), x_min, x_max

def scaled_logit_with_cov(x, cov, k, beta_0, beta_1, beta_cov=0, beta_int=0):
    """
    Scaled logistic with binary covariate and optional interaction.
    """
    linear_term = beta_0 + beta_1 * x + beta_cov * cov + beta_int * (x * cov)
    return k / (1 + np.exp(linear_term))

def neg_log_likelihood_scaled_logit_cov(params, data, lambda_reg=0.1, use_interaction=False):
    """
    Negative log-likelihood for scaled logistic with binary covariate.
    """
    if use_interaction:
        k, beta_0, beta_1, beta_cov, beta_int = params
    else:
        k, beta_0, beta_1, beta_cov = params
        beta_int = 0.0

    Abs, cov, infected = data # assumes Abs are rescaled to be on interval [0,1]
    epsilon = 1e-10

    prob_inf = scaled_logit_with_cov(Abs, cov, k, beta_0, beta_1, beta_cov, beta_int)
    
    prob_pos = infected * np.log(np.clip(prob_inf, epsilon, 1 - epsilon))
    prob_neg = (1 - infected) * np.log(np.clip(1 - prob_inf, epsilon, 1 - epsilon))
    
    # Regularization penalty on slopes
    regularization_penalty = lambda_reg * (beta_1**2 + beta_int**2)

    return -np.sum(prob_pos + prob_neg) + regularization_penalty


def fit_scaled_logit_cov(x_data, cov_data, y_data, 
                         lambda_reg=0.1, 
                         initial_guess=None, 
                         use_interaction=False):
    """
    Fit scaled logit model with binary covariate (+ optional interaction).
    Antibody titers are rescaled to [0,1] internally.
    """
    # Rescale antibody titers
    x_scaled, x_min, x_max = rescale_to_unit_interval(x_data)

    # Default initial guess
    if initial_guess is None:
        if use_interaction:
            initial_guess = (0.5, -1, 1, 0, 0)  # k, beta_0, beta_1, beta_cov, beta_int
        else:
            initial_guess = (0.5, -1, 1, 0)

    data = [x_scaled, np.array(cov_data), np.array(y_data)]

    result = minimize(
        neg_log_likelihood_scaled_logit_cov,
        initial_guess,
        method="Nelder-Mead",
        args=(data, lambda_reg, use_interaction),
        options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 10000, "maxfev": 20000}
    )

    if not result.success:
        print("Did not find optimal parameter fit:", result.message)

    return result.x, (x_min, x_max)  # return scaling too

def predict_scaled_logit_cov(x_vals, cov_vals, params, scale_info, use_interaction=False):
    """
    Predict infection probabilities using fitted scaled logit with covariate.
    
    x_vals : antibody titers (unscaled)
    cov_vals : binary covariate (0/1)
    params : fitted params from fit_scaled_logit_cov
    scale_info : (x_min, x_max) tuple for rescaling
    """
    x_min, x_max = scale_info
    x_scaled = (np.array(x_vals) - x_min) / (x_max - x_min)

    if use_interaction:
        k, beta_0, beta_1, beta_cov, beta_int = params
    else:
        k, beta_0, beta_1, beta_cov = params
        beta_int = 0.0

    return scaled_logit_with_cov(x_scaled, cov_vals, k, beta_0, beta_1, beta_cov, beta_int)


# ---------------------------------------------------------------
# Polynomial expansion of logsitic regression -----------------
# ---------------------------------------------------------------


def generate_polynomial_terms(X,terms):
    polynomial_X = []
    for t in terms:
        polynomial_X.append(X**t)

    arr = np.array(polynomial_X)
    polynomial_X = arr.transpose(1, 0, 2).reshape(arr.shape[1], arr.shape[0])

    return polynomial_X

def fit_polynomial_logistic_regression(titers, infection_status, potential_titers, terms=[1, 3]):
    """
    Fit a logistic regression model with selected polynomial terms to predict infection risk,
    and return predictions for potential_titers.
    
    Parameters:
    - titers: array-like, antibody titers (independent variable).
    - infection_status: array-like, infection statuses (1 = infected, 0 = not infected).
    - potential_titers: array-like, titers for which to predict infection probability.
    - terms: list of polynomial terms to include (e.g., [1, 3] = Ab and Ab^3).
    
    Returns:
    - y_pred: predicted probabilities for potential_titers.
    """
    # Scale titers
    scaler = StandardScaler()
    titers_scaled = scaler.fit_transform(np.array(titers).reshape(-1, 1))
    infection_status = np.array(infection_status)

    # Generate polynomial terms
    polynomial_titers = generate_polynomial_terms(titers_scaled,terms)
    
    # Fit logistic regression model
    model = LogisticRegression(penalty='none', max_iter=1000)
    model.fit(polynomial_titers, infection_status)
    
    # Predict on new titers
    potential_titers_scaled = scaler.transform(np.array(potential_titers).reshape(-1, 1))
    potential_poly = generate_polynomial_terms(potential_titers_scaled,terms)
    y_pred = model.predict_proba(potential_poly)[:, 1]
    
    return y_pred