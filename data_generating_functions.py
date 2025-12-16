import numpy as np
import pandas as pd 
import math
from scipy.stats import loguniform

# Predefined functions
def clamp(x):
    return np.clip(x, a_min=0.0, a_max=1.0)

def sigmoid(x, steepness, mid_point, saturating_point=1):
    return saturating_point / (1 + np.exp(-steepness * (x - mid_point)))

def threshold(x, thresh, max_val):
    return np.where(x < thresh, 0.0, max_val)

def exponential(x, x_intercept, steepness):
    res = 1 - np.exp(steepness * (x - x_intercept))
    return clamp(res)

def no_relationship(x):
    return np.zeros(len(x))

class RiskReductionModel:
    def __init__(self, relationship_type='sigmoid', **params):
        self.relationship_type = relationship_type
        self.params = params  
        self._validate_params()

    def _validate_params(self):
        required_params = {
            'sigmoid': ['mid_point', 'steepness', 'saturating_point'],
            'linear': ['m', 'b'],
            'threshold': ['thresh', 'max_val'],
            'exponential': ['x_intercept', 'steepness'],
            'no_relationship': []
        }
        expected = required_params.get(self.relationship_type, None)
        if expected is None:
            raise ValueError(f"Invalid relationship type: {self.relationship_type}")
        if self.params and not all(key in expected for key in self.params):
            invalid_keys = [key for key in self.params if key not in expected]
            raise ValueError(f"Unexpected parameters for '{self.relationship_type}': {invalid_keys}")

    def calculate_RR(self, log_Ab):
        if self.relationship_type == 'sigmoid':
            return sigmoid(log_Ab, self.params.get('steepness', 5), self.params.get('mid_point', 3), self.params.get('saturating_point', 1))
        elif self.relationship_type == 'threshold':
            return threshold(log_Ab, self.params.get('thresh', 6), self.params.get('max_val', 1))
        elif self.relationship_type == 'exponential':
            return exponential(log_Ab, self.params.get('x_intercept', 1), self.params.get('steepness', -.6))
        elif self.relationship_type == 'no_relationship':
            return no_relationship(log_Ab)
        else:
            raise ValueError("Invalid relationship type specified.")

def get_loguniform_Ab_titers(size: int) -> np.ndarray:
    # Generate uniform random sample in log-space of antibody titers between ~[0,18000]
    return np.log(loguniform.rvs(np.exp(1), np.exp(10), size=size))

def get_uniform_Ab_titers(size: int) -> np.ndarray:
    Abs = np.random.uniform(1, np.exp(10), size=size)
    return np.log(Abs)

def get_lognormal_Ab_titers(size: int) -> np.ndarray:
    Abs = np.random.lognormal(mean=5.5, sigma=2, size=size)
    log_Abs = np.clip(np.log(Abs),1,10)
    return log_Abs

def generate_TND_data(N, protection_function, protection_params=None, controls_per_case=4, 
                      Ab_distribution=get_loguniform_Ab_titers):
    '''
    Generate test-negative design data under the following parameters:
        N - number of individuals in the study population
        protection_function - link function connecting antibody titer to risk reduction
            options: 'threshold', 'sigmoid', 'none'
        protection_params - parameters for protection function
        controls_per_case - ratio of controls to cases in synthetic data
        Ab_distribution - distribution from which logged antibody titers are drawn
            options: get_log_uniform_Ab_titers (default), get_uniform_Ab_titers, get_lognormal_Ab_titers
    Returns samples of [x, y] where x is the sampled antibody titer and y is positive or negative (0/1)
    '''
    Ab_titers = np.array([])
    test_results = np.array([])

    # Set constants
    RR_model = RiskReductionModel(protection_function, **(protection_params or {}))
    num_desired_cases = int(N / (controls_per_case + 1))
    num_desired_controls = int(N - num_desired_cases)
    pop_to_simulate = N * 10  # Study population will be approximately 10x the study sample

    # Generate simulated population with antibody titers
    pop_Abs = Ab_distribution(pop_to_simulate)

    # Generate controls
    controls = np.random.choice(pop_Abs, num_desired_controls, replace=False)
    Ab_titers = np.concatenate((Ab_titers, controls))
    test_results = np.concatenate((test_results, np.zeros(num_desired_controls)))

    # Generate cases
    risk_reduction = RR_model.calculate_RR(pop_Abs)
    prob_sampled = (1 - risk_reduction) / np.sum(1 - risk_reduction)
    cases = np.random.choice(pop_Abs, num_desired_cases, replace=False, p=prob_sampled)
    Ab_titers = np.concatenate((Ab_titers, cases))
    test_results = np.concatenate((test_results, np.ones(num_desired_cases)))

    return [Ab_titers, test_results]

def generate_TND_data_cov(N, protection_function, protection_params=None, controls_per_case=4, 
                prop_vax = 0.7, prop_PI = 0.7, PI_boost = 0.4):
    '''
    Generate test-negative design data *with covariates* under the following parameters:
        N - number of individuals in the study population
        protection_function - link function connecting antibody titer to risk reduction
            options: 'exponential','threshold', 'sigmoid', 'none'
        protection_params - parameters for protection function
        controls_per_case - ratio of controls to cases in synthetic data
        prop_vax, prop_PI - proportion of population that is vaccinated and previously infected, respectively
        PI_boost - boost in protection from prior infection
    Returns 
        tuple of antibody titers, vaccination status, and positive/negative infection status
    '''
    # Set constants
    num_desired_cases = int(N / (controls_per_case + 1))
    num_desired_controls = int(N - num_desired_cases)
    pop_to_simulate = N * 10  # Study population will be approximately 10x the study sample
    
    prop_both = prop_vax * prop_PI # assumes vaccination and prior infection are independent
    N_vax = int(pop_to_simulate * (prop_vax - prop_both))
    N_PI = int(pop_to_simulate * (prop_PI - prop_both))
    N_both = int(pop_to_simulate * prop_both)
    N_none = int(pop_to_simulate - (N_vax + N_PI + N_both))

    # Generate simulated population with antibody titers
    Abs_both = get_loguniform_Ab_titers_high(N_both)
    Abs_vax = get_loguniform_Ab_titers_moderate(N_vax)
    Abs_PI = get_loguniform_Ab_titers_moderate(N_PI)
    Abs_none = np.zeros(N_none)
    pop_Abs = np.concatenate((Abs_both,Abs_vax,Abs_PI,Abs_none))

    # Estimate risk reduction from antibody titers
    RR_model = RiskReductionModel(protection_function, **(protection_params or {}))
    risk_reduction_Ab = RR_model.calculate_RR(pop_Abs)

    # Track vaccination status, prior infection status, and associated risk reduction
    pop_vax_status = np.concatenate((np.ones(N_both+N_vax),np.zeros(N_PI+N_none)))
    pop_PI_status = np.concatenate((np.ones(N_both), np.zeros(N_vax),np.ones(N_PI),np.zeros(N_none)))
    risk_reduction_PI = PI_boost * pop_PI_status

    # Overall probability of infection
    inf_prob = (1 - risk_reduction_Ab) * (1 - risk_reduction_PI)
    # Normalize to sampling probabilities for cases
    prob_cases = inf_prob / np.sum(inf_prob)

    # Sample cases from whole population
    case_indices = np.random.choice(np.arange(pop_to_simulate), 
                                    size=num_desired_cases, 
                                    replace=False, 
                                    p=prob_cases)
    
    # Generate controls 
    control_indices = np.random.choice(np.arange(pop_to_simulate), 
                                       size=num_desired_controls, 
                                       replace=False)

    # Collect and return sampled data
    sampled_Abs = np.concatenate([pop_Abs[control_indices], pop_Abs[case_indices]])
    sampled_y = np.concatenate([np.zeros(len(control_indices)), np.ones(len(case_indices))])
    sampled_vax_status = np.concatenate([pop_vax_status[control_indices], pop_vax_status[case_indices]])
    
    return sampled_Abs, sampled_vax_status, sampled_y