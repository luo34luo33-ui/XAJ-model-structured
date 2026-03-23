"""
XAJ Model Calibration Module
Contains parameter loading, validation, and optimization functions.
"""

import json
import numpy as np
from typing import Dict, Tuple, Optional, Callable
from config import DEFAULT_PARAMS, PARAM_RANGES, validate_params, create_params_array


def load_parameters(file_path: str) -> Dict:
    """Load model parameters from JSON file.
    
    Args:
        file_path: Path to parameter JSON file
        
    Returns:
        Dictionary containing parameters
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        params = json.load(f)
    
    return params


def save_parameters(params: Dict, file_path: str):
    """Save model parameters to JSON file.
    
    Args:
        params: Dictionary containing parameters
        file_path: Output file path
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=4, ensure_ascii=False)


def get_default_parameters() -> Dict:
    """Get default parameter values.
    
    Returns:
        Dictionary containing default parameters
    """
    return DEFAULT_PARAMS.copy()


def get_parameter_ranges() -> Dict:
    """Get parameter ranges for calibration.
    
    Returns:
        Dictionary containing parameter ranges
    """
    return PARAM_RANGES.copy()


def create_parameter_template(file_path: str):
    """Create a parameter template JSON file.
    
    Args:
        file_path: Output file path
    """
    template = {
        'model_name': 'XAJ',
        'description': '新安江模型参数配置',
        'parameters': DEFAULT_PARAMS,
        'parameter_ranges': PARAM_RANGES,
        'calibration': {
            'algorithm': 'SCE-UA',
            'max_iterations': 1000,
            'objective_function': 'NSE',
            'population_size': 50,
            'n_complexes': 5
        }
    }
    
    save_parameters(template, file_path)


def prepare_parameters_for_model(params_dict: Dict) -> np.ndarray:
    """Prepare parameter array for model execution.
    
    Args:
        params_dict: Dictionary containing parameter values
        
    Returns:
        Parameter array in model-expected order
    """
    # Fill missing parameters with defaults
    full_params = DEFAULT_PARAMS.copy()
    full_params.update(params_dict)
    
    # Create array in correct order
    param_order = ['k', 'b', 'im', 'um', 'lm', 'dm', 'c', 'sm', 'ex', 'ki', 'kg', 'cs', 'l', 'ci', 'cg']
    params_array = np.array([full_params[p] for p in param_order])
    
    return params_array


def objective_function_nse(observed: np.ndarray, simulated: np.ndarray) -> float:
    """Calculate Nash-Sutcliffe Efficiency.
    
    Args:
        observed: Observed streamflow
        simulated: Simulated streamflow
        
    Returns:
        NSE value (1.0 is perfect, negative values indicate poor performance)
    """
    # Remove NaN values
    mask = ~(np.isnan(observed) | np.isnan(simulated))
    obs = observed[mask]
    sim = simulated[mask]
    
    if len(obs) == 0:
        return -np.inf
    
    obs_mean = np.mean(obs)
    numerator = np.sum((obs - sim) ** 2)
    denominator = np.sum((obs - obs_mean) ** 2)
    
    if denominator == 0:
        return -np.inf
    
    return 1 - numerator / denominator


def objective_function_rmse(observed: np.ndarray, simulated: np.ndarray) -> float:
    """Calculate Root Mean Square Error.
    
    Args:
        observed: Observed streamflow
        simulated: Simulated streamflow
        
    Returns:
        RMSE value (lower is better)
    """
    mask = ~(np.isnan(observed) | np.isnan(simulated))
    obs = observed[mask]
    sim = simulated[mask]
    
    if len(obs) == 0:
        return np.inf
    
    return np.sqrt(np.mean((obs - sim) ** 2))


def objective_function_mae(observed: np.ndarray, simulated: np.ndarray) -> float:
    """Calculate Mean Absolute Error.
    
    Args:
        observed: Observed streamflow
        simulated: simulated streamflow
        
    Returns:
        MAE value (lower is better)
    """
    mask = ~(np.isnan(observed) | np.isnan(simulated))
    obs = observed[mask]
    sim = simulated[mask]
    
    if len(obs) == 0:
        return np.inf
    
    return np.mean(np.abs(obs - sim))


def sce_ua_optimization(
    objective_func: Callable,
    param_ranges: Dict,
    max_iterations: int = 1000,
    population_size: int = 50,
    n_complexes: int = 5,
    seed: int = 42
) -> Tuple[np.ndarray, float]:
    """SCE-UA optimization algorithm (simplified version).
    
    Args:
        objective_func: Objective function to minimize
        param_ranges: Dictionary of parameter ranges
        max_iterations: Maximum number of iterations
        population_size: Population size
        n_complexes: Number of complexes
        seed: Random seed
        
    Returns:
        Tuple of (best_parameters, best_objective_value)
    """
    np.random.seed(seed)
    
    # Get parameter names and ranges
    param_names = list(param_ranges.keys())
    n_params = len(param_names)
    
    # Initialize population
    population = np.zeros((population_size, n_params))
    for i, name in enumerate(param_names):
        range_info = param_ranges[name]
        population[:, i] = np.random.uniform(range_info['min'], range_info['max'], population_size)
    
    # Evaluate initial population
    objectives = np.zeros(population_size)
    for i in range(population_size):
        params_dict = {name: population[i, j] for j, name in enumerate(param_names)}
        objectives[i] = objective_func(params_dict)
    
    # Sort by objective
    sort_idx = np.argsort(objectives)
    population = population[sort_idx]
    objectives = objectives[sort_idx]
    
    # Simple evolution (simplified SCE-UA)
    for iteration in range(max_iterations):
        # Generate new candidate
        new_params = np.zeros(n_params)
        for i, name in enumerate(param_names):
            range_info = param_ranges[name]
            new_params[i] = np.random.uniform(range_info['min'], range_info['max'])
        
        # Evaluate
        params_dict = {name: new_params[i] for i, name in enumerate(param_names)}
        new_objective = objective_func(params_dict)
        
        # Replace worst if better
        if new_objective < objectives[-1]:
            population[-1] = new_params
            objectives[-1] = new_objective
            
            # Re-sort
            sort_idx = np.argsort(objectives)
            population = population[sort_idx]
            objectives = objectives[sort_idx]
    
    # Return best
    best_idx = np.argmin(objectives)
    best_params = {name: population[best_idx, i] for i, name in enumerate(param_names)}
    
    return best_params, objectives[best_idx]


def parameter_sensitivity_analysis(
    model_func: Callable,
    base_params: Dict,
    param_ranges: Dict,
    n_samples: int = 100,
    seed: int = 42
) -> Dict:
    """Perform parameter sensitivity analysis.
    
    Args:
        model_func: Model function that takes parameters and returns output
        base_params: Base parameter values
        param_ranges: Dictionary of parameter ranges
        n_samples: Number of samples per parameter
        seed: Random seed
        
    Returns:
        Dictionary containing sensitivity results
    """
    np.random.seed(seed)
    
    sensitivity = {}
    
    for param_name, range_info in param_ranges.items():
        # Sample parameter values
        values = np.linspace(range_info['min'], range_info['max'], n_samples)
        
        # Calculate outputs
        outputs = []
        for value in values:
            test_params = base_params.copy()
            test_params[param_name] = value
            try:
                output = model_func(test_params)
                outputs.append(np.mean(output))
            except:
                outputs.append(np.nan)
        
        # Calculate sensitivity metrics
        outputs = np.array(outputs)
        valid_mask = ~np.isnan(outputs)
        
        if np.sum(valid_mask) > 2:
            sensitivity[param_name] = {
                'range': [range_info['min'], range_info['max']],
                'output_mean': np.mean(outputs[valid_mask]),
                'output_std': np.std(outputs[valid_mask]),
                'sensitivity': np.std(outputs[valid_mask]) / (range_info['max'] - range_info['min'])
            }
    
    return sensitivity
