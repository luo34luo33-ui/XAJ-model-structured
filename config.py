"""
XAJ Model Configuration Module
Defines model parameters, default values, and parameter ranges.
"""

import numpy as np

# Model precision
PRECISION = 1e-5

# Default parameter values for XAJ model
DEFAULT_PARAMS = {
    # Evapotranspiration parameters
    'k': 0.8,      # Ratio of potential evapotranspiration to reference crop evaporation
    'b': 0.3,      # Exponent parameter for tension water capacity curve
    'im': 0.01,    # Impermeability coefficient
    'um': 20.0,    # Average soil moisture storage capacity of upper layer (mm)
    'lm': 70.0,    # Average soil moisture storage capacity of lower layer (mm)
    'dm': 60.0,    # Average soil moisture storage capacity of deep layer (mm)
    'c': 0.15,     # Coefficient of deep layer evapotranspiration
    
    # Runoff generation parameters
    'sm': 20.0,    # Areal mean free water capacity of surface layer (mm)
    'ex': 1.5,     # Exponent of free water capacity curve
    'ki': 0.3,     # Outflow coefficient to interflow
    'kg': 0.4,     # Outflow coefficient to groundwater
    
    # Routing parameters
    'cs': 0.8,     # Channel system recession constant
    'l': 1,        # Lag time (time steps)
    'ci': 0.8,     # Interflow recession coefficient
    'cg': 0.98,    # Groundwater recession coefficient
}

# Parameter ranges for calibration
PARAM_RANGES = {
    'k': {'min': 0.5, 'max': 1.5, 'unit': '-'},
    'b': {'min': 0.1, 'max': 0.5, 'unit': '-'},
    'im': {'min': 0.0, 'max': 0.1, 'unit': '-'},
    'um': {'min': 10.0, 'max': 30.0, 'unit': 'mm'},
    'lm': {'min': 50.0, 'max': 100.0, 'unit': 'mm'},
    'dm': {'min': 40.0, 'max': 80.0, 'unit': 'mm'},
    'c': {'min': 0.1, 'max': 0.3, 'unit': '-'},
    'sm': {'min': 10.0, 'max': 50.0, 'unit': 'mm'},
    'ex': {'min': 1.0, 'max': 2.0, 'unit': '-'},
    'ki': {'min': 0.1, 'max': 0.5, 'unit': '-'},
    'kg': {'min': 0.2, 'max': 0.6, 'unit': '-'},
    'cs': {'min': 0.5, 'max': 0.95, 'unit': '-'},
    'l': {'min': 0, 'max': 10, 'unit': 'time steps'},
    'ci': {'min': 0.5, 'max': 0.95, 'unit': '-'},
    'cg': {'min': 0.9, 'max': 0.999, 'unit': '-'},
}

# Parameter descriptions
PARAM_DESCRIPTIONS = {
    'k': '蒸散发系数：潜在蒸散发与参考作物蒸发的比值',
    'b': '蓄水容量曲线指数：反映流域蓄水容量分布不均匀性',
    'im': '不透水面积比例：流域内不透水面积占总面积的比例',
    'um': '上层土壤蓄水容量：上层张力水最大蓄水容量(mm)',
    'lm': '下层土壤蓄水容量：下层张力水最大蓄水容量(mm)',
    'dm': '深层土壤蓄水容量：深层张力水最大蓄水容量(mm)',
    'c': '深层蒸散发系数：深层蒸散发与下层蒸散发的比值',
    'sm': '自由水蓄水容量：表层自由水平均蓄水容量(mm)',
    'ex': '自由水容量曲线指数：反映自由水容量分布不均匀性',
    'ki': '壤中流出流系数：自由水蓄量向壤中流的出流比例',
    'kg': '地下水出流系数：自由水蓄量向地下水的出流比例',
    'cs': '河网蓄水常数：河网汇流的蓄水常数',
    'l': '滞时：河网汇流的滞时(时间步长数)',
    'ci': '壤中流消退系数：壤中流的消退系数',
    'cg': '地下水消退系数：地下水的消退系数',
}

# Model configuration
MODEL_CONFIG = {
    'name': 'XAJ',
    'version': '2.0',
    'description': '新安江模型（结构化模块版本）',
    'time_interval_hours': 24,  # Default time interval in hours
    'warmup_length': 365,       # Default warmup period length
}

# Validation rules
VALIDATION_RULES = {
    'ki_plus_kg_max': 1.0 - PRECISION,  # ki + kg must be less than 1
    'wu_max_ratio': 1.0,                 # wu should not exceed um
    'wl_max_ratio': 1.0,                 # wl should not exceed lm
    'wd_max_ratio': 1.0,                 # wd should not exceed dm
}


def get_param_value(param_name, params_dict=None):
    """Get parameter value from dictionary or default."""
    if params_dict and param_name in params_dict:
        return params_dict[param_name]
    return DEFAULT_PARAMS.get(param_name)


def get_param_range(param_name):
    """Get parameter range."""
    return PARAM_RANGES.get(param_name, {'min': 0, 'max': 100})


def validate_params(params):
    """Validate parameter values."""
    errors = []
    
    for param_name, value in params.items():
        if param_name in PARAM_RANGES:
            range_info = PARAM_RANGES[param_name]
            if value < range_info['min'] or value > range_info['max']:
                errors.append(f"参数 {param_name} 的值 {value} 超出范围 [{range_info['min']}, {range_info['max']}]")
    
    # Check ki + kg constraint
    if 'ki' in params and 'kg' in params:
        if params['ki'] + params['kg'] >= VALIDATION_RULES['ki_plus_kg_max']:
            errors.append(f"ki + kg = {params['ki'] + params['kg']} 必须小于 {VALIDATION_RULES['ki_plus_kg_max']}")
    
    return errors


def create_params_array(params_dict):
    """Create parameter array in the order expected by the model."""
    param_order = ['k', 'b', 'im', 'um', 'lm', 'dm', 'c', 'sm', 'ex', 'ki', 'kg', 'cs', 'l', 'ci', 'cg']
    return np.array([params_dict.get(p, DEFAULT_PARAMS[p]) for p in param_order])
