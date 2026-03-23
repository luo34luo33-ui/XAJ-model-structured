"""
XAJ Model Generation Module
Contains runoff generation functions for the XinAnJiang model.
"""

import numpy as np
from typing import Tuple, Optional
from config import PRECISION


def calculate_evap(
    lm: np.ndarray,
    c: np.ndarray,
    wu0: np.ndarray,
    wl0: np.ndarray,
    prcp: np.ndarray,
    pet: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three-layers evaporation model.
    
    From "Watershed Hydrologic Simulation" by Prof. RenJun Zhao.
    The three-layers evaporation model is described in Page 76.
    
    Args:
        lm: Average soil moisture storage capacity of lower layer
        c: Coefficient of deep layer
        wu0: Initial soil moisture of upper layer
        wl0: Initial soil moisture of lower layer
        prcp: Basin mean precipitation
        pet: Potential evapotranspiration
        
    Returns:
        Tuple of (eu, el, ed) - evaporation from upper, lower, deeper layers
    """
    # Upper layer evaporation
    eu = np.where(wu0 + prcp >= pet, pet, wu0 + prcp)
    
    # Deep layer evaporation
    ed = np.where(
        (wl0 < c * lm) & (wl0 < c * (pet - eu)),
        c * (pet - eu) - wl0,
        0.0
    )
    
    # Lower layer evaporation
    el = np.where(
        wu0 + prcp >= pet,
        0.0,
        np.where(
            wl0 >= c * lm,
            (pet - eu) * wl0 / lm,
            np.where(
                wl0 >= c * (pet - eu),
                c * (pet - eu),
                wl0
            )
        )
    )
    
    return eu, el, ed


def calculate_prcp_runoff(
    b: np.ndarray,
    im: np.ndarray,
    wm: np.ndarray,
    w0: np.ndarray,
    pe: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate runoff generated from rainfall.
    
    From "Watershed Hydrologic Simulation" and "Hydrologic Forecasting (5-th version)"
    
    Args:
        b: Exponent coefficient
        im: Impermeability coefficient
        wm: Average soil moisture storage capacity
        w0: Initial soil moisture
        pe: Net precipitation
        
    Returns:
        Tuple of (r, r_im) - runoff and runoff from impervious part
    """
    wmm = wm * (1.0 + b)
    a = wmm * (1.0 - (1.0 - w0 / wm) ** (1.0 / (1.0 + b)))
    
    if np.isnan(a).any():
        raise ArithmeticError("Please check if w0>wm or b is a negative value!")
    
    # Calculate runoff
    r_cal = np.where(
        pe > 0.0,
        np.where(
            pe + a < wmm,
            pe - (wm - w0) + wm * (1.0 - np.minimum(a + pe, wmm) / wmm) ** (1.0 + b),
            pe - (wm - w0)
        ),
        np.full(pe.shape, 0.0)
    )
    
    r = np.maximum(r_cal, 0.0)
    
    # Impervious part runoff
    r_im_cal = pe * im
    r_im = np.maximum(r_im_cal, 0.0)
    
    return r, r_im


def calculate_w_storage(
    um: np.ndarray,
    lm: np.ndarray,
    dm: np.ndarray,
    wu0: np.ndarray,
    wl0: np.ndarray,
    wd0: np.ndarray,
    eu: np.ndarray,
    el: np.ndarray,
    ed: np.ndarray,
    pe: np.ndarray,
    r: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Update soil moisture values of the three layers.
    
    According to equation 2.60 in "Hydrologic Forecasting"
    
    Args:
        um: Average soil moisture storage capacity of upper layer
        lm: Average soil moisture storage capacity of lower layer
        dm: Average soil moisture storage capacity of deep layer
        wu0: Initial soil moisture in upper layer
        wl0: Initial soil moisture in lower layer
        wd0: Initial soil moisture in deep layer
        eu: Evaporation of upper layer
        el: Evaporation of lower layer
        ed: Evaporation of deep layer
        pe: Net precipitation (can be negative)
        r: Runoff
        
    Returns:
        Tuple of (wu, wl, wd) - updated soil moisture
    """
    # Update upper layer moisture
    wu = np.where(
        pe > 0.0,
        np.where(wu0 + pe - r < um, wu0 + pe - r, um),
        np.where(wu0 + pe > 0.0, wu0 + pe, 0.0)
    )
    
    # Update deep layer moisture
    wd = np.where(
        pe > 0.0,
        np.where(
            wu0 + wl0 + pe - r > um + lm,
            wu0 + wl0 + wd0 + pe - r - um - lm,
            wd0
        ),
        wd0 - ed
    )
    
    # Update lower layer moisture (water balance)
    wl = np.where(
        pe > 0.0,
        wu0 + wl0 + wd0 + pe - r - wu - wd,
        wl0 - el
    )
    
    # Clip to valid ranges
    wu_ = np.clip(wu, a_min=0.0, a_max=um)
    wl_ = np.clip(wl, a_min=0.0, a_max=lm)
    wd_ = np.clip(wd, a_min=0.0, a_max=dm)
    
    return wu_, wl_, wd_


def generation(
    p_and_e: np.ndarray,
    k: np.ndarray,
    b: np.ndarray,
    im: np.ndarray,
    um: np.ndarray,
    lm: np.ndarray,
    dm: np.ndarray,
    c: np.ndarray,
    wu0: Optional[np.ndarray] = None,
    wl0: Optional[np.ndarray] = None,
    wd0: Optional[np.ndarray] = None
) -> Tuple:
    """Single-step runoff generation in XAJ.
    
    Args:
        p_and_e: Precipitation and potential evapotranspiration [basin, feature=2]
        k: Ratio of potential evapotranspiration to reference crop evaporation
        b: Exponent parameter
        im: Impermeability coefficient
        um: Average soil moisture storage capacity of upper layer
        lm: Average soil moisture storage capacity of lower layer
        dm: Average soil moisture storage capacity of deep layer
        c: Coefficient of deep layer evapotranspiration
        wu0: Initial soil moisture in upper layer
        wl0: Initial soil moisture in lower layer
        wd0: Initial soil moisture in deep layer
        
    Returns:
        Tuple of ((r, rim, e, pe), (wu, wl, wd))
    """
    # Ensure physical variables are in correct ranges
    prcp = np.maximum(p_and_e[:, 0], 0.0)
    pet = np.maximum(p_and_e[:, 1] * k, 0.0)
    
    # Total moisture capacity
    wm = um + lm + dm
    
    # Initialize soil moisture if not provided
    if wu0 is None:
        wu0 = 0.6 * um
    if wl0 is None:
        wl0 = 0.6 * lm
    if wd0 is None:
        wd0 = 0.6 * dm
    
    w0_ = wu0 + wl0 + wd0
    
    # Ensure w0 is within valid range
    w0 = np.minimum(w0_, wm - PRECISION)
    
    # Calculate evaporation
    eu, el, ed = calculate_evap(lm, c, wu0, wl0, prcp, pet)
    e = eu + el + ed
    
    # Calculate net precipitation
    prcp_difference = prcp - e
    pe = np.maximum(prcp_difference, 0.0)
    
    # Calculate runoff
    r, rim = calculate_prcp_runoff(b, im, wm, w0, pe)
    
    # Update soil moisture
    wu, wl, wd = calculate_w_storage(
        um, lm, dm, wu0, wl0, wd0, eu, el, ed, prcp_difference, r
    )
    
    return (r, rim, e, pe), (wu, wl, wd)


def run_generation_loop(
    inputs: np.ndarray,
    k: np.ndarray,
    b: np.ndarray,
    im: np.ndarray,
    um: np.ndarray,
    lm: np.ndarray,
    dm: np.ndarray,
    c: np.ndarray,
    w0: Tuple[np.ndarray, np.ndarray, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple]:
    """Run generation loop for all time steps.
    
    Args:
        inputs: Input data [time, basin, feature=2]
        k, b, im, um, lm, dm, c: Model parameters
        w0: Initial soil moisture (wu0, wl0, wd0)
        
    Returns:
        Tuple of (runoff, runoff_im, es, pe, w_final)
    """
    time_steps = inputs.shape[0]
    n_basin = inputs.shape[1]
    
    # Initialize output arrays
    runoff_ = np.full((time_steps, n_basin), 0.0)
    runoff_im_ = np.full((time_steps, n_basin), 0.0)
    es_ = np.full((time_steps, n_basin), 0.0)
    pe_ = np.full((time_steps, n_basin), 0.0)
    
    w = w0
    
    for i in range(time_steps):
        if i == 0:
            (r, rim, e, pe), w = generation(
                inputs[i, :, :], k, b, im, um, lm, dm, c, *w0
            )
        else:
            (r, rim, e, pe), w = generation(
                inputs[i, :, :], k, b, im, um, lm, dm, c, *w
            )
        
        runoff_[i, :] = r
        runoff_im_[i, :] = rim
        es_[i, :] = e
        pe_[i, :] = pe
    
    return runoff_, runoff_im_, es_, pe_, w
