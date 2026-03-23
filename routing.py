"""
XAJ Model Routing Module
Contains runoff routing functions for the XinAnJiang model.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.special import gamma
from config import PRECISION


def sources(
    pe: np.ndarray,
    r: np.ndarray,
    sm: np.ndarray,
    ex: np.ndarray,
    ki: np.ndarray,
    kg: np.ndarray,
    s0: Optional[np.ndarray] = None,
    fr0: Optional[np.ndarray] = None
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Divide runoff to different sources.
    
    From "Hydrologic Forecasting" (HF) Page 148-149.
    
    Args:
        pe: Net precipitation
        r: Runoff from generation
        sm: Areal mean free water capacity of surface layer
        ex: Exponent of free water capacity curve
        ki: Outflow coefficient to interflow
        kg: Outflow coefficient to groundwater
        s0: Initial free water capacity
        fr0: Initial runoff area
        
    Returns:
        Tuple of ((rs, ri, rg), (s1, fr))
        rs: surface runoff
        ri: interflow runoff
        rg: groundwater runoff
        s1: final free water capacity
        fr: final runoff area
    """
    # Maximum free water storage capacity
    ms = sm * (1.0 + ex)
    
    if fr0 is None:
        fr0 = 0.1
    if s0 is None:
        s0 = 0.5 * sm
    
    # Handle fr0=0 case
    fr = np.copy(fr0)
    if any(fr == 0.0):
        raise ArithmeticError("fr==0.0 will cause error in the next step!")
    
    # Calculate current runoff area
    fr_mask = r > 0.0
    fr[fr_mask] = r[fr_mask] / pe[fr_mask]
    
    if np.isnan(fr).any():
        raise ArithmeticError("Please check pe's data! there may be 0.0")
    
    # Transfer s0 to current period
    ss = np.copy(s0)
    s = np.copy(s0)
    ss[fr_mask] = fr0[fr_mask] * s0[fr_mask] / fr[fr_mask]
    
    # Calculate surface runoff
    ss = np.minimum(ss, sm)
    au = ms * (1.0 - (1.0 - ss / sm) ** (1.0 / (1.0 + ex)))
    
    if np.isnan(au).any():
        raise ValueError("Error: NaN values detected. Try set clip function or check your data!!!")
    
    # Surface runoff calculation
    rs = np.full(r.shape, 0.0)
    rs[fr_mask] = np.where(
        pe[fr_mask] + au[fr_mask] < ms[fr_mask],
        fr[fr_mask] * (
            pe[fr_mask] - sm[fr_mask] + ss[fr_mask] + sm[fr_mask] * (
                (1 - np.minimum(pe[fr_mask] + au[fr_mask], ms[fr_mask]) / ms[fr_mask])
                ** (1 + ex[fr_mask])
            )
        ),
        fr[fr_mask] * (pe[fr_mask] + ss[fr_mask] - sm[fr_mask])
    )
    rs = np.minimum(rs, r)
    
    # Update free water storage
    s[fr_mask] = ss[fr_mask] + (r[fr_mask] - rs[fr_mask]) / fr[fr_mask]
    s = np.minimum(s, sm)
    
    if np.isnan(s).any():
        raise ArithmeticError("Please check fr's data! there may be 0.0")
    
    # Calculate interflow and groundwater
    ri = ki * s * fr
    rg = kg * s * fr
    
    # Final free water storage
    s1 = s * (1 - ki - kg)
    
    return (rs, ri, rg), (s1, fr)


def linear_reservoir(
    x: np.ndarray,
    weight: np.ndarray,
    last_y: Optional[np.ndarray] = None
) -> np.ndarray:
    """Linear reservoir release function.
    
    Args:
        x: Input to linear reservoir
        weight: Coefficient of linear reservoir
        last_y: Output of last period
        
    Returns:
        One-step forward result
    """
    weight1 = 1 - weight
    
    if last_y is None:
        last_y = np.full(weight.shape, 0.001)
    
    return weight * last_y + weight1 * x


def uh_gamma(
    a: np.ndarray,
    theta: np.ndarray,
    len_uh: int = 15
) -> np.ndarray:
    """Two-parameter Gamma distribution as unit hydrograph.
    
    From mizuRoute -- http://www.geosci-model-dev.net/9/2223/2016/
    
    Args:
        a: Shape parameter
        theta: Timescale parameter
        len_uh: Time length of unit hydrograph
        
    Returns:
        Unit hydrograph array
    """
    m = a.shape
    
    if len_uh > m[0]:
        raise RuntimeError("length of unit hydrograph should be smaller than the whole length of input")
    
    # Ensure positive values
    aa = np.maximum(0.0, a[0:len_uh, :, :]) + 0.1
    theta = np.maximum(0.0, theta[0:len_uh, :, :]) + 0.5
    
    # Create time array
    t = np.expand_dims(
        np.swapaxes(np.tile(np.arange(0.5, len_uh * 1.0), (m[1], 1)), 0, 1),
        axis=-1
    )
    
    # Calculate unit hydrograph
    denominator = gamma(aa) * (theta ** aa)
    w = 1 / denominator * (t ** (aa - 1)) * (np.exp(-t / theta))
    w = w / w.sum(0)  # Scale to 1
    
    return w


def uh_conv(uh: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Unit hydrograph convolution.
    
    Args:
        uh: Unit hydrograph
        q: Input runoff
        
    Returns:
        Convolved output
    """
    # Simple convolution implementation
    n_time = q.shape[0]
    n_basin = q.shape[1]
    output = np.zeros_like(q)
    
    for i in range(n_time):
        for j in range(min(i + 1, uh.shape[0])):
            output[i, :] += uh[j, :, 0] * q[i - j, :] if i - j >= 0 else 0
    
    return output


def run_routing(
    rss: np.ndarray,
    ris: np.ndarray,
    rgs: np.ndarray,
    runoff_im: np.ndarray,
    cs: np.ndarray,
    l: np.ndarray,
    ci: np.ndarray,
    cg: np.ndarray,
    qi0: Optional[np.ndarray] = None,
    qg0: Optional[np.ndarray] = None
) -> np.ndarray:
    """Run routing for all time steps using CSL method.
    
    Args:
        rss: Surface runoff [time, basin]
        ris: Interflow runoff [time, basin]
        rgs: Groundwater runoff [time, basin]
        runoff_im: Impervious runoff [time, basin]
        cs: Channel system recession constant
        l: Lag time
        ci: Interflow recession coefficient
        cg: Groundwater recession coefficient
        qi0: Initial interflow
        qg0: Initial groundwater
        
    Returns:
        Simulated discharge [time, basin]
    """
    time_steps = rss.shape[0]
    n_basin = rss.shape[1]
    
    # Initialize
    qt = np.full((time_steps, n_basin), 0.0)
    qs = np.full((time_steps, n_basin), 0.0)
    
    if qi0 is None:
        qi0 = np.full(ci.shape, 0.1)
    if qg0 is None:
        qg0 = np.full(cg.shape, 0.1)
    
    # Calculate total inflow
    for i in range(time_steps):
        if i == 0:
            qi = linear_reservoir(ris[i], ci, qi0)
            qg = linear_reservoir(rgs[i], cg, qg0)
        else:
            qi = linear_reservoir(ris[i], ci, qi)
            qg = linear_reservoir(rgs[i], cg, qg)
        
        qs_ = rss[i] + runoff_im[i]
        qt[i, :] = qs_ + qi + qg
    
    # Apply lag and channel routing
    for j in range(len(l)):
        lag = int(l[j])
        effective_lag = min(lag, time_steps - 1)
        
        for i in range(effective_lag):
            qs[i, j] = qt[i, j]
        
        for i in range(effective_lag, time_steps):
            qs[i, j] = cs[j] * qs[i - 1, j] + (1 - cs[j]) * qt[i - effective_lag, j]
    
    return qs


def run_routing_mz(
    rss: np.ndarray,
    runoff_im: np.ndarray,
    ris: np.ndarray,
    rgs: np.ndarray,
    a: np.ndarray,
    theta: np.ndarray,
    ci: np.ndarray,
    cg: np.ndarray,
    kernel_size: int = 15,
    qi0: Optional[np.ndarray] = None,
    qg0: Optional[np.ndarray] = None
) -> np.ndarray:
    """Run routing using mizuRoute method.
    
    Args:
        rss: Surface runoff [time, basin]
        runoff_im: Impervious runoff [time, basin]
        ris: Interflow runoff [time, basin]
        rgs: Groundwater runoff [time, basin]
        a: Shape parameter for unit hydrograph
        theta: Timescale parameter for unit hydrograph
        ci: Interflow recession coefficient
        cg: Groundwater recession coefficient
        kernel_size: Size of unit hydrograph kernel
        qi0: Initial interflow
        qg0: Initial groundwater
        
    Returns:
        Simulated discharge [time, basin]
    """
    time_steps = rss.shape[0]
    n_basin = rss.shape[1]
    
    # Initialize
    qs = np.full((time_steps, n_basin), 0.0)
    
    if qi0 is None:
        qi0 = np.full(ci.shape, 0.1)
    if qg0 is None:
        qg0 = np.full(cg.shape, 0.1)
    
    # Prepare unit hydrograph
    rout_a = a.repeat(rss.shape[0]).reshape(rss.shape)
    rout_b = theta.repeat(rss.shape[0]).reshape(rss.shape)
    
    effective_kernel_size = min(kernel_size, time_steps)
    conv_uh = uh_gamma(rout_a, rout_b, effective_kernel_size)
    
    # Convolve surface runoff
    qs_ = uh_conv(runoff_im + rss, conv_uh)
    
    # Add interflow and groundwater
    for i in range(time_steps):
        if i == 0:
            qi = linear_reservoir(ris[i], ci, qi0)
            qg = linear_reservoir(rgs[i], cg, qg0)
        else:
            qi = linear_reservoir(ris[i], ci, qi)
            qg = linear_reservoir(rgs[i], cg, qg)
        
        qs[i, :] = qs_[i, :, 0] + qi + qg
    
    return qs
