import cupy as cp
from flow_direction import load_tif_image
from plot import create_plot
import time
from make_tif import GeoTIFFHandler
import warnings

def compute_cn2(soil: cp.ndarray, lulc: cp.ndarray) -> cp.ndarray:
    """
    Compute CN2 values from soil and lulc matrices using CuPy.

    Parameters:
        soil (cp.ndarray): Soil type matrix (values 0–4).
        lulc (cp.ndarray): LULC class matrix (values 0–7).

    Returns:
        cp.ndarray: CN2 values.
    """
    # Define lookup table [soil_type][lulc_class]
    LUT = cp.array([
        [ 0,  0,  0,  0,  0,  0,  0,  0],  # soil 0 → CN2 = 0 (as fallback)
        [ 0, 30, 39,  0, 64, 39, 82, 49],  # soil 1
        [ 0, 55, 61,  0, 75, 61, 88, 69],  # soil 2
        [ 0, 70, 74,  0, 82, 74, 91, 79],  # soil 3
        [ 0, 77, 80,  0, 85, 80, 93, 84],  # soil 4
    ], dtype=cp.int32)

    # Ensure valid bounds before indexing
    soil = cp.clip(soil.astype(cp.int32), 0, 4)
    lulc = cp.clip(lulc.astype(cp.int32), 0, 7)

    # Apply lookup
    CN2 = LUT[soil, lulc]
    return CN2

def compute_cn1(CN2: cp.ndarray) -> cp.ndarray:
    """
    Compute CN1 from CN2 using the formula: CN1 = -75 * CN2 / (CN2 - 175)

    Parameters:
        CN2 (cp.ndarray): Curve Number 2 matrix (usually int or float)

    Returns:
        cp.ndarray: CN1 values as float32
    """
    CN2 = CN2.astype(cp.float32)
    denom = CN2 - 175
    # Prevent division by zero
    denom = cp.where(denom == 0, cp.finfo(cp.float32).eps, denom)
    CN1 = (-75 * CN2) / denom
    return CN1

def compute_cn3(CN2: cp.ndarray) -> cp.ndarray:
    """
    Compute CN3 from CN2 using the formula:
        CN3 = CN2 * (e ** (0.00673 * (100 - CN2)))

    Parameters:
        CN2 (cp.ndarray): CuPy array of Curve Number 2 values.

    Returns:
        cp.ndarray: CuPy array of CN3 values.
    """
    CN2 = CN2.astype(cp.float32)
    exponent = 0.00673 * (100.0 - CN2)
    CN3 = CN2 * cp.exp(exponent)
    return CN3

def compute_part1(CN3: cp.ndarray, CN2: cp.ndarray) -> cp.ndarray:
    CN3 = CN3.astype(cp.float32)
    CN2 = CN2.astype(cp.float32)
    p1 = (CN3 - CN2) / 3.0
    return p1

def compute_part2(slope: cp.ndarray) -> cp.ndarray:
    slope = slope.astype(cp.float32)
    p2 = 1.0 - 2.0 * cp.exp(-13.86 * slope)
    return p2

def compute_CN2a(part1: cp.ndarray, part2: cp.ndarray, CN2: cp.ndarray) -> cp.ndarray:
    # Ensure type consistency
    part1 = part1.astype(cp.float32)
    part2 = part2.astype(cp.float32)
    CN2 = CN2.astype(cp.float32)
    
    CN2a = part1 * part2 + CN2
    return CN2a

def compute_CN1a(CN2a: cp.ndarray) -> cp.ndarray:
    CN2a = CN2a.astype(cp.float32)
    CN1a = 4.2 * CN2a / (10 - 0.058 * CN2a)
    return CN1a

def compute_CN3a(CN2a: cp.ndarray) -> cp.ndarray:
    CN2a = CN2a.astype(cp.float32)
    CN3a = 23 * CN2a / (10 + 0.13 * CN2a)
    return CN3a

def compute_sr(CN: cp.ndarray) -> cp.ndarray:
    CN = CN.astype(cp.float32)
    
    # Mask where CN is invalid (e.g., 0 or very small)
    mask = CN <= 10
    CN = cp.where(mask, 100.0, CN)  # default value or np.nan
    
    # Clip CN to range 30–100
    CN = cp.clip(CN, 30.0, 100.0)
    
    sr = (25400.0 / CN) - 254.0
    
    # Optional: mask output back where CN was invalid
    sr = cp.where(mask, -254.0, sr)
    
    return sr




def compute_M(sr, p):
    """
    Compute M2 using CuPy, ensuring that if either sr or p is NaN, the result is NaN.
    """
    nan_mask = cp.isnan(sr) | cp.isnan(p)
    sqrt_term = cp.sqrt(cp.maximum(sr**2 + 4 * p * sr, 0.0))
    M2 = 0.5 * (-sr + sqrt_term)
    M2[nan_mask] = cp.nan  # Preserve NaN values
    return M2

def compute_M_alt(sr, p):
    # Ensure float type, potentially float64 for precision
    sr = sr.astype(cp.float64)
    p = p.astype(cp.float64)

    # Preserve input NaNs
    nan_mask_input = cp.isnan(sr) | cp.isnan(p)

    # Calculate term inside sqrt
    term = sr**2 + 4 * p * sr

    # Allow sqrt to produce NaN for negative inputs (and suppress warning)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning) # Ignore sqrt domain error warning
        sqrt_term = cp.sqrt(term) # This will be NaN where term < 0

    M_result = 0.5 * (-sr + sqrt_term)

    # Ensure input NaNs propagate, and sqrt NaNs are kept
    M_result[nan_mask_input | cp.isnan(sqrt_term)] = cp.nan

    return M_result

def sum_tif_images(images, start, end):
    """
    Sum multiple TIFF images using CuPy on the GPU.
    """
    return cp.sum(images[start:end], axis=0)

def calculate_p_and_p5(file_paths):
    """
    Load images, convert to CuPy, and compute total precipitation sums.
    """
    images = [cp.asarray(load_tif_image(fp), dtype=cp.float32) for fp in file_paths]
    total_sum = cp.sum(images, axis=0)
    mid_sum = cp.sum(images[-2:], axis=0) if len(images) >= 2 else total_sum
    return total_sum, mid_sum

def calculate_runoff(P, P5, M1, M2, M3, sr1, sr2, sr3):
    """
    Compute runoff using CuPy.
    """
    nan_mask = cp.isnan(P) | cp.isnan(P5) | cp.isnan(M1) | cp.isnan(M2) | cp.isnan(M3) | cp.isnan(sr1) | cp.isnan(sr2) | cp.isnan(sr3)
    runoff = cp.zeros_like(sr1)
    mask1 = (~nan_mask) & (P >= 0.2 * sr1) & (P5 >= 0) & (P5 <= 35)
    mask2 = (~nan_mask) & (P >= 0.2 * sr2) & (P5 > 35) & (P5 <= 52.5)
    mask3 = (~nan_mask) & (P >= 0.2 * sr3) & (P5 > 52.5)
    
    runoff[mask1] = ((P[mask1] - 0.2 * sr1[mask1]) * (P[mask1] - 0.2 * sr1[mask1] + M1[mask1])) / (P[mask1] + 0.2 * sr1[mask1] + sr1[mask1] + M1[mask1])
    runoff[mask2] = ((P[mask2] - 0.2 * sr2[mask2]) * (P[mask2] - 0.2 * sr2[mask2] + M2[mask2])) / (P[mask2] + 0.2 * sr2[mask2] + sr2[mask2] + M2[mask2])
    runoff[mask3] = ((P[mask3] - 0.2 * sr3[mask3]) * (P[mask3] - 0.2 * sr3[mask3] + M3[mask3])) / (P[mask3] + 0.2 * sr3[mask3] + sr3[mask3] + M3[mask3])
    
    runoff[nan_mask] = cp.nan  # Restore NaNs
    return runoff


def calculate_runoff_cupy(P, P5, m1, m2, m3, sr1, sr2, sr3):
    """
    Calculates runoff using a CuPy implementation mirroring a GEE expression.

    Follows the logic:
    Q = f(P, P5, sr, m) based on AMC I, II, III where P5 determines AMC.
    Uses derived m1, m2, m3 and potential max retention sr1, sr2, sr3.
    Ensures runoff >= 0 and handles NaN inputs (NaN in any input -> NaN output).

    Args:
        P (cp.ndarray): Precipitation matrix.
        P5 (cp.ndarray): 5-day antecedent precipitation matrix.
        m1 (cp.ndarray): Derived moisture parameter for AMC I.
        m2 (cp.ndarray): Derived moisture parameter for AMC II.
        m3 (cp.ndarray): Derived moisture parameter for AMC III.
        sr1 (cp.ndarray): Potential maximum retention for AMC I (S derived from CN1).
        sr2 (cp.ndarray): Potential maximum retention for AMC II (S derived from CN2).
        sr3 (cp.ndarray): Potential maximum retention for AMC III (S derived from CN3).

    Returns:
        cp.ndarray: Calculated runoff matrix, with NaN where any input was NaN.
    """
    # Optional: Check if inputs are indeed CuPy arrays (if function might receive others)
    # P = cp.asarray(P) # etc. for all inputs

    # --- 0. Input Validation (Optional but Recommended) ---
    if not P.shape == P5.shape == m1.shape == m2.shape == m3.shape == \
             sr1.shape == sr2.shape == sr3.shape:
        raise ValueError("All input CuPy arrays must have the same shape.")

    # --- 1. Handle NaN Inputs: Create combined mask ---
    # If any input pixel is NaN, the output for that pixel will be NaN.
    nan_mask = cp.isnan(P) | cp.isnan(P5) | cp.isnan(m1) | cp.isnan(m2) | cp.isnan(m3) | \
               cp.isnan(sr1) | cp.isnan(sr2) | cp.isnan(sr3)

    # --- 2. Calculate Intermediate Terms (Initial Abstraction) ---
    Ia1 = 0.2 * sr1
    Ia2 = 0.2 * sr2
    Ia3 = 0.2 * sr3

    # --- 3. Calculate Potential Runoff Values (Q1, Q2, Q3) ---
    # Suppress potential division-by-zero or invalid value warnings as we handle them
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # Calculate Denominators for the runoff formula
        den1 = P + Ia1 + sr1 + m1
        den2 = P + Ia2 + sr2 + m2
        den3 = P + Ia3 + sr3 + m3

        # Calculate Numerators for the runoff formula
        num1 = (P - Ia1) * (P - Ia1 + m1)
        num2 = (P - Ia2) * (P - Ia2 + m2)
        num3 = (P - Ia3) * (P - Ia3 + m3)

        # Calculate potential runoff Q. Use cp.where to avoid division by zero.
        # If denominator is 0, runoff is 0. Use float64 for precision.
        Q1 = cp.where(den1 != 0, num1 / den1, 0.0).astype(cp.float64)
        Q2 = cp.where(den2 != 0, num2 / den2, 0.0).astype(cp.float64)
        Q3 = cp.where(den3 != 0, num3 / den3, 0.0).astype(cp.float64)

    # --- 4. Define Conditions using Boolean Masks ---
    # Basic precipitation conditions: P must be >= Initial Abstraction (Ia)
    condP1 = (P >= Ia1)
    condP2 = (P >= Ia2)
    condP3 = (P >= Ia3)

    # Antecedent Moisture Conditions based on P5 thresholds from GEE expression
    # Note: P5>=0 check is included in the GEE expression, so we replicate it.
    condAMC1 = (P5 >= 0) & (P5 <= 35)
    condAMC2 = (P5 >= 0) & (P5 > 35)   # Corresponds to the second check in GEE ternary
    condAMC3 = (P5 >= 0) & (P5 > 52.5)  # Corresponds to the third check in GEE ternary

    # Runoff non-negativity conditions (Q must be >= 0) from GEE expression
    condQ1_pos = (Q1 >= 0)
    condQ2_pos = (Q2 >= 0)
    condQ3_pos = (Q3 >= 0)

    # Combine all conditions for each case
    # These directly represent the full condition before the '?' in the GEE expression
    cond1_full = condP1 & condAMC1 & condQ1_pos
    cond2_full = condP2 & condAMC2 & condQ2_pos
    cond3_full = condP3 & condAMC3 & condQ3_pos

    # --- 5. Apply Conditions using Nested cp.where (Mirrors GEE Ternary Logic) ---
    # This structure directly implements: cond1 ? Q1 : (cond2 ? Q2 : (cond3 ? Q3 : 0))
    final_runoff = cp.where(cond1_full, Q1,             # If Cond1 is true, use Q1
                       cp.where(cond2_full, Q2,         # Else, if Cond2 is true, use Q2
                           cp.where(cond3_full, Q3,     # Else, if Cond3 is true, use Q3
                               0.0)))                   # Else (all conditions false), use 0.0

    # --- 6. Apply NaN Mask ---
    # Ensure any pixel that had NaN in any input results in NaN output
    # Note: cp.where might already propagate NaNs correctly in many cases,
    # but applying the mask explicitly guarantees it.
    final_runoff[nan_mask] = cp.nan

    return final_runoff


def runoff_total_volume(runoff):
    nan_mask = cp.isnan(runoff)
    runoff[~nan_mask] = runoff[~nan_mask] * 900
    return runoff


# ------------------------------------ Lower Ganga Example --------------------------------------

# handler = GeoTIFFHandler('../tifs/masalia/gee_outputs/sr1_masalia_ak.tif')
# sr1 = cp.asarray(load_tif_image('../tifs/masalia/gee_outputs/sr1_masalia_23-07-07_ak.tif'), dtype=cp.float32)
# sr2 = cp.asarray(load_tif_image('../tifs/masalia/gee_outputs/sr2_masalia_23-07-07_ak.tif'), dtype=cp.float32)
# sr3 = cp.asarray(load_tif_image('../tifs/masalia/gee_outputs/sr3_masalia_23-07-07_ak.tif'), dtype=cp.float32)

# def get_p_p5(index):
    
#     rain1 = cp.asarray(load_tif_image(f'../tifs/masalia/masalia_daily_rain/day_{index - 3}.tif'), dtype=cp.float32)
#     rain2 = cp.asarray(load_tif_image(f'../tifs/masalia/masalia_daily_rain/day_{index - 2}.tif'), dtype=cp.float32)
#     rain4 = cp.asarray(load_tif_image(f'../tifs/masalia/masalia_daily_rain/day_{index - 1}.tif'), dtype=cp.float32)
#     rain3 = cp.asarray(load_tif_image(f'../tifs/masalia/masalia_daily_rain/day_{index}.tif'), dtype=cp.float32)

#     P = rain4
#     P5 = P + rain2  + rain1 + rain3
#     return P, P5


















# -------------------------- Masalia Example ----------------------------------------------
# Load static images
handler = GeoTIFFHandler('../tifs/masalia/masalia_dem/dem.tif')
soil = cp.asarray(load_tif_image('../tifs/masalia/soil_lulc/soil.tif'), dtype=cp.float32)
lulc = cp.asarray(load_tif_image('../tifs/masalia/soil_lulc/lulc.tif'), dtype=cp.float32)
slope = cp.asarray(load_tif_image('../tifs/masalia/masalia_dem/dem.tif'), dtype=cp.float32)

CN2 = compute_cn2(soil, lulc)
CN1 = compute_cn1(CN2)
CN3 = compute_cn3(CN2)

# handler.save_tiff(CN1.get(), '../tifs/masalia/local/cn1.tif')
# handler.save_tiff(CN2.get(), '../tifs/masalia/local/cn2.tif')
# handler.save_tiff(CN3.get(), '../tifs/masalia/local/cn3.tif')

p1 = compute_part1(CN3, CN2)
p2 = compute_part2(slope)

CN2a = compute_CN2a(p1, p2, CN2)
CN1a = compute_CN1a(CN2a)
CN3a = compute_CN3a(CN2a)

# handler.save_tiff(CN1a.get(), '../tifs/masalia/local/CN1a.tif')
# handler.save_tiff(CN2a.get(), '../tifs/masalia/local/CN2a.tif')
# handler.save_tiff(CN3a.get(), '../tifs/masalia/local/CN3a.tif')

sr1 = compute_sr(CN1a)
sr2 = compute_sr(CN2a)
sr3 = compute_sr(CN3a)

# handler.save_tiff(sr1.get(), '../tifs/masalia/local/sr1.tif')
# handler.save_tiff(sr2.get(), '../tifs/masalia/local/sr2.tif')
# handler.save_tiff(sr3.get(), '../tifs/masalia/local/sr3.tif')

# sr1 = cp.asarray(load_tif_image('../tifs/masalia/gee_outputs/sr1_masalia_23-07-07_ak.tif'), dtype=cp.float32)
# sr2 = cp.asarray(load_tif_image('../tifs/masalia/gee_outputs/sr2_masalia_23-07-07_ak.tif'), dtype=cp.float32)
# sr3 = cp.asarray(load_tif_image('../tifs/masalia/gee_outputs/sr3_masalia_23-07-07_ak.tif'), dtype=cp.float32)

# def get_p_p5(index):
    
#     rain1 = cp.asarray(load_tif_image(f'../tifs/masalia/masalia_daily_rain/day_{index - 3}.tif'), dtype=cp.float32)
#     rain2 = cp.asarray(load_tif_image(f'../tifs/masalia/masalia_daily_rain/day_{index - 2}.tif'), dtype=cp.float32)
#     rain4 = cp.asarray(load_tif_image(f'../tifs/masalia/masalia_daily_rain/day_{index - 1}.tif'), dtype=cp.float32)
#     rain3 = cp.asarray(load_tif_image(f'../tifs/masalia/masalia_daily_rain/day_{index}.tif'), dtype=cp.float32)

#     P = rain4
#     P5 = P + rain2  + rain1 + rain3
#     return P, P5


# def example(i):
#     P, P5 = get_p_p5(i)

#     # # Compute soil moisture
#     m1_cp = compute_M(sr1, P5)
#     m2_cp = compute_M(sr2, P5)
#     m3_cp = compute_M(sr3, P5)

#     # handler.save_tiff(cp.asnumpy(m1_cp), '../tifs/masalia/experiment_tifs/m1_local.tif')
#     # handler.save_tiff(cp.asnumpy(m2_cp), '../tifs/masalia/experiment_tifs/m2_local.tif')
#     # handler.save_tiff(cp.asnumpy(m3_cp), '../tifs/masalia/experiment_tifs/m3_local.tif')

#     # Compute final runoff
#     start_time = time.time()
#     runoff = calculate_runoff_cupy(P, P5, m1_cp, m2_cp, m3_cp, sr1, sr2, sr3)
#     # runoff = runoff_total_volume(runoff)
#     end_time = time.time()
#     print("Runoff time:", end_time - start_time)

#     row_index = 560  # Replace with your desired Row Index
#     col_index = 678 # Replace with your desired Column Index

#     print(f'row {row_index}, col {col_index} value : {runoff[row_index, col_index]}')

#     handler.save_tiff(cp.asnumpy(runoff), f'../tifs/masalia/experiment_tifs/runoff_simulation_2023_07-09/runoff_cp_{i}.tif')
#     # create_plot(cp.asnumpy(runoff),  '../tifs/masalia/experiment_pngs/runoff_cp.png', colors=['blue', 'cyan', 'green', 'yellow', 'orange', 'red'], max=30, normalize=True)


# # example(94)


# # for i in range(4, 92):
# #     # P = cp.asarray(load_tif_image(f'../tifs/masalia/gee_outputs/p_masalia_2023-09-18_ak.tif'), dtype=cp.float32)
# #     # P5 = cp.asarray(load_tif_image(f'../tifs/masalia/gee_outputs/antecedent_masalia_2023-09-18_ak.tif'), dtype=cp.float32)
# #     example(i)