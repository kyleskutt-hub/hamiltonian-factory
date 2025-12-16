#!/usr/bin/env python3
"""
gue_calibrate.py - Generate GUE reference using SAME pipeline

Creates gue_ref.json with:
  - ⟨r⟩ mean and std across 10 realizations
  - Σ²(L) mean and std for L=[2,5,10,15,20]
  - Δ₃(L) mean and std for L=[2,5,10,15,20]

This ensures apples-to-apples comparison with Hamiltonian results.
"""

import numpy as np
import json
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

N_REALIZATIONS = 10
N_MATRIX = 800  # GUE matrix size
N_BULK = 480    # Central eigenvalues to use (matches Hamiltonian)
L_STAT = [2, 5, 10, 15, 20]

OUTPUT_FILE = "gue_ref.json"

# =============================================================================
# SPECTRAL STATISTICS (EXACT SAME as run_batch.py)
# =============================================================================

def compute_r_statistic(eigenvalues):
    E = np.sort(np.real(eigenvalues))
    spacings = np.diff(E)
    spacings = spacings[spacings > 1e-15]
    if len(spacings) < 3:
        return np.nan, np.nan
    r_vals = np.minimum(spacings[:-1], spacings[1:]) / np.maximum(spacings[:-1], spacings[1:])
    return float(np.mean(r_vals)), float(np.std(r_vals) / np.sqrt(len(r_vals)))

def unfold_polynomial(eigenvalues, degree=5):
    E = np.sort(np.real(eigenvalues))
    N = len(E)
    staircase = np.arange(1, N + 1)
    coeffs = np.polyfit(E, staircase, degree)
    return np.polyval(coeffs, E)

def number_variance(unfolded, L, n_samples=200):
    if L >= (unfolded[-1] - unfolded[0]) / 2:
        return np.nan
    n_s = min(n_samples, len(unfolded) - int(L) - 1)
    if n_s < 20:
        return np.nan
    counts = []
    starts = np.linspace(unfolded[0], unfolded[-1] - L, n_s)
    for start in starts:
        count = np.sum((unfolded >= start) & (unfolded < start + L))
        counts.append(count)
    return float(np.var(counts))

def spectral_rigidity(unfolded, L, n_samples=150):
    if L >= (unfolded[-1] - unfolded[0]) / 2:
        return np.nan
    n_s = min(n_samples, len(unfolded) - int(L) - 1)
    if n_s < 20:
        return np.nan
    d3_samples = []
    starts = np.linspace(unfolded[0], unfolded[-1] - L, n_s)
    for start in starts:
        mask = (unfolded >= start) & (unfolded < start + L)
        x = unfolded[mask] - start
        n = len(x)
        if n < 3:
            continue
        y = np.arange(1, n + 1)
        x_mean, y_mean = np.mean(x), np.mean(y)
        Sxx = np.sum((x - x_mean)**2)
        if Sxx < 1e-10:
            continue
        a = np.sum((x - x_mean) * (y - y_mean)) / Sxx
        b = y_mean - a * x_mean
        SS_res = np.sum((y - a*x - b)**2)
        d3_samples.append(SS_res / n)
    return float(np.mean(d3_samples)) if d3_samples else np.nan

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("GUE CALIBRATION - Generate Reference Statistics")
    print("="*70)
    print(f"\nRealizations: {N_REALIZATIONS}")
    print(f"Matrix size: {N_MATRIX}×{N_MATRIX}")
    print(f"Bulk eigenvalues: {N_BULK}")
    print(f"L values: {L_STAT}")
    
    # Storage
    r_values = []
    sigma2_values = {str(L): [] for L in L_STAT}
    delta3_values = {str(L): [] for L in L_STAT}
    spacing_means = []
    spacing_stds = []
    
    print("\n" + "-"*70)
    
    for i in range(N_REALIZATIONS):
        print(f"Realization {i+1}/{N_REALIZATIONS}...", end=" ", flush=True)
        
        # Generate GUE matrix
        np.random.seed(i + 100)  # Reproducible
        A = np.random.randn(N_MATRIX, N_MATRIX) + 1j * np.random.randn(N_MATRIX, N_MATRIX)
        H = (A + A.conj().T) / 2
        
        # Eigenvalues
        eigs = np.linalg.eigvalsh(H)
        eigs = np.sort(eigs)
        
        # Central bulk
        start_idx = (len(eigs) - N_BULK) // 2
        eigs_bulk = eigs[start_idx:start_idx + N_BULK]
        
        # ⟨r⟩
        r_mean, _ = compute_r_statistic(eigs_bulk)
        r_values.append(r_mean)
        
        # Unfold
        unfolded = unfold_polynomial(eigs_bulk, degree=5)
        
        # Spacing sanity
        spacings = np.diff(unfolded)
        spacing_means.append(float(np.mean(spacings)))
        spacing_stds.append(float(np.std(spacings)))
        
        # Σ² and Δ₃
        for L in L_STAT:
            s2 = number_variance(unfolded, L)
            d3 = spectral_rigidity(unfolded, L)
            sigma2_values[str(L)].append(s2)
            delta3_values[str(L)].append(d3)
        
        print(f"⟨r⟩={r_mean:.4f}")
    
    # Compute statistics
    print("\n" + "="*70)
    print("GUE REFERENCE VALUES")
    print("="*70)
    
    result = {
        'timestamp': datetime.now().isoformat(),
        'n_realizations': N_REALIZATIONS,
        'n_matrix': N_MATRIX,
        'n_bulk': N_BULK,
        'r': {
            'mean': float(np.mean(r_values)),
            'std': float(np.std(r_values)),
        },
        'spacing': {
            'mean': float(np.mean(spacing_means)),
            'std': float(np.mean(spacing_stds)),
        },
        'sigma2': {},
        'delta3': {},
    }
    
    print(f"\n⟨r⟩ = {result['r']['mean']:.4f} ± {result['r']['std']:.4f}")
    print(f"Mean spacing = {result['spacing']['mean']:.4f} ± {result['spacing']['std']:.4f}")
    
    print(f"\n{'L':<6} {'Σ² mean':<12} {'Σ² std':<12} {'Δ₃ mean':<12} {'Δ₃ std':<12}")
    print("-"*55)
    
    for L in L_STAT:
        s2_mean = float(np.nanmean(sigma2_values[str(L)]))
        s2_std = float(np.nanstd(sigma2_values[str(L)]))
        d3_mean = float(np.nanmean(delta3_values[str(L)]))
        d3_std = float(np.nanstd(delta3_values[str(L)]))
        
        result['sigma2'][str(L)] = {'mean': s2_mean, 'std': s2_std}
        result['delta3'][str(L)] = {'mean': d3_mean, 'std': d3_std}
        
        print(f"{L:<6} {s2_mean:<12.4f} {s2_std:<12.4f} {d3_mean:<12.4f} {d3_std:<12.4f}")
    
    # Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✓ Saved to: {OUTPUT_FILE}")
    print("="*70)

if __name__ == "__main__":
    main()
