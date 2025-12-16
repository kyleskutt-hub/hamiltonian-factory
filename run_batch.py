#!/usr/bin/env python3
"""
run_batch.py - Cloud batch runner with robust checkpointing

Grid:
  σ ∈ {18, 20, 22, 24, 26}
  twists = 8 random (θx, θy, θz)
  Total: 5 × 8 = 40 realizations

Each realization:
  - k=600 eigenvalues, bulk=480
  - Computes ⟨r⟩, Σ²(L), Δ₃(L) for L=[2,5,10,15,20]
  - Appends JSON line immediately (crash-safe)

Resume support:
  - If results.jsonl exists, skips completed (σ, twist_id) pairs
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
import json
import time
import os
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

# Grid parameters
SIGMA_VALUES = [18, 20, 22, 24, 26]
N_TWISTS = 8
L_LATTICE = 25
PHI = 0.28
ALPHA = 8.0
K = 600
L_STAT = [2, 5, 10, 15, 20]

# Output file
RESULTS_FILE = "results.jsonl"

# Random seed for reproducible twists
TWIST_SEED = 42

# =============================================================================
# HAMILTONIAN BUILDER (Phase 4 exact)
# =============================================================================

def idx3(x, y, z, L):
    return x + y * L + z * L * L

def mobius_sieve(n):
    mu = np.ones(n + 1, dtype=int)
    mu[0] = 0
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[:2] = False
    for p in range(2, n + 1):
        if is_prime[p]:
            mu[p::p] *= -1
            p2 = p * p
            if p2 <= n:
                mu[p2::p2] = 0
            is_prime[2*p::p] = False
    return mu

def build_hamiltonian(L, Phi, alpha, twist):
    """Build Phase 4 Hamiltonian with boundary twist."""
    theta_x, theta_y, theta_z = twist
    N = L ** 3
    rows, cols, data = [], [], []
    
    for z in range(L):
        for y in range(L):
            for x in range(L):
                i = idx3(x, y, z, L)
                
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            if dx == 0 and dy == 0 and dz == 0:
                                continue
                            if (dx, dy, dz) <= (0, 0, 0):
                                continue
                            
                            nx = (x + dx) % L
                            ny = (y + dy) % L
                            nz = (z + dz) % L
                            j = idx3(nx, ny, nz, L)
                            
                            # Landau gauge phase
                            if dy != 0 and dx == 0 and dz == 0:
                                phase = Phi * x * dy
                            elif dy != 0:
                                phase = Phi * x * dy * 0.5
                            else:
                                phase = 0.0
                            
                            # Boundary twist
                            if (x + dx) < 0 or (x + dx) >= L:
                                phase += theta_x * np.sign(dx)
                            if (y + dy) < 0 or (y + dy) >= L:
                                phase += theta_y * np.sign(dy)
                            if (z + dz) < 0 or (z + dz) >= L:
                                phase += theta_z * np.sign(dz)
                            
                            hop = np.exp(1j * phase)
                            rows += [i, j]
                            cols += [j, i]
                            data += [hop, np.conj(hop)]
    
    A = sparse.csr_matrix((np.array(data, dtype=np.complex128), (rows, cols)),
                          shape=(N, N), dtype=np.complex128)
    deg = np.array(np.abs(A).sum(axis=1)).ravel()
    L_mat = sparse.diags(deg, dtype=np.complex128) - A
    
    mu = mobius_sieve(N)
    V = sparse.diags([alpha * mu[n + 1] for n in range(N)], dtype=np.complex128)
    
    return L_mat + V

# =============================================================================
# SPECTRAL STATISTICS (same as Phase 5f)
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
# CHECKPOINTING
# =============================================================================

def load_completed():
    """Load set of completed (sigma, twist_id) pairs."""
    completed = set()
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    key = (data['sigma'], data['twist_id'])
                    completed.add(key)
                except:
                    pass
    return completed

def append_result(result):
    """Append single result as JSON line."""
    with open(RESULTS_FILE, 'a') as f:
        f.write(json.dumps(result) + '\n')

# =============================================================================
# MAIN
# =============================================================================

def run_single(sigma, twist_id, twist):
    """Run a single (sigma, twist) configuration."""
    result = {
        'sigma': sigma,
        'twist_id': twist_id,
        'twist': [float(t) for t in twist],
        'L': L_LATTICE,
        'Phi': PHI,
        'alpha': ALPHA,
        'k': K,
        'timestamp': datetime.now().isoformat(),
    }
    
    try:
        # Build Hamiltonian
        t0 = time.time()
        H = build_hamiltonian(L_LATTICE, PHI, ALPHA, twist)
        result['build_time'] = time.time() - t0
        
        # Solve eigenvalues
        t0 = time.time()
        eigs = eigsh(H, k=min(K, H.shape[0]-2), sigma=sigma, which='LM',
                    return_eigenvectors=False, maxiter=5000)
        result['solve_time'] = time.time() - t0
        
        eigs = np.sort(np.real(eigs))
        
        # Bulk selection (trim 10% from each end)
        n_trim = int(len(eigs) * 0.1)
        eigs_bulk = eigs[n_trim:-n_trim] if n_trim > 0 else eigs
        result['n_bulk'] = len(eigs_bulk)
        
        # ⟨r⟩ statistic
        r_mean, r_err = compute_r_statistic(eigs_bulk)
        result['r_mean'] = r_mean
        result['r_err'] = r_err
        
        # Unfold
        unfolded = unfold_polynomial(eigs_bulk, degree=5)
        
        # Unfolding sanity check
        spacings = np.diff(unfolded)
        result['spacing_mean'] = float(np.mean(spacings))
        result['spacing_std'] = float(np.std(spacings))
        
        # Σ² and Δ₃ for each L
        sigma2_dict = {}
        delta3_dict = {}
        for L in L_STAT:
            sigma2_dict[str(L)] = number_variance(unfolded, L)
            delta3_dict[str(L)] = spectral_rigidity(unfolded, L)
        
        result['sigma2'] = sigma2_dict
        result['delta3'] = delta3_dict
        result['status'] = 'success'
        
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
    
    return result

def main():
    print("="*70)
    print("BATCH RUNNER - Phase 5 Replication Study")
    print("="*70)
    print(f"\nGrid: σ ∈ {SIGMA_VALUES}, {N_TWISTS} twists")
    print(f"Total configurations: {len(SIGMA_VALUES) * N_TWISTS}")
    print(f"Output: {RESULTS_FILE}")
    
    # Generate reproducible twists
    np.random.seed(TWIST_SEED)
    twists = [(np.random.uniform(0, 2*np.pi),
               np.random.uniform(0, 2*np.pi),
               np.random.uniform(0, 2*np.pi)) for _ in range(N_TWISTS)]
    
    # Load completed
    completed = load_completed()
    print(f"\nAlready completed: {len(completed)}")
    
    # Build task list
    tasks = []
    for sigma in SIGMA_VALUES:
        for twist_id in range(N_TWISTS):
            if (sigma, twist_id) not in completed:
                tasks.append((sigma, twist_id, twists[twist_id]))
    
    print(f"Remaining tasks: {len(tasks)}")
    
    if not tasks:
        print("\n✓ All tasks complete!")
        return
    
    # Run tasks
    print("\n" + "-"*70)
    start_time = time.time()
    
    for i, (sigma, twist_id, twist) in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] σ={sigma}, twist={twist_id}")
        
        result = run_single(sigma, twist_id, twist)
        append_result(result)
        
        if result['status'] == 'success':
            print(f"  ⟨r⟩={result['r_mean']:.4f}, "
                  f"Σ²(10)={result['sigma2']['10']:.4f}, "
                  f"Δ₃(10)={result['delta3']['10']:.4f}")
            print(f"  Build: {result['build_time']:.1f}s, Solve: {result['solve_time']:.1f}s")
        else:
            print(f"  ✗ Error: {result.get('error', 'unknown')}")
        
        # ETA
        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed
        eta = (len(tasks) - i - 1) / rate if rate > 0 else 0
        print(f"  Elapsed: {elapsed/60:.1f}m, ETA: {eta/60:.1f}m")
    
    total_time = time.time() - start_time
    print("\n" + "="*70)
    print(f"COMPLETE! Total time: {total_time/60:.1f} minutes")
    print(f"Results saved to: {RESULTS_FILE}")
    print("="*70)

if __name__ == "__main__":
    main()
