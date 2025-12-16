#!/usr/bin/env python3
"""
summarize.py - Build Phase 5f-style comparison table from batch results

Reads:
  - results.jsonl (Hamiltonian batch results)
  - gue_ref.json (GUE calibration)
  - riemann_ref.json (optional, or uses hardcoded)

Outputs:
  - Console table comparing H vs GUE vs Riemann
  - summary.json with aggregated statistics
"""

import numpy as np
import json
import os

# =============================================================================
# RIEMANN REFERENCE (hardcoded from Phase 5f)
# =============================================================================

RIEMANN_REF = {
    'r': {'mean': 0.618, 'std': 0.01},
    'sigma2': {
        '2': {'mean': 0.32, 'std': 0.02},
        '5': {'mean': 0.33, 'std': 0.03},
        '10': {'mean': 0.34, 'std': 0.03},
        '15': {'mean': 0.27, 'std': 0.04},
        '20': {'mean': 0.33, 'std': 0.04},
    },
    'delta3': {
        '2': {'mean': 0.023, 'std': 0.004},
        '5': {'mean': 0.045, 'std': 0.005},
        '10': {'mean': 0.066, 'std': 0.005},
        '15': {'mean': 0.068, 'std': 0.006},
        '20': {'mean': 0.069, 'std': 0.006},
    },
}

L_STAT = ['2', '5', '10', '15', '20']

# =============================================================================
# LOAD DATA
# =============================================================================

def load_results(filename='results.jsonl'):
    """Load batch results from JSONL."""
    results = []
    if not os.path.exists(filename):
        print(f"✗ {filename} not found")
        return []
    
    with open(filename, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if data.get('status') == 'success':
                    results.append(data)
            except:
                pass
    
    return results

def load_gue_ref(filename='gue_ref.json'):
    """Load GUE reference."""
    if not os.path.exists(filename):
        print(f"✗ {filename} not found - run gue_calibrate.py first")
        return None
    
    with open(filename, 'r') as f:
        return json.load(f)

def load_riemann_ref(filename='riemann_ref.json'):
    """Load Riemann reference (or use hardcoded)."""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return RIEMANN_REF

# =============================================================================
# AGGREGATE STATISTICS
# =============================================================================

def aggregate_results(results):
    """Aggregate Hamiltonian results across all realizations."""
    if not results:
        return None
    
    r_values = [r['r_mean'] for r in results if r.get('r_mean') is not None]
    
    sigma2_by_L = {L: [] for L in L_STAT}
    delta3_by_L = {L: [] for L in L_STAT}
    
    for r in results:
        s2 = r.get('sigma2', {})
        d3 = r.get('delta3', {})
        for L in L_STAT:
            if s2.get(L) is not None and not np.isnan(s2[L]):
                sigma2_by_L[L].append(s2[L])
            if d3.get(L) is not None and not np.isnan(d3[L]):
                delta3_by_L[L].append(d3[L])
    
    agg = {
        'n_realizations': len(results),
        'r': {
            'mean': float(np.mean(r_values)) if r_values else None,
            'std': float(np.std(r_values)) if r_values else None,
        },
        'sigma2': {},
        'delta3': {},
    }
    
    for L in L_STAT:
        if sigma2_by_L[L]:
            agg['sigma2'][L] = {
                'mean': float(np.mean(sigma2_by_L[L])),
                'std': float(np.std(sigma2_by_L[L])),
            }
        if delta3_by_L[L]:
            agg['delta3'][L] = {
                'mean': float(np.mean(delta3_by_L[L])),
                'std': float(np.std(delta3_by_L[L])),
            }
    
    return agg

# =============================================================================
# COMPARISON TABLE
# =============================================================================

def print_comparison(H_agg, gue_ref, riemann_ref):
    """Print Phase 5f-style comparison table."""
    
    print("="*90)
    print("PHASE 5f-STYLE COMPARISON TABLE")
    print("="*90)
    
    # ⟨r⟩ comparison
    print(f"\n⟨r⟩ STATISTIC:")
    print(f"  Hamiltonian: {H_agg['r']['mean']:.4f} ± {H_agg['r']['std']:.4f} (n={H_agg['n_realizations']})")
    print(f"  GUE:         {gue_ref['r']['mean']:.4f} ± {gue_ref['r']['std']:.4f}")
    print(f"  Riemann:     {riemann_ref['r']['mean']:.4f} ± {riemann_ref['r']['std']:.4f}")
    
    # Distance check for r
    dist_H_GUE_r = abs(H_agg['r']['mean'] - gue_ref['r']['mean'])
    dist_H_R_r = abs(H_agg['r']['mean'] - riemann_ref['r']['mean'])
    closer_r = "GUE" if dist_H_GUE_r < dist_H_R_r else "Riemann"
    print(f"  → Closer to: {closer_r}")
    
    # Σ² table
    print(f"\nΣ² NUMBER VARIANCE:")
    print(f"{'L':<6} {'Hamiltonian':<20} {'GUE':<20} {'Riemann':<20} {'H closer to':<12}")
    print("-"*80)
    
    s2_closer_gue = 0
    s2_closer_r = 0
    
    for L in L_STAT:
        H_s2 = H_agg['sigma2'].get(L, {})
        G_s2 = gue_ref['sigma2'].get(L, {})
        R_s2 = riemann_ref['sigma2'].get(L, {})
        
        H_str = f"{H_s2.get('mean', np.nan):.4f} ± {H_s2.get('std', np.nan):.4f}" if H_s2 else "N/A"
        G_str = f"{G_s2.get('mean', np.nan):.4f} ± {G_s2.get('std', np.nan):.4f}" if G_s2 else "N/A"
        R_str = f"{R_s2.get('mean', np.nan):.4f} ± {R_s2.get('std', np.nan):.4f}" if R_s2 else "N/A"
        
        if H_s2 and G_s2 and R_s2:
            dist_G = abs(H_s2['mean'] - G_s2['mean'])
            dist_R = abs(H_s2['mean'] - R_s2['mean'])
            closer = "GUE" if dist_G < dist_R else "Riemann"
            if dist_G < dist_R:
                s2_closer_gue += 1
            else:
                s2_closer_r += 1
        else:
            closer = "N/A"
        
        print(f"{L:<6} {H_str:<20} {G_str:<20} {R_str:<20} {closer:<12}")
    
    # Δ₃ table
    print(f"\nΔ₃ SPECTRAL RIGIDITY:")
    print(f"{'L':<6} {'Hamiltonian':<20} {'GUE':<20} {'Riemann':<20} {'H closer to':<12}")
    print("-"*80)
    
    d3_closer_gue = 0
    d3_closer_r = 0
    
    for L in L_STAT:
        H_d3 = H_agg['delta3'].get(L, {})
        G_d3 = gue_ref['delta3'].get(L, {})
        R_d3 = riemann_ref['delta3'].get(L, {})
        
        H_str = f"{H_d3.get('mean', np.nan):.4f} ± {H_d3.get('std', np.nan):.4f}" if H_d3 else "N/A"
        G_str = f"{G_d3.get('mean', np.nan):.4f} ± {G_d3.get('std', np.nan):.4f}" if G_d3 else "N/A"
        R_str = f"{R_d3.get('mean', np.nan):.4f} ± {R_d3.get('std', np.nan):.4f}" if R_d3 else "N/A"
        
        if H_d3 and G_d3 and R_d3:
            dist_G = abs(H_d3['mean'] - G_d3['mean'])
            dist_R = abs(H_d3['mean'] - R_d3['mean'])
            closer = "GUE" if dist_G < dist_R else "Riemann"
            if dist_G < dist_R:
                d3_closer_gue += 1
            else:
                d3_closer_r += 1
        else:
            closer = "N/A"
        
        print(f"{L:<6} {H_str:<20} {G_str:<20} {R_str:<20} {closer:<12}")
    
    # Verdict
    print("\n" + "="*90)
    print("VERDICT")
    print("="*90)
    
    print(f"\n  Σ²: {s2_closer_gue}/{len(L_STAT)} L-values closer to GUE, {s2_closer_r}/{len(L_STAT)} closer to Riemann")
    print(f"  Δ₃: {d3_closer_gue}/{len(L_STAT)} L-values closer to GUE, {d3_closer_r}/{len(L_STAT)} closer to Riemann")
    
    if s2_closer_gue >= 3 and d3_closer_gue >= 3:
        print("""
  ✓ WORLD A CONFIRMED (Generic GUE)
  
  Both Σ² and Δ₃ predominantly closer to GUE than Riemann.
  The Hamiltonian exhibits generic quantum chaos statistics.
""")
        verdict = "WORLD_A"
    elif s2_closer_r >= 3 and d3_closer_r >= 3:
        print("""
  ⚡ WORLD B SIGNAL (Riemann-like)
  
  Both Σ² and Δ₃ predominantly closer to Riemann than GUE!
  This is worth investigating further.
""")
        verdict = "WORLD_B"
    else:
        print(f"""
  ? MIXED RESULT
  
  Σ² favors: {'GUE' if s2_closer_gue >= 3 else 'Riemann'}
  Δ₃ favors: {'GUE' if d3_closer_gue >= 3 else 'Riemann'}
  
  Further investigation needed.
""")
        verdict = "MIXED"
    
    return {
        's2_closer_gue': s2_closer_gue,
        's2_closer_r': s2_closer_r,
        'd3_closer_gue': d3_closer_gue,
        'd3_closer_r': d3_closer_r,
        'verdict': verdict,
    }

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*90)
    print("SUMMARIZE - Phase 5f Comparison from Batch Results")
    print("="*90)
    
    # Load data
    print("\nLoading data...")
    results = load_results()
    print(f"  Hamiltonian results: {len(results)}")
    
    gue_ref = load_gue_ref()
    if gue_ref:
        print(f"  GUE reference: loaded ({gue_ref.get('n_realizations', '?')} realizations)")
    
    riemann_ref = load_riemann_ref()
    print(f"  Riemann reference: loaded")
    
    if not results:
        print("\n✗ No valid results to summarize")
        return
    
    if not gue_ref:
        print("\n✗ Missing GUE reference - run gue_calibrate.py first")
        return
    
    # Aggregate
    H_agg = aggregate_results(results)
    
    # Print comparison
    verdict_info = print_comparison(H_agg, gue_ref, riemann_ref)
    
    # Save summary
    summary = {
        'hamiltonian': H_agg,
        'gue_ref': gue_ref,
        'riemann_ref': riemann_ref,
        'verdict': verdict_info,
    }
    
    with open('summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Summary saved to: summary.json")
    print("="*90)

if __name__ == "__main__":
    main()
