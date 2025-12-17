# Hamiltonian Factory

Spectral evaluation environment for testing quantum Hamiltonians via statistical signatures.

## Overview

This project builds parameterized quantum Hamiltonians and analyzes their eigenvalue spectra to detect signatures of quantum chaos vs. arithmetic structure (Riemann zeta zeros).

**Key Statistics:**
- `⟨r⟩` — Level spacing ratio (GUE ≈ 0.603, Poisson ≈ 0.386)
- `Σ²(L)` — Number variance over interval length L
- `Δ₃(L)` — Spectral rigidity (Dyson-Mehta statistic)

## Method

1. **Build Hamiltonian** — 3D lattice with Möbius potential, Landau gauge magnetic flux, and twisted boundary conditions
2. **Extract bulk spectrum** — Diagonalize and unfold central eigenvalues
3. **Compute statistics** — Compare against calibrated GUE and Riemann zero references
4. **Verdict** — Classify as World A (generic GUE) or World B (Riemann-like)

## Results

**40 realizations on V100 GPU** across parameter grid:
- σ ∈ {18, 20, 22, 24, 26}
- 8 random boundary twists per σ

| Statistic | Hamiltonian | GUE | Riemann |
|-----------|-------------|-----|---------|
| ⟨r⟩ | 0.598 ± 0.014 | 0.602 ± 0.012 | 0.618 ± 0.01 |
| Δ₃(10) | 0.093 ± 0.008 | 0.092 ± 0.004 | 0.066 ± 0.005 |

**Verdict: World A (Generic GUE)**

Both Σ² and Δ₃ align with GUE, not Riemann. The Hamiltonian exhibits standard quantum chaos statistics.

## Files

| File | Description |
|------|-------------|
| `run_batch.py` | Main batch runner with checkpointing |
| `gue_calibrate.py` | Generates GUE reference from random matrices |
| `summarize.py` | Builds comparison table and verdict |
| `results.jsonl` | Raw data (40 realizations) |
| `gue_ref.json` | GUE baseline statistics |
| `summary.json` | Aggregated results and final verdict |

## Usage
```bash
# Generate GUE reference
python gue_calibrate.py

# Run batch (requires GPU for speed)
python run_batch.py

# Summarize results
python summarize.py
```

## Dependencies
```
numpy
scipy
```

## Hardware

- **Local**: ~1 week on CPU
- **V100 GPU**: ~55 minutes ($0.90 on Prime Intellect)

## Background

This work explores connections between random matrix theory and number theory. The Hilbert-Pólya conjecture suggests Riemann zeta zeros might be eigenvalues of some self-adjoint operator. We test whether specific Hamiltonian constructions exhibit Riemann-like spectral statistics.

**Result**: This particular construction shows generic GUE behavior, not the enhanced rigidity characteristic of Riemann zeros. A clean null result.

## License

MIT
```

---

**requirements.txt**
```
numpy>=1.24.0
scipy>=1.11.0