# Exomoon Doppler Shadow (XDS)
A lightweight, fully documented Python framework to simulate the **Exomoon Doppler Shadow**:  
the time-dependent distortion imprinted by an exomoon on the **transmission line profile** of a rotating exoplanet during transit.

XDS implements a **1D, terminator-based model** in which:
- the planet has a rotating atmosphere,
- the transmission spectrum is modelled as a Gaussian absorption line,
- the terminator is discretized in azimuthal angle,
- each cell contributes a Doppler-shifted absorption component,
- a moon periodically **masks** part of the terminator, producing a moving residual in velocity space.

The goal of this repository is to provide:
- a transparent and modular numerical testbed,
- a clean end-to-end demonstration of the XDS signal,
- an educational sandbox for exoplanet/exomoon spectroscopy,
- a backbone for more advanced physical models.

---

## 🔍 **What is the Exomoon Doppler Shadow?**

During a planetary transit, the light transmitted through the planet’s atmosphere samples different regions of the terminator.  
If the planet rotates, each region imprints a **Doppler-shifted absorption**.  
A passing moon temporarily blocks part of the atmosphere, removing or renormalizing contributions from specific velocities.

The moon therefore carves a **time-dependent signature** across the line profile — a *Doppler shadow* — analogous to the Rossiter–McLaughlin effect, but acting on the *planetary atmosphere* instead of the star.

XDS models this effect at minimal computational cost.

---

## ✨ Features

- **1D physical model** of atmospheric transmission during transit  
- Rotation-induced Doppler shifts across the terminator  
- Arbitrary opacity models (uniform, hemispheric, custom)  
- Exomoon geometry with time-dependent masking  
- Full line-profile reconstruction  
- Computation of:
  - line centroid,  
  - skewness,  
  - wing integrals (blue/red),  
  - residual maps,  
  - noisy simulations (Gaussian SNR).  
- Clean, modular API  
- Fully documented codebase  
- Example scripts with figures saved automatically

---
## 📁 Repository Structure

XDS/
│
├── xds_basics.py # Planet, opacity model, terminator grid
├── xds_lineprofile.py # Line synthesis and diagnostics
├── xds_moon.py # Moon model and masking function
├── xds_simulation.py # High-level simulation loops
│
├── examples/
│ ├── XDS_example_full_Kepler167e.py # Complete tutorial simulation
│ └── generated_figures/ # Figures produced by example
│
├── docs/
│ └── XDS_documentation.md # Detailed technical documentation
│
└── README.md # You are here


---

## 🛠 Installation

XDS has **no heavy dependencies**. Just:

- Python ≥ 3.8  
- NumPy  
- Matplotlib  

Install with:
Replace yourusername with your GitHub user.

pip install numpy matplotlib
git clone https://github.com/yourusername/XDS.git
cd XDS

## 🚀 Quick Start Example

This is the minimal working example to run a transit simulation:

from xds_basics import OpacityModel, Planet, TerminatorGrid
from xds_moon import Moon
from xds_simulation import simulate_xds_during_transit
import numpy as np

# Atmosphere
opacity = OpacityModel(tau0=0.2, pattern="uniform")
planet = Planet(radius=1.0, v_eq=5.0, sigma_line=1.0, depth_line=0.01,
                opacity_model=opacity)

# Terminator
grid = TerminatorGrid(Nphi=512)

# Moon
moon = Moon(r_moon=0.3, a_moon=3.0, period=0.5)

# Velocity grid
vgrid = np.linspace(-20, 20, 401)

# Transit simulation
res = simulate_xds_during_transit(
    planet=planet,
    moon=moon,
    grid=grid,
    vgrid=vgrid,
    T_transit=1.0,
    n_points=200,
    phase_midtransit=0.25,
    return_profiles=True
)

print(res.keys())


Run it:

python examples/XDS_example_full_Kepler167e.py


This produces:

Absorption map

Noisy vs clean maps

Residual maps

Time-series diagnostics

Example line profiles

All saved in a folder with the planet’s name.

---

## 📘 Documentation

Full technical documentation (classes, functions, physical assumptions) is available in:

docs/XDS_documentation.md


It includes:

formal model description,

equations implemented,

dataset structure,

design philosophy,

extension guidelines.

---

## 🧪 Extending the Model

The XDS codebase is intentionally minimal.
You can extend it by modifying:

✔ planetary model (in Planet)

differential rotation

multiple spectral lines

limb darkening of the star

latitude dependence

✔ moon model (in Moon)

real 3D orbits

non-sinusoidal projected motion

multiple moons

exorings & occulting arcs

✔ line profile model

Voigt profiles

temperature gradients

multi-species atmospheres

✔ star–planet geometry

include projected RM-like component

eccentric orbits

time-varying illumination

If you want, I can help you modularize the next level (RM+XDS combined).
---


📚 Scientific Context & References

The XDS method is inspired by techniques used in:

Rossiter–McLaughlin effect

Transmission spectroscopy

Planet–moon dynamical interactions

If this work contributes to your research, please cite:

Sucerquia, M. (2025). Exomoon Doppler Shadow: A 1D model for atmospheric velocity residuals during transit.
In preparation.

(A placeholder; update with your final publication.)

---

## 👨‍💻 Author

Mario Sucerquia (IPAG / Université Grenoble Alpes - France)
Jaime Alevarado Montes (Macquarie University - Australia)

Astrophysicist – Planetary Dynamics, Exomoons & Exorings
https://malsuar.wixsite.com/mario-sucerquia

---

## 🤝 Contributions

Pull requests are welcome!
Feel free to open issues, suggest improvements, or extend the physics.

---

## 📜 License

MIT License — free to use, modify, and distribute.
See LICENSE file.



