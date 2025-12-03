"""
XDS full example: Kepler-167e-like case with residuals and noise

This script illustrates how to use the XDS core modules:
    - xds_basics
    - xds_moon
    - xds_simulation
    - xds_lineprofile (for diagnostics on noisy profiles)

It performs:
  * A "clean" XDS simulation with an exomoon during a planetary transit
  * A reference (no-moon) profile
  * Residual maps (with-moon minus no-moon)
  * Addition of Gaussian noise to the line profiles
  * Re-computation of line centroid and skewness from noisy profiles

It generates and saves several figures into a folder named after the planet
(e.g. "Kepler-167e"):

  1) Time series of centroid, skewness, and wing areas (clean + noisy)
  2) 2D absorption map A(v, t) for the clean case
  3) 2D absorption map A(v, t) for the noisy case
  4) 2D residual map (clean) relative to the no-moon reference
  5) 2D residual map (noisy)
  6) Example line profiles at selected transit phases

Run as:
    python XDS_example_full_Kepler167e.py

Assumes that xds_basics.py, xds_moon.py, xds_simulation.py, and
xds_lineprofile.py are importable from the current working directory.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from xds_basics import OpacityModel, Planet, TerminatorGrid, build_terminator_grid
from xds_moon import Moon
from xds_simulation import simulate_xds_during_transit
from xds_lineprofile import (
    compute_line_profile,
    line_centroid,
    line_skewness,
)


def add_noise_to_profiles(I_profiles, snr, random_seed=42):
    """
    Add Gaussian noise to a set of line profiles I(v, t).

    Parameters
    ----------
    I_profiles : ndarray, shape (ntimes, nvel)
        Clean transmission profiles.
    snr : float
        Signal-to-noise ratio. The noise standard deviation is 1/snr
        in absolute units of I.
    random_seed : int, optional
        Seed for reproducibility.

    Returns
    -------
    I_noisy : ndarray, shape (ntimes, nvel)
        Noisy transmission profiles.
    noise : ndarray, shape (ntimes, nvel)
        The actual noise realization added to the profiles.
    """
    rng = np.random.default_rng(random_seed)
    noise = rng.normal(loc=0.0, scale=1.0 / snr, size=I_profiles.shape)
    I_noisy = I_profiles + noise
    return I_noisy, noise


def compute_diagnostics_from_profiles(A_profiles, vgrid):
    """
    Compute line centroid and skewness for each time step from
    a set of absorption profiles A(v, t).

    Parameters
    ----------
    A_profiles : ndarray, shape (ntimes, nvel)
        Absorption profiles as a function of time and velocity.
    vgrid : ndarray, shape (nvel,)
        Velocity grid.

    Returns
    -------
    centroids : ndarray, shape (ntimes,)
        Line centroids in km/s.
    skews : ndarray, shape (ntimes,)
        Line skewness at each time step.
    """
    nt, _ = A_profiles.shape
    centroids = np.zeros(nt)
    skews = np.zeros(nt)

    for i in range(nt):
        A = A_profiles[i]
        c = line_centroid(A, vgrid)
        s = line_skewness(A, vgrid)
        centroids[i] = c
        skews[i] = s

    return centroids, skews


def ensure_output_folder(planet_label):
    """
    Ensure that the output folder for the given planet exists.

    Parameters
    ----------
    planet_label : str
        Name of the planet; used as folder name.

    Returns
    -------
    outdir : str
        Path to the output directory.
    """
    outdir = planet_label
    os.makedirs(outdir, exist_ok=True)
    return outdir


def main():
    # ==============================================================
    # 0. Planet label and output folder
    # ==============================================================

    planet_label = "Kepler-167e"
    outdir = ensure_output_folder(planet_label)
    print(f"Figures will be saved in folder: {outdir}")

    # ==============================================================
    # 1. Define a simple atmospheric opacity model
    # ==============================================================

    # We adopt a uniform optical depth across the terminator.
    # You may change this to "hemisphere_cloud" or "custom"
    # for more complex patterns.
    opacity = OpacityModel(
        tau0=0.2,          # base optical depth (dimensionless)
        pattern="uniform"
    )

    # ==============================================================
    # 2. Define the planet properties
    # ==============================================================

    # Toy "Kepler-167e-like" giant planet.
    planet = Planet(
        radius=1.0,        # planet radius in arbitrary units
        v_eq=5.0,          # equatorial rotation speed [km/s]
        sigma_line=1.0,    # intrinsic Gaussian width of the line [km/s]
        depth_line=0.01,   # local line depth (max absorption per cell)
        opacity_model=opacity
    )

    # ==============================================================
    # 3. Define the terminator grid
    # ==============================================================

    grid = TerminatorGrid(
        Nphi=512           # number of angular cells along the terminator
        # phi_min and phi_max are default: [-pi/2, +pi/2]
    )

    # Inspect basic quantities on the terminator
    phi, vlos, weights = build_terminator_grid(planet, grid)
    print("Terminator grid:")
    print(f"  Number of cells       : {len(phi)}")
    print(f"  v_los range [km/s]    : [{vlos.min():.2f}, {vlos.max():.2f}]")
    print(f"  Sum of weights        : {weights.sum():.3f}")

    # ==============================================================
    # 4. Define the exomoon properties
    # ==============================================================

    # A relatively close exomoon that completes several orbits
    # during a single planetary transit (exaggerated but illustrative).
    moon = Moon(
        r_moon=0.5,        # moon radius in units of R_p
        a_moon=3.0,        # semi-major axis in units of R_p
        period=0.5,        # orbital period in "time units"
        phase0=0.0         # initial orbital phase
    )

    # ==============================================================
    # 5. Define the velocity grid for the transmission line
    # ==============================================================

    # This grid should cover the full rotational broadening range
    # plus some margin.
    vgrid = np.linspace(-20.0, 20.0, 401)  # [km/s]

    # ==============================================================
    # 6. Define the time grid for the planetary transit
    # ==============================================================

    # The absolute units of T_transit are arbitrary; consistency with
    # the moon period is the only requirement.
    T_transit = 1.0        # total transit duration (arbitrary)
    n_points = 1000         # number of time samples across the transit

    # Phase of the moon at mid-transit:
    phase_midtransit = 0.25

    # ==============================================================
    # 7. Run the clean XDS simulation during the transit
    # ==============================================================

    results_clean = simulate_xds_during_transit(
        planet=planet,
        moon=moon,
        grid=grid,
        vgrid=vgrid,
        T_transit=T_transit,
        n_points=n_points,
        phase_midtransit=phase_midtransit,
        return_profiles=True,
    )

    times         = results_clean["times"]             # array of times
    transit_phase = results_clean["transit_phase"]     # from -0.5 to +0.5
    centroid_clean = results_clean["centroid"]         # line centroid [km/s]
    skew_clean     = results_clean["skewness"]         # skewness
    blue_area      = results_clean["blue_area"]        # blue wing area
    red_area       = results_clean["red_area"]         # red wing area
    I_clean        = results_clean["I_profiles"]       # I(t, v)
    A_clean        = results_clean["A_profiles"]       # A(t, v) = 1 - I

    print("\nClean simulation finished.")
    print(f"  Number of time steps : {len(times)}")
    print(f"  vgrid size           : {len(vgrid)}")
    print(f"  Centroid range [km/s]: [{np.nanmin(centroid_clean):.3f}, {np.nanmax(centroid_clean):.3f}]")
    print(f"  Skewness range       : [{np.nanmin(skew_clean):.3f}, {np.nanmax(skew_clean):.3f}]")

    # ==============================================================
    # 8. Compute a reference profile without the moon
    # ==============================================================

    # Here we build a single terminator profile (no time dependence)
    # with the same planet. This represents the "no-moon" reference.
    phi_ref, vlos_ref, w_ref = build_terminator_grid(planet, grid)
    I_ref, A_ref = compute_line_profile(planet, vlos_ref, w_ref, vgrid)

    # Tile the reference profile across all times so we can build
    # residual maps (with-moon minus no-moon).
    I_ref_all = np.tile(I_ref, (len(times), 1))
    A_ref_all = np.tile(A_ref, (len(times), 1))

    # Residual absorption (clean) relative to the no-moon case
    A_res_clean = A_clean - A_ref_all

    # ==============================================================
    # 9. Add noise to the line profiles
    # ==============================================================

    # Define the SNR for the noisy simulation. This is a simple,
    # wavelength-independent white Gaussian noise.
    snr = 100.0

    I_noisy, noise = add_noise_to_profiles(I_clean, snr=snr, random_seed=42)
    A_noisy = 1.0 - I_noisy

    # Recompute centroid and skewness from noisy profiles
    centroid_noisy, skew_noisy = compute_diagnostics_from_profiles(A_noisy, vgrid)

    # Residual absorption (noisy) relative to the no-moon case
    A_res_noisy = A_noisy - A_ref_all

    # ==============================================================
    # 10. Plot time series diagnostics: centroid, skewness, wings
    # ==============================================================

    fig_ts, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    # --- Panel 1: Centroid vs transit phase (clean + noisy) ---
    ax = axes[0]
    ax.plot(transit_phase, centroid_clean, label="Clean", color="C0", lw=2)
    '''ax.plot(
        transit_phase,
        centroid_noisy,
        label="Noisy",
        color="C1",
        lw=0,
        marker="o",
        ms=4,
        alpha=0.7,
    )
    '''
    ax.set_ylabel("Line centroid [km/s]")
    ax.set_title(f"XDS diagnostics during transit ({planet_label})")
    ax.grid(True)
    ax.legend()

    # --- Panel 2: Skewness vs transit phase (clean + noisy) ---
    ax = axes[1]
    ax.plot(transit_phase, skew_clean, label="Clean", color="C0", lw=2)
    '''ax.plot(
        transit_phase,
        skew_noisy,
        label="Noisy",
        color="C1",
        lw=0,
        marker="o",
        ms=4,
        alpha=0.7,
    )
    '''
    ax.set_ylabel("Line skewness")
    ax.grid(True)
    ax.legend()

    # --- Panel 3: Blue / red wing areas vs transit phase (clean only) ---
    # For simplicity, we show wings only for the clean case.
    ax = axes[2]
    ax.plot(
        transit_phase,
        blue_area,
        label="Blue wing (clean)",
        color="C0",
        lw=2,
    )
    ax.plot(
        transit_phase,
        red_area,
        label="Red wing (clean)",
        color="C3",
        lw=2,
    )
    ax.set_xlabel("Transit phase (center = 0)")
    ax.set_ylabel("Wing area (arb. units)")
    ax.legend()
    ax.grid(True)

    fig_ts.tight_layout()
    fig_ts.savefig(os.path.join(outdir, "time_series_diagnostics.png"), dpi=200)
    plt.show()
    plt.close(fig_ts)

    # ==============================================================
    # 11. 2D absorption maps (clean and noisy)
    # ==============================================================

    # --- Clean absorption map ---
    fig_A_clean, axA = plt.subplots(figsize=(8, 6))
    imA = axA.imshow(
        A_clean,
        origin="lower",
        aspect="auto",
        extent=[vgrid[0], vgrid[-1], transit_phase[0], transit_phase[-1]],
        cmap="viridis",
    )
    axA.set_xlabel("Velocity [km/s]")
    axA.set_ylabel("Transit phase")
    axA.set_title(f"Absorption map A(v, t) - clean ({planet_label})")
    cbarA = fig_A_clean.colorbar(imA, ax=axA)
    cbarA.set_label("Absorption A = 1 - I")
    fig_A_clean.tight_layout()
    fig_A_clean.savefig(os.path.join(outdir, "absorption_map_clean.png"), dpi=200)
    plt.show()
    plt.close(fig_A_clean)

    # --- Noisy absorption map ---
    fig_A_noisy, axA2 = plt.subplots(figsize=(8, 6))
    imA2 = axA2.imshow(
        A_noisy,
        origin="lower",
        aspect="auto",
        extent=[vgrid[0], vgrid[-1], transit_phase[0], transit_phase[-1]],
        cmap="viridis",
    )
    axA2.set_xlabel("Velocity [km/s]")
    axA2.set_ylabel("Transit phase")
    axA2.set_title(f"Absorption map A(v, t) - noisy (SNR={snr:.0f})")
    cbarA2 = fig_A_noisy.colorbar(imA2, ax=axA2)
    cbarA2.set_label("Absorption A = 1 - I")
    fig_A_noisy.tight_layout()
    fig_A_noisy.savefig(os.path.join(outdir, "absorption_map_noisy.png"), dpi=200)
    plt.show()
    plt.close(fig_A_noisy)

    # ==============================================================
    # 12. Residual maps (with-moon minus no-moon reference)
    # ==============================================================

    # --- Clean residuals ---
    fig_res_clean, axR = plt.subplots(figsize=(8, 6))
    imR = axR.imshow(
        A_res_clean,
        origin="lower",
        aspect="auto",
        extent=[vgrid[0], vgrid[-1], transit_phase[0], transit_phase[-1]],
        cmap="RdBu_r",
        vmin=-np.max(np.abs(A_res_clean)),
        vmax=+np.max(np.abs(A_res_clean)),
    )
    axR.set_xlabel("Velocity [km/s]")
    axR.set_ylabel("Transit phase")
    axR.set_title(f"Residual map (clean) A(v, t) - A_ref(v)")
    cbarR = fig_res_clean.colorbar(imR, ax=axR)
    cbarR.set_label("Residual absorption")
    fig_res_clean.tight_layout()
    fig_res_clean.savefig(os.path.join(outdir, "residual_map_clean.png"), dpi=200)
    plt.show()
    plt.close(fig_res_clean)

    # --- Noisy residuals ---
    fig_res_noisy, axR2 = plt.subplots(figsize=(8, 6))
    imR2 = axR2.imshow(
        A_res_noisy,
        origin="lower",
        aspect="auto",
        extent=[vgrid[0], vgrid[-1], transit_phase[0], transit_phase[-1]],
        cmap="RdBu_r",
        vmin=-np.max(np.abs(A_res_clean)),  # keep same scale as clean
        vmax=+np.max(np.abs(A_res_clean)),
    )
    axR2.set_xlabel("Velocity [km/s]")
    axR2.set_ylabel("Transit phase")
    axR2.set_title(f"Residual map (noisy) A(v, t) - A_ref(v)")
    cbarR2 = fig_res_noisy.colorbar(imR2, ax=axR2)
    cbarR2.set_label("Residual absorption")
    fig_res_noisy.tight_layout()
    fig_res_noisy.savefig(os.path.join(outdir, "residual_map_noisy.png"), dpi=200)
    plt.show()
    plt.close(fig_res_noisy)

    # ==============================================================
    # 13. Example line profiles at selected transit phases
    # ==============================================================

    # We select a few times: ingress, mid-transit, egress
    idx_ingress = np.argmin(np.abs(transit_phase + 0.4))
    idx_mid     = np.argmin(np.abs(transit_phase - 0.0))
    idx_egress  = np.argmin(np.abs(transit_phase - 0.4))

    idx_list = [idx_ingress, idx_mid, idx_egress]
    labels   = [
        f"Ingress (phase ~ {transit_phase[idx_ingress]:+.2f})",
        f"Mid-transit (phase ~ {transit_phase[idx_mid]:+.2f})",
        f"Egress (phase ~ {transit_phase[idx_egress]:+.2f})",
    ]

    fig_prof, axes_prof = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    for ax, idx, lab in zip(axes_prof, idx_list, labels):
        # No-moon reference profile
        ax.plot(
            vgrid,
            I_ref,
            label="No-moon reference",
            color="k",
            lw=1.5,
            alpha=0.7,
        )
        # Clean with-moon profile
        ax.plot(
            vgrid,
            I_clean[idx],
            label="With moon (clean)",
            color="C0",
            lw=2,
        )
        # Noisy with-moon profile
        ax.plot(
            vgrid,
            I_noisy[idx],
            label="With moon (noisy)",
            color="C1",
            lw=0,
            marker=".",
            ms=3,
            alpha=0.8,
        )

        ax.set_ylabel("Transmission I(v)")
        ax.set_title(lab)
        ax.grid(True)

    axes_prof[-1].set_xlabel("Velocity [km/s]")

    # Use a single legend for all panels (from the first axis)
    handles, legend_labels = axes_prof[0].get_legend_handles_labels()
    fig_prof.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.02),
    )

    fig_prof.tight_layout(rect=[0, 0.05, 1, 1])
    fig_prof.savefig(os.path.join(outdir, "example_profiles.png"), dpi=200)
    plt.close(fig_prof)

    print(f"\nAll figures saved in folder: {outdir}")


if __name__ == "__main__":
    main()
