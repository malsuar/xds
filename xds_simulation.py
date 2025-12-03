"""
xds_simulation.py

Herramientas de alto nivel para simular el Exomoon Doppler Shadow (XDS)
como función del tiempo.

Depende de:
- xds_basics  : Planet, TerminatorGrid, build_terminator_grid
- xds_lineprofile : compute_line_profile, line_centroid, line_skewness
- xds_moon    : Moon, apply_moon_mask
"""

import numpy as np
from xds_lineprofile import (
    compute_line_profile,
    line_centroid,
    line_skewness,
)


def compute_wing_depths(A, vgrid, v_split=0.0):
    """
    Estima la "fuerza" relativa de las alas azul y roja de la línea.

    Integra la absorción A(v) por separado en:
      - ala azul : v < v_split
      - ala roja : v > v_split

    Parameters
    ----------
    A : array_like
        Perfil de absorción A(v) = 1 - I(v).
    vgrid : array_like
        Rejilla de velocidades [km/s].
    v_split : float, optional
        Velocidad que separa alas azul y roja (por defecto 0 km/s).

    Returns
    -------
    blue_area : float
        Área de absorción en el ala azul.
    red_area : float
        Área de absorción en el ala roja.
    """
    A = np.asarray(A)
    v = np.asarray(vgrid)

    mask_blue = v < v_split
    mask_red = v > v_split

    blue_area = np.trapz(A[mask_blue], v[mask_blue]) if np.any(mask_blue) else 0.0
    red_area  = np.trapz(A[mask_red],  v[mask_red])  if np.any(mask_red)  else 0.0

    return blue_area, red_area


def simulate_xds(planet, moon, grid, vgrid, times, return_profiles=False):
    """
    Simula la señal Exomoon Doppler Shadow (XDS) como función del tiempo.

    Para cada instante t:
      1. Aplica la máscara de la luna sobre los pesos del terminador.
      2. Calcula el perfil de transmisión I(v) y la absorción A(v).
      3. Mide:
         - centroide de la línea,
         - skewness,
         - áreas de absorción en las alas azul y roja.

    Parameters
    ----------
    planet : Planet
        Planeta con parámetros de línea ya definidos.
    moon : Moon
        Luna con radio, semieje y periodo orbital.
    grid : TerminatorGrid
        Malla angular del terminador.
    vgrid : array_like
        Rejilla de velocidades [km/s].
    times : array_like
        Tiempos en las mismas unidades que moon.period.
    return_profiles : bool, optional
        Si True, devuelve además los perfiles I(v, t) y A(v, t).

    Returns
    -------
    results : dict
        Diccionario con:
        - 'times'      : array de tiempos
        - 'centroid'   : centroide(t) [km/s]
        - 'skewness'   : skewness(t)
        - 'blue_area'  : área azul(t)
        - 'red_area'   : área roja(t)
        - 'I_profiles' : ndarray [ntimes, Nv] (si return_profiles=True)
        - 'A_profiles' : ndarray [ntimes, Nv] (si return_profiles=True)
    """
    from xds_basics import build_terminator_grid
    from xds_moon import apply_moon_mask

    times = np.asarray(times)
    vgrid = np.asarray(vgrid)

    # Grid sin luna (campo de velocidades + pesos base)
    phi, vlos, w0 = build_terminator_grid(planet, grid)

    nt = len(times)
    Nv = len(vgrid)

    centroids = np.zeros(nt)
    skews = np.zeros(nt)
    blue_areas = np.zeros(nt)
    red_areas = np.zeros(nt)

    if return_profiles:
        I_all = np.zeros((nt, Nv))
        A_all = np.zeros((nt, Nv))
    else:
        I_all = A_all = None

    for i, t in enumerate(times):
        # Pesos con la luna en el instante t
        w_masked, mask = apply_moon_mask(phi, w0, moon, t, grid)

        # Perfil de línea
        I, A = compute_line_profile(planet, vlos, w_masked, vgrid)

        # Diagnósticos
        c = line_centroid(A, vgrid)
        s = line_skewness(A, vgrid)
        blue_area, red_area = compute_wing_depths(A, vgrid, v_split=0.0)

        centroids[i] = c
        skews[i] = s
        blue_areas[i] = blue_area
        red_areas[i] = red_area

        if return_profiles:
            I_all[i, :] = I
            A_all[i, :] = A

    results = {
        "times": times,
        "centroid": centroids,
        "skewness": skews,
        "blue_area": blue_areas,
        "red_area": red_areas,
    }

    if return_profiles:
        results["I_profiles"] = I_all
        results["A_profiles"] = A_all

    return results

def simulate_xds_during_transit(
    planet,
    moon,
    grid,
    vgrid,
    T_transit,
    n_points=200,
    phase_midtransit=0.0,
    return_profiles=False,
):
    """
    Simula la señal XDS *solo durante el tránsito planetario*.

    La variable temporal se define alrededor del centro del tránsito:

        t ∈ [-T_transit/2, +T_transit/2]

    y se asume que en t = 0 el planeta está en el centro del disco estelar.

    La fase orbital de la luna se controla con `phase_midtransit`:

        orbital_phase(t=0) = phase_midtransit

    Parameters
    ----------
    planet : Planet
        Planeta con parámetros de línea ya definidos.
    moon : Moon
        Luna con radio, semieje y periodo orbital.
    grid : TerminatorGrid
        Malla angular del terminador.
    vgrid : array_like
        Rejilla de velocidades [km/s].
    T_transit : float
        Duración del tránsito planetario (en las mismas unidades de tiempo
        que `moon.period`, p.ej. días).
    n_points : int, optional
        Número de muestras temporales durante el tránsito.
    phase_midtransit : float, optional
        Fase orbital de la luna en t = 0 (centro del tránsito), en [0,1).
        Por ejemplo:
            0.0  → luna en el borde azul del terminador
            0.5  → luna en el borde rojo
    return_profiles : bool, optional
        Si True, también devuelve los perfiles I(v,t) y A(v,t).

    Returns
    -------
    results : dict
        Igual que simulate_xds, pero con campos extra:
        - 'times'           : tiempos reales [misma unidad que T_transit]
        - 'transit_phase'   : fase del tránsito en [-0.5, +0.5]
        - 'transit_phase01' : fase del tránsito en [0, 1]
    """
    import copy
    import numpy as np

    # Creamos una copia de la luna para no pisar moon.phase0 original
    moon_eff = copy.copy(moon)
    moon_eff.phase0 = phase_midtransit % 1.0

    # Tiempos durante el tránsito, centrados en t=0
    times = np.linspace(-0.5 * T_transit, 0.5 * T_transit, n_points)

    # Llamamos a la rutina general
    res = simulate_xds(
        planet=planet,
        moon=moon_eff,
        grid=grid,
        vgrid=vgrid,
        times=times,
        return_profiles=return_profiles,
    )

    # Fase del tránsito: -0.5 en el inicio, +0.5 en el final
    transit_phase = times / T_transit  # [-0.5, +0.5]
    transit_phase01 = (times + 0.5 * T_transit) / T_transit  # [0, 1]

    res["times"] = times
    res["transit_phase"] = transit_phase
    res["transit_phase01"] = transit_phase01
    res["vgrid"] = np.array(vgrid)

    return res

# Demo rápida si se ejecuta como script
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from xds_basics import OpacityModel, Planet, TerminatorGrid
    from xds_moon import Moon

    # Planeta tipo Júpiter, rotador rápido
    opacity = OpacityModel(tau0=0.1, pattern="uniform")
    planet = Planet(
        radius=1.0,
        v_eq=10.0,       # km/s
        sigma_line=1.0,  # km/s
        depth_line=0.01,
        opacity_model=opacity
    )

    grid = TerminatorGrid(Nphi=512)

    # Luna
    moon = Moon(
        r_moon=0.3,
        a_moon=3.0,
        period=1.0,   # unidades arbitrarias
        phase0=0.0
    )

    vgrid = np.linspace(-20.0, 20.0, 2001)
    times = np.linspace(0.0, moon.period, 50)

    res = simulate_xds(planet, moon, grid, vgrid, times)

    # ---- Plots de la curva XDS ----
    t = res["times"]
    c = res["centroid"]
    s = res["skewness"]
    blue = res["blue_area"]
    red = res["red_area"]

    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)

    axes[0].plot(t, c * 1e3)   # pasar a m/s si quieres
    axes[0].set_ylabel("centroide [m s$^{-1}$]")

    axes[1].plot(t, s)
    axes[1].set_ylabel("skewness")

    axes[2].plot(t, blue, label="blue wing")
    axes[2].plot(t, red,  label="red wing")
    axes[2].set_ylabel("área alas")
    axes[2].set_xlabel("tiempo [unidades de periodo lunar]")
    axes[2].legend()

    fig.suptitle("Curva temporal del Exomoon Doppler Shadow (demo)")
    plt.tight_layout()
    plt.show()
