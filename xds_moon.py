"""
xds_moon.py

Módulo para incluir una luna en el modelo Exomoon Doppler Shadow (XDS).

- Clase Moon: parámetros físicos y orbitales básicos.
- moon_phi: posición angular de la luna sobre el terminador visible.
- apply_moon_mask: aplica la máscara de ocultación sobre los pesos
  del terminador en un instante dado.

Convenciones:
- r_moon y a_moon en unidades de R_p (radio planetario).
- El terminador visible está en el rango phi ∈ [phi_min, phi_max]
  tal como lo define TerminatorGrid.
"""

import numpy as np


class Moon:
    """
    Luna en el modelo XDS (geometría 1D simplificada).

    Parameters
    ----------
    r_moon : float
        Radio de la luna en unidades de R_p.
    a_moon : float
        Semieje de la órbita de la luna en unidades de R_p.
        (Por ahora sólo se usa para definir el ancho angular de la sombra).
    period : float
        Periodo orbital de la luna (en las mismas unidades de tiempo que uses
        para la variable t en las simulaciones).
    phase0 : float, optional
        Fase inicial en [0,1). phase=0 → luna en el borde izquierdo
        del rango de phi del terminador.
    """

    def __init__(self, r_moon, a_moon, period, phase0=0.0):
        self.r_moon = float(r_moon)
        self.a_moon = float(a_moon)
        self.period = float(period)
        self.phase0 = float(phase0 % 1.0)

        if self.a_moon <= self.r_moon:
            raise ValueError("Se requiere a_moon > r_moon para una geometría razonable.")


def orbital_phase(t, moon):
    """
    Fase orbital de la luna en el instante t.

    Parameters
    ----------
    t : float or array_like
        Tiempo.
    moon : Moon
        Objeto luna.

    Returns
    -------
    phase : float or ndarray
        Fase orbital en el rango [0,1).
    """
    t = np.asarray(t)
    phase = (t / moon.period + moon.phase0) % 1.0
    return phase

'''
def moon_phi(t, moon, grid):
    """
    Posición angular de la proyección de la luna sobre el terminador visible.

    En este modelo 1D simple, se mapea la fase orbital a un recorrido
    lineal de izquierda a derecha a lo largo del rango de phi del grid:

        phase = 0   → phi = phi_min
        phase = 0.5 → phi = (phi_min + phi_max)/2
        phase = 1   → phi = phi_max

    Parameters
    ----------
    t : float or array_like
        Tiempo.
    moon : Moon
        Objeto luna.
    grid : TerminatorGrid
        Malla angular del terminador, con phi_min y phi_max.

    Returns
    -------
    phi_m : float or ndarray
        Ángulo central de la sombra de la luna [rad].
    """
    phase = orbital_phase(t, moon)
    phi_range = grid.phi_max - grid.phi_min
    phi_m = grid.phi_min + phase * phi_range
    return phi_m
'''

def moon_phi(t, moon, grid):
    """
    Posición angular de la proyección de la luna sobre el terminador visible.

    Modelo mejorado: la luna oscila sinusoidalmente entre phi_min y phi_max
    con periodo P_moon.
    """
    t = np.asarray(t)

    phi_mid = 0.5 * (grid.phi_min + grid.phi_max)
    dphi    = 0.5 * (grid.phi_max - grid.phi_min)

    # ángulo orbital continuo
    theta = 2.0 * np.pi * (t / moon.period + moon.phase0)

    # oscilación suave a lo largo del terminador
    phi_m = phi_mid + dphi * np.sin(theta)

    return phi_m

    
def moon_shadow_halfwidth(moon):
    """
    Ancho angular de la sombra de la luna sobre el terminador.

    Aproximación simple:

        Δphi ≈ arcsin( r_moon / a_moon )

    lo que es válido para r_moon <= a_moon.

    Returns
    -------
    dphi : float
        Semiancho angular de la sombra [rad].
    """
    ratio = np.clip(moon.r_moon / moon.a_moon, 0.0, 0.99)
    return np.arcsin(ratio)


def apply_moon_mask(phi_grid, weights, moon, t, grid):
    """
    Aplica la máscara de la luna sobre los pesos del terminador
    en el instante t.

    Las celdas con |phi - phi_m(t)| <= Δphi se consideran completamente
    ocultadas (peso = 0).

    Parameters
    ----------
    phi_grid : array_like
        Ángulos phi de la malla del terminador [rad].
    weights : array_like
        Pesos originales (normalizados) sin luna.
    moon : Moon
        Objeto luna.
    t : float
        Tiempo.
    grid : TerminatorGrid
        Malla angular (se usa para el rango de phi en moon_phi).

    Returns
    -------
    weights_masked : ndarray
        Pesos normalizados tras aplicar la sombra de la luna.
    mask : ndarray (bool)
        Máscara booleana: True donde la luna oculta el terminador.
    """
    phi = np.asarray(phi_grid)
    w = np.asarray(weights)

    phi_m = moon_phi(t, moon, grid)
    dphi = moon_shadow_halfwidth(moon)

    # regiones ocultadas
    mask = np.abs(phi - phi_m) <= dphi

    weights_masked = w.copy()
    weights_masked[mask] = 0.0

    total = np.sum(weights_masked)
    if total > 0:
        weights_masked /= total
    else:
        # Si la luna tapara todo (no debería ocurrir), devolvemos todo cero
        weights_masked[:] = 0.0

    return weights_masked, mask


# Ejemplo mínimo de uso y visualización rápida
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from xds_basics import OpacityModel, Planet, TerminatorGrid, build_terminator_grid
    from xds_lineprofile import compute_line_profile

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
    phi, vlos, w0 = build_terminator_grid(planet, grid)

    # Luna compacta
    moon = Moon(
        r_moon=0.3,   # 0.3 R_p
        a_moon=3.0,   # 3 R_p
        period=1.0,   # unidades de tiempo arbitrarias
        phase0=0.0
    )

    # Rejilla de velocidades
    vgrid = np.linspace(-20.0, 20.0, 2001)

    # Tres instantes: luna sobre lado azul, centro, lado rojo
    times = [0.0, 0.5 * moon.period, 0.9 * moon.period]
    labels = ["blue side", "center", "red side"]
    colors = ["C0", "C1", "C3"]

    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharey="row")

    # Perfil sin luna para referencia
    I_base, A_base = compute_line_profile(planet, vlos, w0, vgrid)

    for ax_col, t, lab, col in zip(axes.T, times, labels, colors):
        # pesos en este instante
        w_masked, mask = apply_moon_mask(phi, w0, moon, t, grid)
        I, A = compute_line_profile(planet, vlos, w_masked, vgrid)

        # fila superior: pesos vs phi
        ax_phi = ax_col[0]
        ax_phi.plot(phi, w0, "--", color="grey", label="sin luna")
        ax_phi.plot(phi, w_masked, color=col, label=f"con luna ({lab})")
        ax_phi.set_xlabel(r"$\phi$ [rad]")
        ax_phi.set_ylabel("peso")
        ax_phi.legend(fontsize=8)

        # fila inferior: perfil de línea
        ax_I = ax_col[1]
        ax_I.plot(vgrid, I_base, "--", color="grey", label="sin luna")
        ax_I.plot(vgrid, I, color=col, label=f"con luna ({lab})")
        ax_I.set_xlabel(r"$v$ [km\,s$^{-1}$]")
        ax_I.set_ylabel(r"$I(v)$")
        ax_I.legend(fontsize=8)

    axes[0,0].set_title("Luna sobre lado azul")
    axes[0,1].set_title("Luna centrada")
    axes[0,2].set_title("Luna sobre lado rojo")

    plt.tight_layout()
    plt.show()
