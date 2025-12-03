"""
xds_basics.py

Módulo básico para el proyecto Exomoon Doppler Shadow (XDS).

Contiene:
- OpacityModel: describe la transmitancia atmosférica a lo largo del terminador.
- Planet: parámetros físicos del planeta y campo de velocidades v_los(phi).
- TerminatorGrid: discretización 1D del terminador visible.
- build_terminator_grid: construye phi, v_los y pesos normalizados.

Convención de unidades:
- Ángulos en radianes.
- Velocidades en km/s.
- Radios en unidades arbitrarias (p.ej. R_J), pero consistentes.
"""

import numpy as np


class OpacityModel:
    """
    Modelo sencillo de opacidad / transmitancia atmosférica.

    Parameters
    ----------
    tau0 : float
        Profundidad óptica base (adimensional). tau0 = 0 → atmósfera transparente.
    pattern : str, optional
        Patrón de opacidad. Opciones:
        - 'uniform'          : misma tau en todo el terminador.
        - 'hemisphere_cloud' : un hemisferio más nublado (phi > 0).
        - 'custom'           : usa una función proporcionada por el usuario.
    custom_func : callable, optional
        Función de la forma f(phi) → tau(phi), usada sólo si pattern='custom'.
    """

    def __init__(self, tau0=0.0, pattern="uniform", custom_func=None):
        self.tau0 = float(tau0)
        self.pattern = pattern
        self.custom_func = custom_func

    def tau(self, phi):
        """
        Devuelve la profundidad óptica tau(phi) para un array de ángulos.

        Parameters
        ----------
        phi : array_like
            Ángulo azimutal en el terminador [rad].

        Returns
        -------
        tau_phi : ndarray
            Profundidad óptica en cada punto.
        """
        phi = np.asarray(phi)

        if self.pattern == "uniform":
            tau_phi = np.full_like(phi, self.tau0, dtype=float)

        elif self.pattern == "hemisphere_cloud":
            # Hemisferio "positivo" más nublado; el otro casi transparente.
            tau_phi = np.where(phi > 0.0, self.tau0, 0.0)

        elif self.pattern == "custom":
            if self.custom_func is None:
                raise ValueError("custom_func must be provided when pattern='custom'")
            tau_phi = np.asarray(self.custom_func(phi), dtype=float)

        else:
            raise ValueError(f"Unknown pattern '{self.pattern}'")

        return tau_phi

    def transmission(self, phi):
        """
        Devuelve la transmitancia T(phi) = exp(-tau(phi)).

        Parameters
        ----------
        phi : array_like
            Ángulo azimutal en el terminador [rad].

        Returns
        -------
        T : ndarray
            Transmitancia atmosférica en cada punto (0–1).
        """
        return np.exp(-self.tau(phi))


class Planet:
    """
    Planeta gigante para el modelo XDS.

    Parameters
    ----------
    radius : float
        Radio del planeta (unidades arbitrarias, p.ej. R_J).
    v_eq : float
        Velocidad ecuatorial de rotación [km/s].
    sigma_line : float
        Ancho gaussiano local de la línea de transmisión [km/s].
    depth_line : float
        Profundidad máxima de la línea de transmisión (0–1).
    opacity_model : OpacityModel
        Modelo de opacidad asociado a la atmósfera.
    """

    def __init__(self, radius, v_eq, sigma_line, depth_line, opacity_model):
        self.radius = float(radius)
        self.v_eq = float(v_eq)
        self.sigma_line = float(sigma_line)
        self.depth_line = float(depth_line)
        self.opacity_model = opacity_model

    def v_los(self, phi):
        """
        Velocidad line-of-sight en el terminador, para un rotador rígido.

        Asume que observamos el terminador "de canto", de modo que
        v_los(phi) = v_eq * sin(phi).

        Parameters
        ----------
        phi : array_like
            Ángulo azimutal en el terminador [rad].

        Returns
        -------
        v_los_phi : ndarray
            Velocidad line-of-sight en cada punto [km/s].
        """
        phi = np.asarray(phi)
        return self.v_eq * np.sin(phi)


class TerminatorGrid:
    """
    Discretización 1D del terminador visible.

    Parameters
    ----------
    Nphi : int, optional
        Número de celdas en ángulo.
    phi_min, phi_max : float, optional
        Rango angular del terminador [rad]. Por defecto, [-pi/2, +pi/2].
    """

    def __init__(self, Nphi=512, phi_min=-np.pi/2, phi_max=np.pi/2):
        self.Nphi = int(Nphi)
        self.phi_min = float(phi_min)
        self.phi_max = float(phi_max)
        self.phi = np.linspace(self.phi_min, self.phi_max, self.Nphi)


def build_terminator_grid(planet, grid):
    """
    Construye el campo de velocidades y los pesos normalizados en el terminador.

    Parameters
    ----------
    planet : Planet
        Objeto planeta con rotación y modelo de opacidad.
    grid : TerminatorGrid
        Malla angular del terminador.

    Returns
    -------
    phi : ndarray
        Ángulos azimutales de cada celda [rad].
    vlos : ndarray
        Velocidad line-of-sight en cada celda [km/s].
    weights : ndarray
        Pesos normalizados (suma = 1), proporcionales a la transmitancia.
    """
    phi = grid.phi
    vlos = planet.v_los(phi)

    # Transmitancia atmosférica
    T = planet.opacity_model.transmission(phi)

    # --- NUEVO: factor geométrico ---
    geom = np.cos(phi)  # en este rango es >=0

    weights = T * geom
    total = np.sum(weights)

    if total <= 0:
        raise ValueError("Total transmission is zero or negative; check opacity model.")

    weights /= total

    return phi, vlos, weights


# Ejemplo mínimo de uso (puedes comentar o borrar esta parte si no la necesitas):
if __name__ == "__main__":
    # Planeta tipo Júpiter, rotador "rápido"
    opacity = OpacityModel(tau0=0.2, pattern="uniform")
    planet = Planet(
        radius=1.0,       # R_J arbitrario
        v_eq=10.0,        # km/s
        sigma_line=1.0,   # km/s
        depth_line=0.01,  # 1% de profundidad
        opacity_model=opacity
    )

    grid = TerminatorGrid(Nphi=256)
    phi, vlos, w = build_terminator_grid(planet, grid)

    print("N puntos:", len(phi))
    print("v_los min/max [km/s]:", vlos.min(), vlos.max())
    print("Suma de pesos:", w.sum())
