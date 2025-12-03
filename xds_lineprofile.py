"""
xds_lineprofile.py

Módulo para integrar perfiles de línea de transmisión a partir del
campo de velocidades en el terminador (proyecto Exomoon Doppler Shadow).

Depende de:
- xds_basics.Planet
- numpy

Incluye:
- compute_line_profile: integra el perfil en velocidad.
- line_centroid: centroide de la absorción.
- line_moments: momentos centrales de la absorción.
- line_skewness: asimetría (skewness) de la línea.
- line_fwhm: ancho FWHM estimado mediante interpolación.
"""

import numpy as np


def compute_line_profile(planet, vlos, weights, vgrid):
    """
    Calcula el perfil de línea de transmisión integrado sobre el terminador.

    Trabaja en espacio de velocidades, suponiendo que la fuente de fondo
    (la estrella) tiene un continuo plano (normalizado a 1).

    La contribución de cada celda i se modela como una gaussiana local:

        A_i(v) = depth_line * exp( - (v - v_los_i)^2 / (2 sigma_line^2) )

    y el perfil total de absorción es la suma ponderada por weights:

        A(v) = sum_i weights_i * A_i(v)

    De aquí, el perfil de transmisión es:

        I(v) = 1 - A(v)

    Parameters
    ----------
    planet : Planet
        Objeto planeta con sigma_line y depth_line definidos.
    vlos : array_like
        Velocidades line-of-sight de cada celda del terminador [km/s].
    weights : array_like
        Pesos normalizados de cada celda (suma = 1).
    vgrid : array_like
        Rejilla de velocidades donde se calcula el perfil [km/s].

    Returns
    -------
    I : ndarray
        Perfil de transmisión I(v) en la rejilla vgrid.
    A : ndarray
        Perfil de absorción A(v) = 1 - I(v), útil para diagnósticos.
    """
    vlos = np.asarray(vlos)
    weights = np.asarray(weights)
    vgrid = np.asarray(vgrid)

    sigma = planet.sigma_line
    depth = planet.depth_line

    # Absorción total A(v)
    A = np.zeros_like(vgrid, dtype=float)

    # Suma sobre contribuciones locales
    # (Se puede vectorizar más adelante si hace falta acelerar)
    for v_i, w_i in zip(vlos, weights):
        A += w_i * depth * np.exp(-0.5 * ((vgrid - v_i) / sigma) ** 2)

    I = 1.0 - A
    return I, A


def line_centroid(A, vgrid):
    """
    Calcula el centroide de la absorción en velocidad.

    Se define como:

        v_c = sum( v * A(v) ) / sum( A(v) )

    Parameters
    ----------
    A : array_like
        Perfil de absorción A(v).
    vgrid : array_like
        Rejilla de velocidades [km/s].

    Returns
    -------
    v_c : float
        Centroide de la línea [km/s].
    """
    A = np.asarray(A)
    v = np.asarray(vgrid)
    area = np.trapz(A, v)
    if area <= 0:
        return np.nan
    return np.trapz(v * A, v) / area


def line_moments(A, vgrid, order=2, about_centroid=True):
    """
    Calcula momentos de orden n de la absorción en velocidad.

    Por defecto, devuelve el momento central de segundo orden (varianza).

    Parameters
    ----------
    A : array_like
        Perfil de absorción A(v).
    vgrid : array_like
        Rejilla de velocidades [km/s].
    order : int, optional
        Orden del momento (2 = varianza, 3 = momento cúbico, etc.).
    about_centroid : bool, optional
        Si True, calcula momentos centrales alrededor del centroide
        de absorción. Si False, alrededor de v=0.

    Returns
    -------
    m_n : float
        Momento de orden 'order' [km^order / s^order].
    """
    A = np.asarray(A)
    v = np.asarray(vgrid)

    area = np.trapz(A, v)
    if area <= 0:
        return np.nan

    if about_centroid:
        v_c = line_centroid(A, v)
        dv = v - v_c
    else:
        dv = v

    return np.trapz((dv ** order) * A, v) / area


def line_skewness(A, vgrid):
    """
    Calcula el skewness (asimetría) de la línea de absorción.

    Se define como:

        skew = m3 / (m2^(3/2))

    donde m2 y m3 son el segundo y tercer momento central de la absorción.

    Parameters
    ----------
    A : array_like
        Perfil de absorción A(v).
    vgrid : array_like
        Rejilla de velocidades [km/s].

    Returns
    -------
    skew : float
        Skewness adimensional de la línea.
        skew > 0  → cola hacia velocidades positivas (roja)
        skew < 0  → cola hacia velocidades negativas (azul)
    """
    m2 = line_moments(A, vgrid, order=2, about_centroid=True)
    m3 = line_moments(A, vgrid, order=3, about_centroid=True)

    if m2 <= 0 or np.isnan(m2) or np.isnan(m3):
        return np.nan

    return m3 / (m2 ** 1.5)


def line_fwhm(I, vgrid):
    """
    Estima el FWHM de la línea de transmisión.

    Asume un continuo plano I=1 y una línea de absorción con mínimo I_min.
    Se define la profundidad efectiva:

        depth_eff = 1 - I_min

    y se busca el ancho a mitad de profundidad:

        I_half = 1 - 0.5 * depth_eff

    El FWHM se obtiene midiendo la separación en velocidad entre
    los dos puntos donde I(v) cruza I_half, usando interpolación lineal.

    Parameters
    ----------
    I : array_like
        Perfil de transmisión I(v).
    vgrid : array_like
        Rejilla de velocidades [km/s].

    Returns
    -------
    fwhm : float
        Ancho FWHM en km/s. Devuelve np.nan si no se puede determinar.
    """
    I = np.asarray(I)
    v = np.asarray(vgrid)

    I_min = np.min(I)
    depth_eff = 1.0 - I_min

    if depth_eff <= 0:
        return np.nan

    I_half = 1.0 - 0.5 * depth_eff

    # Buscamos los índices donde el perfil cruza I_half
    # a la izquierda y a la derecha del mínimo.
    idx_min = np.argmin(I)

    # Lado izquierdo
    left = np.where(I[:idx_min] > I_half)[0]
    if len(left) == 0:
        return np.nan
    i1 = left[-1]
    i2 = i1 + 1

    # Interpolación lineal para v_left
    v1, v2 = v[i1], v[i2]
    I1, I2 = I[i1], I[i2]
    if I2 == I1:
        return np.nan
    frac_left = (I_half - I1) / (I2 - I1)
    v_left = v1 + frac_left * (v2 - v1)

    # Lado derecho
    right = np.where(I[idx_min:] > I_half)[0]
    if len(right) == 0:
        return np.nan
    j1 = idx_min + right[0] - 1
    j2 = j1 + 1

    v1r, v2r = v[j1], v[j2]
    I1r, I2r = I[j1], I[j2]
    if I2r == I1r:
        return np.nan
    frac_right = (I_half - I1r) / (I2r - I1r)
    v_right = v1r + frac_right * (v2r - v1r)

    return v_right - v_left


# Ejemplo mínimo de uso (puede comentarse o borrarse si no se quiere ejecutar al importar)
if __name__ == "__main__":
    from xds_basics import OpacityModel, Planet, TerminatorGrid, build_terminator_grid

    # Planeta tipo Júpiter, rotador rápido, atmósfera casi transparente
    opacity = OpacityModel(tau0=0.1, pattern="uniform")
    planet = Planet(
        radius=1.0,
        v_eq=10.0,        # km/s
        sigma_line=1.0,   # km/s
        depth_line=0.01,  # 1% de profundidad
        opacity_model=opacity
    )

    # Malla del terminador y campo de velocidades
    grid = TerminatorGrid(Nphi=512)
    phi, vlos, w = build_terminator_grid(planet, grid)

    # Rejilla de velocidades para el perfil
    vgrid = np.linspace(-20.0, 20.0, 2001)  # km/s

    # Perfil integrado
    I, A = compute_line_profile(planet, vlos, w, vgrid)

    # Diagnósticos
    vc = line_centroid(A, vgrid)
    skew = line_skewness(A, vgrid)
    fwhm = line_fwhm(I, vgrid)

    print("Centroide [km/s]:", vc)
    print("Skewness:", skew)
    print("FWHM [km/s]:", fwhm)
