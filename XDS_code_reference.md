Documentación del núcleo XDS
(Exomoon Doppler Shadow – versión actual del código)

====================================================================

1. Visión general del paquete
   ====================================================================

Este conjunto de módulos implementa un modelo 1D mínimo para simular la señal Exomoon Doppler Shadow (XDS): la modulación en el perfil de transmisión de una línea espectral atmosférica planetaria producida por el tránsito de una luna sobre el terminador del planeta.

La cadena lógica es:

1. xds_basics:

   * Define el planeta, la opacidad atmosférica y una malla 1D del terminador visible.
   * Devuelve, para cada celda del terminador, el ángulo azimutal, la velocidad line-of-sight y un peso geométrico-atmosférico.

2. xds_lineprofile:

   * Toma el campo de velocidades y los pesos, y construye el perfil de transmisión de la línea en el espacio de velocidades.
   * Extrae parámetros observacionales: centroide, momentos, skewness, FWHM.

3. xds_moon:

   * Modela una luna simplificada y su sombra sobre el terminador.
   * Genera una máscara que “apaga” parte de los pesos del terminador (las regiones ocultadas por la luna).

4. xds_simulation:

   * Orquesta todo: recorre una serie temporal, aplica la sombra de la luna, recalcula el perfil de línea y mide la respuesta (centroide, skewness, alas azul/roja) en función del tiempo.
   * Ofrece una versión “global” (simulate_xds) y una restringida al tránsito planetario (simulate_xds_during_transit).

Convenciones generales:

* Ángulos: radianes.
* Velocidades: km/s.
* Radios: unidades arbitrarias pero coherentes (típicamente R_p o R_J).
* Tiempo: unidades arbitrarias; sólo importa de forma relativa (periodos, escalas).
* Todo está normalizado a un continuo I = 1 (flujo de la estrella).

====================================================================
2. Organización de módulos y dependencias
=========================================

* xds_basics.py

  * Clases: OpacityModel, Planet, TerminatorGrid
  * Función: build_terminator_grid
  * Usa: numpy

* xds_lineprofile.py

  * Funciones: compute_line_profile, line_centroid, line_moments, line_skewness, line_fwhm
  * Usa: numpy

* xds_moon.py

  * Clase: Moon
  * Funciones: orbital_phase, moon_phi, moon_shadow_halfwidth, apply_moon_mask
  * Usa: numpy
  * Depende conceptualmente de TerminatorGrid (para rangos de phi) pero no lo importa directamente.

* xds_simulation.py

  * Funciones: compute_wing_depths, simulate_xds, simulate_xds_during_transit
  * Importa:

    * xds_lineprofile: compute_line_profile, line_centroid, line_skewness
    * xds_basics: build_terminator_grid (en simulate_xds)
    * xds_moon: Moon, apply_moon_mask (Moon sólo en el bloque de ejemplo)

El flujo típico de datos es:

Planet + OpacityModel + TerminatorGrid
→ build_terminator_grid
→ (phi, v_los, weights_base)
→ apply_moon_mask (modifica weights en función de t)
→ compute_line_profile
→ métricas de línea (centroide, skewness, alas)
→ simulate_xds / simulate_xds_during_transit empaquetan todo en una serie temporal.

====================================================================
3. xds_basics.py
================

Este módulo define el escenario físico mínimo en el terminador del planeta.

---

## 3.1. Clase OpacityModel

class OpacityModel:
"""
Modelo sencillo de opacidad / transmitancia atmosférica.
"""

Propósito:

* Representar una profundidad óptica tau(φ) a lo largo del terminador y su transmitancia asociada T(φ) = exp(-tau).

Parámetros del constructor:

* tau0 : float
  Profundidad óptica base (adimensional). Si tau0 = 0, la atmósfera es completamente transparente.
* pattern : str, opcional
  Patron de variación de la opacidad. Opciones implementadas:

  * "uniform": tau(φ) = tau0 constante.
  * "hemisphere_cloud": un hemisferio con tau = tau0 y el otro casi transparente (0), usando phi > 0 como criterio.
  * "custom": la profundidad óptica se calcula con una función externa.
* custom_func : callable, opcional
  Función f(phi) → tau(phi) usada cuando pattern = "custom". Debe aceptar arrays de numpy.

Atributos:

* self.tau0 : float
* self.pattern : str
* self.custom_func : callable o None

Métodos:

1. tau(phi)

   * Entrada:

     * phi : array_like (ángulos azimutales en el terminador, en rad).
   * Lógica:

     * Convierte phi a array de numpy.
     * Si pattern == "uniform":

       * tau_phi = tau0 en todo el rango.
     * Si pattern == "hemisphere_cloud":

       * tau_phi = tau0 donde phi > 0.
       * tau_phi = 0 donde phi ≤ 0.
     * Si pattern == "custom":

       * Usa custom_func(phi) para obtener tau_phi.
       * Si custom_func es None, lanza ValueError.
     * En cualquier otro pattern, lanza ValueError.
   * Devuelve:

     * tau_phi : ndarray con la profundidad óptica en cada punto.

2. transmission(phi)

   * Entrada:

     * phi : array_like
   * Lógica:

     * Llama internamente a self.tau(phi).
     * Calcula T = exp(-tau(phi)).
   * Devuelve:

     * T : ndarray (0 ≤ T ≤ 1), transmitancia atmosférica en cada punto.

---

## 3.2. Clase Planet

class Planet:
"""
Planeta gigante para el modelo XDS.
"""

Propósito:

* Describir los parámetros físicos relevantes de la atmósfera para la línea espectral de transmisión.
* Proporcionar el campo de velocidades line-of-sight v_los(φ) bajo la hipótesis de rotación rígida.

Parámetros del constructor:

* radius : float
  Radio del planeta (unidades arbitrarias, pero coherentes con r_moon y a_moon en el módulo de la luna).
* v_eq : float
  Velocidad ecuatorial de rotación [km/s].
* sigma_line : float
  Ancho gaussiano local de la línea de transmisión [km/s].
* depth_line : float
  Profundidad máxima de la línea (0–1). Se interpreta como la profundidad de absorción local de cada celda del terminador.
* opacity_model : OpacityModel
  Instancia de OpacityModel que define la opacidad atmosférica.

Atributos almacenados:

* self.radius
* self.v_eq
* self.sigma_line
* self.depth_line
* self.opacity_model

Métodos:

1. v_los(phi)

   * Entrada:

     * phi : array_like (ángulo en el terminador).
   * Modelo físico:

     * Rotación rígida vista “de canto” (terminador perpendicular a la línea de visión).
     * v_los(φ) = v_eq * sin(φ).
   * Devuelve:

     * v_los_phi : ndarray, velocidades line-of-sight en km/s para cada punto.

---

## 3.3. Clase TerminatorGrid

class TerminatorGrid:
"""
Discretización 1D del terminador visible.
"""

Propósito:

* Definir una malla 1D en ángulo φ sobre el terminador visible del planeta, típicamente desde −π/2 a +π/2.

Parámetros del constructor:

* Nphi : int (opcional, por defecto 512)
  Número de celdas en la malla angular.
* phi_min : float (opcional, por defecto −π/2 rad)
  Límite inferior del rango angular del terminador.
* phi_max : float (opcional, por defecto +π/2 rad)
  Límite superior del rango angular del terminador.

Atributos:

* self.Nphi : int
* self.phi_min : float
* self.phi_max : float
* self.phi : ndarray
  Malla uniforme de Nphi puntos entre phi_min y phi_max (incluidos).

---

## 3.4. Función build_terminator_grid

def build_terminator_grid(planet, grid):
"""
Construye el campo de velocidades y los pesos normalizados en el terminador.
"""

Propósito:

* A partir de un Planet y un TerminatorGrid, calcula:

  * Los ángulos φ.
  * La velocidad line-of-sight v_los(φ).
  * Los pesos geométrico-atmosféricos normalizados que representarán la contribución relativa de cada celda al perfil de línea.

Entradas:

* planet : Planet
  Incluye el modelo de opacidad asociado.
* grid : TerminatorGrid

Lógica:

1. Recupera la malla angular: phi = grid.phi.
2. Calcula el campo de velocidades: vlos = planet.v_los(phi).
3. Calcula la transmitancia atmosférica: T = planet.opacity_model.transmission(phi).
4. Aplica un factor geométrico simple geom = cos(φ), que:

   * Es ≥ 0 en el rango [-π/2, +π/2].
   * Representa aproximadamente el factor de proyección de cada elemento del terminador.
5. Define los pesos no normalizados:

   * weights_raw = T * geom.
6. Normaliza:

   * total = sum(weights_raw)
   * Si total ≤ 0, lanza ValueError (atmósfera totalmente opaca o mal definida).
   * weights = weights_raw / total.

Salidas:

* phi : ndarray (Nphi)
* vlos : ndarray (Nphi)
* weights : ndarray (Nphi, suma = 1)

Uso típico:

* Este trío (phi, vlos, weights) se pasa luego a compute_line_profile y a apply_moon_mask.

====================================================================
4. xds_lineprofile.py
=====================

Módulo enfocado en calcular el perfil de transmisión de la línea en velocidad y extraer propiedades globales del perfil.

---

## 4.1. Función compute_line_profile

def compute_line_profile(planet, vlos, weights, vgrid)

Propósito:

* Integrar el perfil de línea de transmisión I(v) a lo largo del terminador, dadas las velocidades v_los(φ) y los pesos de cada celda.

Modelo:

* Para cada celda i, la absorción local se modela como una gaussiana:

  A_i(v) = depth_line * exp( - (v - v_los_i)^2 / (2 σ^2) )

  donde:

  * depth_line = planet.depth_line
  * σ = planet.sigma_line

* El perfil de absorción total es la suma ponderada:

  A(v) = Σ_i w_i A_i(v)

* El perfil de transmisión es:

  I(v) = 1 - A(v)

Entradas:

* planet : Planet (usa sigma_line y depth_line)
* vlos : array_like (N)
  Velocidades line-of-sight por celda.
* weights : array_like (N)
  Pesos normalizados (suma = 1).
* vgrid : array_like (Nv)
  Rejilla de velocidades en la que se evalúa el perfil [km/s].

Lógica:

* Convierte vlos, weights, vgrid a arrays de numpy.
* Inicializa A(v) = 0 en toda la rejilla.
* Para cada celda (v_i, w_i):

  * Suma w_i * depth * exp[-0.5 * ((v - v_i)/σ)^2] a A.
* Devuelve:

  * I = 1 − A
  * A

Salidas:

* I : ndarray (Nv)
  Perfil de transmisión I(v).
* A : ndarray (Nv)
  Perfil de absorción A(v) = 1 − I(v).

---

## 4.2. Función line_centroid

def line_centroid(A, vgrid)

Propósito:

* Calcular el centroide de la absorción en velocidad:

  v_c = ∫ v A(v) dv / ∫ A(v) dv

Entradas:

* A : array_like
  Perfil de absorción A(v).
* vgrid : array_like
  Rejilla de velocidades.

Lógica:

* Usa integración por trapecios (np.trapz).
* Calcula area = ∫ A(v) dv.

  * Si area ≤ 0, devuelve np.nan.
* Calcula numerador = ∫ v A(v) dv.
* v_c = numerador / area.

Salida:

* v_c : float (km/s)

---

## 4.3. Función line_moments

def line_moments(A, vgrid, order=2, about_centroid=True)

Propósito:

* Calcular momentos de orden n del perfil de absorción.
* Por defecto (order=2, about_centroid=True), devuelve el momento central de segundo orden (varianza de la línea).

Entradas:

* A : array_like (perfil de absorción).
* vgrid : array_like.
* order : int (2, 3, …).
* about_centroid : bool

  * True: momentos centrales respecto al centroide.
  * False: momentos respecto al origen v = 0.

Lógica:

* Convierte A y v a arrays.
* Calcula area = ∫ A(v) dv.

  * Si area ≤ 0, devuelve np.nan.
* Si about_centroid:

  * Calcula centroide v_c con line_centroid.
  * dv = v − v_c.
* Si no:

  * dv = v.
* Momento de orden n:

  m_n = ∫ [dv^n A(v)] dv / area

Salida:

* m_n : float

---

## 4.4. Función line_skewness

def line_skewness(A, vgrid)

Propósito:

* Calcular el skewness (asimetría) de la línea:

  skew = m_3 / (m_2^(3/2))

  donde m_2 y m_3 son los momentos centrales de orden 2 y 3.

Entradas:

* A : array_like
* vgrid : array_like

Lógica:

* Llama a line_moments(A, vgrid, order=2, about_centroid=True) → m2.
* Llama a line_moments(A, vgrid, order=3, about_centroid=True) → m3.
* Si m2 ≤ 0 o ambos momentos no son válidos (NaN), devuelve np.nan.
* Calcula skew = m3 / (m2**1.5).

Interpretación:

* skew > 0  → cola hacia velocidades positivas (ala roja).
* skew < 0  → cola hacia velocidades negativas (ala azul).

Salida:

* skew : float

---

## 4.5. Función line_fwhm

def line_fwhm(I, vgrid)

Propósito:

* Estimar el ancho FWHM de la línea de transmisión I(v).

Definición:

* I_min = min(I)
* depth_eff = 1 − I_min
* Si depth_eff ≤ 0, no hay línea → np.nan.
* I_half = 1 − 0.5 * depth_eff (nivel de mitad de profundidad).
* Se busca el ancho en velocidad entre los dos cruces de I(v) = I_half (uno a la izquierda y otro a la derecha del mínimo).

Entradas:

* I : array_like
  Perfil de transmisión.
* vgrid : array_like

Lógica:

1. Busca el índice del mínimo de I: idx_min.
2. Lado izquierdo:

   * Encuentra el último índice i donde I[i] > I_half antes de idx_min.
   * Interpola linealmente entre (v[i], I[i]) y (v[i+1], I[i+1]) para hallar v_left.
3. Lado derecho:

   * Busca el primer índice j donde I[j] > I_half después del mínimo.
   * Interpola linealmente entre (v[j−1], I[j−1]) y (v[j], I[j]) para obtener v_right.
4. Devuelve FWHM = v_right − v_left.

Salida:

* fwhm : float (km/s) o np.nan si no se puede determinar.

====================================================================
5. xds_moon.py
==============

Introduce un modelo geométrico sencillo para una luna y su sombra sobre el terminador.

---

## 5.1. Clase Moon

class Moon:
"""
Luna en el modelo XDS (geometría 1D simplificada).
"""

Propósito:

* Definir los parámetros físicos y orbitales de la luna.

Parámetros del constructor:

* r_moon : float
  Radio de la luna en unidades de R_p (radio planetario).
* a_moon : float
  Semieje de la órbita de la luna en unidades de R_p. Se usa para estimar el ancho angular de la sombra.
* period : float
  Periodo orbital P_moon de la luna (unidades de tiempo arbitrarias).
* phase0 : float, opcional (por defecto 0.0)
  Fase inicial en [0, 1). En este modelo, controla la posición inicial de la luna en su oscilación sobre el terminador.

Atributos:

* self.r_moon
* self.a_moon
* self.period
* self.phase0 (reducida módulo 1.0)

Chequeos:

* Si a_moon ≤ r_moon, lanza ValueError (geometría irreal: la luna no puede orbitar a una distancia menor que su propio radio).

---

## 5.2. Función orbital_phase

def orbital_phase(t, moon)

Propósito:

* Calcular la fase orbital de la luna en el instante t.

Entradas:

* t : float o array_like
  Tiempo (misma unidad que moon.period).
* moon : Moon

Lógica:

* Convierte t a array (np.asarray).
* phase = (t / moon.period + moon.phase0) % 1.0

Salida:

* phase : float o ndarray en [0, 1).

---

## 5.3. Función moon_phi

def moon_phi(t, moon, grid)

Propósito:

* Posición angular de la proyección de la luna sobre el terminador visible.

Modelo actual (“mejorado”):

* La luna oscila sinusoidalmente entre phi_min y phi_max con periodo P_moon.

Implementación:

* t se convierte a array.
* Se define el centro del terminador y el semiancho:

  * phi_mid = (phi_min + phi_max)/2
  * dphi    = (phi_max − phi_min)/2
* Se define un ángulo orbital continuo:

  * theta = 2π * (t / moon.period + moon.phase0)
* Se proyecta la oscilación:

  * phi_m(t) = phi_mid + dphi * sin(theta)

Entradas:

* t : float o array_like
* moon : Moon
* grid : TerminatorGrid (usa grid.phi_min, grid.phi_max)

Salida:

* phi_m : float o ndarray
  Ángulo central de la sombra de la luna [rad].

Interpretación:

* Cuando sin(theta) = −1 → phi_m = phi_min.
* Cuando sin(theta) = +1 → phi_m = phi_max.
* Recorrido suave, oscilando adelante y atrás a lo largo del terminador.

---

## 5.4. Función moon_shadow_halfwidth

def moon_shadow_halfwidth(moon)

Propósito:

* Estimar el semiancho angular Δφ de la sombra de la luna sobre el terminador.

Aproximación:

* En geometría simple:

  Δφ ≈ arcsin(r_moon / a_moon)

* El ratio r_moon/a_moon se recorta (clip) a [0, 0.99] para evitar problemas numéricos en arcsin.

Entradas:

* moon : Moon

Salida:

* dphi : float (radianes)

---

## 5.5. Función apply_moon_mask

def apply_moon_mask(phi_grid, weights, moon, t, grid)

Propósito:

* Aplicar la sombra de la luna sobre un conjunto de pesos del terminador en un instante de tiempo.

Modelo:

* Las celdas que satisfacen |φ − φ_m(t)| ≤ Δφ se consideran completamente ocultadas por la luna (peso = 0).
* Después de suprimir esas celdas, los pesos restantes se renormalizan para que su suma vuelva a ser 1 (si hay flujo residual).

Entradas:

* phi_grid : array_like (N)
  Valores de φ de la malla del terminador.
* weights : array_like (N)
  Pesos base del terminador sin la luna.
* moon : Moon
* t : float
  Tiempo donde se evalúa la máscara.
* grid : TerminatorGrid
  Necesario para el rango de φ en moon_phi.

Lógica:

1. Convierte phi_grid y weights a arrays.
2. Calcula phi_m = moon_phi(t, moon, grid).
3. Calcula dphi = moon_shadow_halfwidth(moon).
4. Define la máscara:

   * mask = |φ − φ_m| ≤ dphi → celdas ocultadas.
5. weights_masked = copia de weights.

   * weights_masked[mask] = 0.
6. Renormaliza:

   * total = sum(weights_masked)
   * Si total > 0: divide por total.
   * Si total = 0: deja todo en 0 (caso límite).

Salidas:

* weights_masked : ndarray (N)
  Pesos tras aplicar la sombra y renormalizar.
* mask : ndarray (bool)
  True donde la luna oculta el terminador, False en celdas visibles.

====================================================================
6. xds_simulation.py
====================

Módulo de alto nivel para construir la serie temporal de la señal XDS.

---

## 6.1. Función compute_wing_depths

def compute_wing_depths(A, vgrid, v_split=0.0)

Propósito:

* Estimar la fuerza relativa de las alas azul y roja de la línea.
* Integra la absorción A(v) en dos dominios:

  * Ala azul: v < v_split
  * Ala roja: v > v_split

Entradas:

* A : array_like
  Perfil de absorción A(v).
* vgrid : array_like
  Rejilla de velocidades.
* v_split : float, opcional (por defecto 0.0 km/s)
  Velocidad que separa ambas alas.

Lógica:

* Convierte A y v a arrays.
* Define máscaras:

  * mask_blue = v < v_split
  * mask_red  = v > v_split
* Calcula integrales por trapecios:

  * blue_area = ∫ A(v) dv sobre v < v_split (si hay puntos).
  * red_area  = ∫ A(v) dv sobre v > v_split (si hay puntos).
* Si una máscara no tiene puntos, el área correspondiente se deja en 0.

Salidas:

* blue_area : float
* red_area  : float

---

## 6.2. Función simulate_xds

def simulate_xds(planet, moon, grid, vgrid, times, return_profiles=False)

Propósito:

* Simular la señal XDS como función del tiempo, sin restringirse explícitamente a un intervalo de tránsito planetario (es una rutina general).

Para cada instante t:

1. Construye el terminador sin luna (siempre es el mismo, tiempo-independiente).
2. Aplica la máscara de la luna para ese t (lo único tiempo-dependiente).
3. Calcula el perfil de línea I(v,t) y la absorción A(v,t).
4. Extrae:

   * Centroide de la línea.
   * Skewness.
   * Áreas integradas en alas azul y roja.

Entradas:

* planet : Planet
* moon : Moon
* grid : TerminatorGrid
* vgrid : array_like
  Rejilla de velocidades.
* times : array_like
  Lista de tiempos en los que se evalúa la señal. Lo razonable es cubrir varias órbitas lunares o el intervalo de interés.
* return_profiles : bool, opcional (False por defecto)
  Si True, además de las métricas devuelve los perfiles I(v,t) y A(v,t) completos.

Lógica detallada:

1. Importa localmente:

   * from xds_basics import build_terminator_grid
   * from xds_moon import apply_moon_mask
2. Convierte times y vgrid a arrays.
3. Construye el grid base sin luna:

   * phi, vlos, w0 = build_terminator_grid(planet, grid)
   * Esto se hace una sola vez (independiente de t).
4. Define dimensiones:

   * nt = len(times)
   * Nv = len(vgrid)
5. Inicializa arrays de salida:

   * centroids[nt], skews[nt], blue_areas[nt], red_areas[nt]
   * Si return_profiles: I_all[nt, Nv], A_all[nt, Nv]
6. Bucle principal sobre tiempos:

   * Para cada índice i y tiempo t_i:
     a) Aplica la luna:

     * w_t, mask = apply_moon_mask(phi, w0, moon, t_i, grid)
       b) Calcula el perfil de línea:
     * I, A = compute_line_profile(planet, vlos, w_t, vgrid)
       c) Métricas:
     * c = line_centroid(A, vgrid)
     * s = line_skewness(A, vgrid)
     * blue_area, red_area = compute_wing_depths(A, vgrid, v_split=0.0)
       d) Guarda en los arrays:
     * centroids[i] = c
     * skews[i] = s
     * blue_areas[i] = blue_area
     * red_areas[i] = red_area
     * Si return_profiles:

       * I_all[i,:] = I
       * A_all[i,:] = A
7. Empaqueta resultados en un diccionario:

   * "times": times
   * "centroid": centroids
   * "skewness": skews
   * "blue_area": blue_areas
   * "red_area": red_areas
   * Si return_profiles:

     * "I_profiles": I_all
     * "A_profiles": A_all

Salida:

* results : dict con los campos anteriores.

---

## 6.3. Función simulate_xds_during_transit

def simulate_xds_during_transit(
planet,
moon,
grid,
vgrid,
T_transit,
n_points=200,
phase_midtransit=0.0,
return_profiles=False,
)

Propósito:

* Simular la señal XDS restringida únicamente al intervalo temporal del tránsito planetario.
* Enmarca la simulación en t ∈ [−T_transit/2, +T_transit/2], centrando t=0 en el medio del tránsito.

Idea:

* Se diferencia entre:

  * Fase del tránsito (para la geometría planeta-estrella).
  * Fase orbital de la luna (para su posición sobre el terminador).
* Se permite fijar la fase orbital de la luna en el centro del tránsito: orbital_phase(t=0) = phase_midtransit.

Parámetros:

* planet : Planet
* moon : Moon
* grid : TerminatorGrid
* vgrid : array_like
* T_transit : float
  Duración total del tránsito (misma unidad de tiempo que times, se usa como escala).
* n_points : int, opcional (por defecto 200)
  Número de muestras temporales durante el tránsito.
* phase_midtransit : float, opcional (0.0)
  Fase orbital de la luna en el instante t=0 (centro del tránsito). Se aplica a una copia de la luna para no modificar el objeto original.
* return_profiles : bool, opcional
  Igual que en simulate_xds.

Lógica:

1. Importa copy y numpy.
2. Crea una copia de la luna:

   * moon_eff = copy.copy(moon)
   * moon_eff.phase0 = phase_midtransit % 1.0
3. Define el vector de tiempos durante el tránsito:

   * times = linspace(-T_transit/2, +T_transit/2, n_points)
4. Llama a simulate_xds con:

   * planet, moon_eff, grid, vgrid, times, return_profiles.
   * res = simulate_xds(...)
5. Define la fase de tránsito:

   * transit_phase = times / T_transit     → rango [-0.5, +0.5]
   * transit_phase01 = (times + T_transit/2) / T_transit → rango [0, 1]
6. Añade estas fases al diccionario de resultados:

   * res["transit_phase"] = transit_phase
   * res["transit_phase01"] = transit_phase01
7. Devuelve res.

Salidas:

* results : dict
  Contiene los mismos campos que simulate_xds más:

  * "transit_phase"
  * "transit_phase01"

====================================================================
7. Flujo de uso típico
======================

Un pipeline mínimo típico para simular XDS durante un tránsito podría ser:

1. Definir atmósfera y planeta:

   opacity = OpacityModel(tau0=0.1, pattern="uniform")
   planet = Planet(
   radius=1.0,
   v_eq=10.0,
   sigma_line=1.0,
   depth_line=0.01,
   opacity_model=opacity,
   )

2. Definir la malla del terminador:

   grid = TerminatorGrid(Nphi=512)

3. Definir la luna:

   moon = Moon(
   r_moon=0.3,
   a_moon=3.0,
   period=1.0,
   phase0=0.0,
   )

4. Definir la rejilla de velocidades:

   vgrid = np.linspace(-20.0, 20.0, 401)

5. Simular durante un tránsito:

   res = simulate_xds_during_transit(
   planet=planet,
   moon=moon,
   grid=grid,
   vgrid=vgrid,
   T_transit=0.1,        # por ejemplo
   n_points=200,
   phase_midtransit=0.25,
   return_profiles=True,
   )

6. Analizar resultados:

   * res["times"]
   * res["centroid"]
   * res["skewness"]
   * res["blue_area"], res["red_area"]
   * res["I_profiles"], res["A_profiles"] si se solicitaron.

====================================================================
8. Notas, simplificaciones y posibles extensiones
=================================================

* El modelo es 1D en el terminador:

  * No se resuelve explícitamente la latitud; todo se proyecta en una única coordenada φ.
* La rotación del planeta es rígida y siempre alineada de forma que el terminador se ve “de canto”.
* La opacidad atmosférica es extremadamente simple (uniforme, hemisférica o arbitraria via custom_func).
* La luna:

  * No proyecta una sombra real tridimensional, sino una franja 1D sobre φ.
  * Se supone que su órbita da lugar a una oscilación sinusoidal a lo largo del terminador, independiente de la geometría con la estrella.
* No se consideran:

  * Inclinaciones orbitales.
  * Efectos RM estelares, ni movimiento orbital del planeta en torno a la estrella.
  * Variación de la iluminación estelar con el tiempo durante el tránsito.

La estructura del código está pensada para que:

* Planet, OpacityModel y TerminatorGrid puedan reemplazarse por versiones más realistas (p.ej. rotación diferencial).
* Moon y apply_moon_mask puedan adaptarse a geometrías 2D/3D más complejas.
* xds_lineprofile pueda extenderse con otros diagnósticos (kurtosis, fitting de perfiles, etc.).
* xds_simulation actúe como “pegamento” que orquesta las piezas en un pipeline limpio.

Esta documentación describe el estado actual del código y sus supuestos, de forma que puedas ubicar con claridad qué hace cada módulo, cómo se conectan entre sí, y en qué puntos es más natural introducir modificaciones o extensiones físicas para el proyecto XDS.

