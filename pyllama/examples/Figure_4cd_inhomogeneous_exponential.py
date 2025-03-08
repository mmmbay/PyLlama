import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pyllama.pyllama as ll
from itertools import accumulate
import matplotlib.pylab as pl


"""
The system modelled in this script is an inhomogeneous slab where the refractive index has an exponential profile 
(index-matched at the entry and exit media).
P. Yeh (Optical Waves in Layered Media, ISBN: 978-0-471-73192-4, chapter 8) provides 
- analytical formulas for the modelling of an inhomogeneous slab with an exponential profile
- a discretisation in small layers of equal thickness, where the refractive indices of the layers vary exponentially
"""

def exponential_index(z_nm, n_0, n_s, thickness_inhomogeneous_layer_nm):
    """Calculates the exponentially-varying refractive index at position z in an inhomogeneous layer.

    Args:
        z_nm (float): coordinate where the refractive index is evaluated
        n_0 (float): entry index
        n_s (float): exit index
        thickness_inhomogeneous_layer_nm (float): thickness of the inhomogeneous layer in nm

    Returns:
        _type_: _description_
    """
    return n_0 * np.exp((z_nm / thickness_inhomogeneous_layer_nm) * np.log(n_s / n_0))

def reflectance_analytical(n_0, n_s, thickness_nm, wavelength_nm):
    """Analytically computes the reflection coefficient for a layer with an exponential 
    refractive index profile.

    Parameters:
    n_0 : float
        Initial refractive index
    n_s : float
        Final refractive index
    thickness_nm : float
        Thickness of the layer in nanometers
    wavelength_nm : float
        Wavelength of incident light in nanometers

    Returns:
    reflectance : float
        Reflectance
    """
    log_ratio = np.log(n_s / n_0)

    y_0 = (2 * np.pi * n_0 * thickness_nm) / (wavelength_nm * log_ratio)
    y_s = (2 * np.pi * n_s * thickness_nm) / (wavelength_nm * log_ratio)

    # Replace the zeros by np.nan to avoid warnings
    y_0[y_0 == 0] = np.nan
    y_s[y_s == 0] = np.nan

    A, B = sp.special.jv(0, y_0) - 1j * sp.special.jv(1, y_0), sp.special.yv(0, y_s) - 1j * sp.special.yv(1, y_s)
    C, D = sp.special.yv(0, y_0) - 1j * sp.special.yv(1, y_0), sp.special.jv(0, y_s) - 1j * sp.special.jv(1, y_s)
    E, F = sp.special.jv(0, y_0) + 1j * sp.special.jv(1, y_0), sp.special.yv(0, y_s) - 1j * sp.special.yv(1, y_s)
    G, H = sp.special.yv(0, y_0) + 1j * sp.special.yv(1, y_0), sp.special.jv(0, y_s) - 1j * sp.special.jv(1, y_s)

    reflection_coefficient = (A * B - C * D) / (E * F - G * H)
    reflectance = np.abs(reflection_coefficient) ** 2

    return reflectance


def generate_layers_same_thickness_exponentially_growing_indices(n_0, n_s, thickness_nm, N_sublayers):
    """Generates discrete layers of the same thickness with exponentially-growing refractive indices

    Args:
        n_0 (float): index of the first layer
        n_s (float): index of the last layer
        thickness_nm (float): length of the stack
        N_sublayers (int): number of discrete layers making up this stack

    Returns:
        _type_: _description_
    """
    # Thickness
    thickness_sublayer_nm = thickness_nm / N_sublayers
    thickness_sublayers_array_nm = np.full(N_sublayers + 1, thickness_sublayer_nm)
    
    # Refractive index and permittivity
    z_array = thickness_nm * np.linspace(0, 1, N_sublayers + 1)
    n_array = exponential_index(z_array, n_0, n_s, thickness_nm)

    return n_array, thickness_sublayers_array_nm


def reflectance_spectrum_pyllama(n_0, n_s, thickness_nm, wavelength_array_nm, N_sublayers):
    """Numerically cmputes the reflection coefficient for a layer with an exponential 
    refractive index profile using PyLlama.

    Parameters:
    n_0 : float
        Initial refractive index
    n_s : float
        Final refractive index
    thickness_nm : float
        Thickness of the layer in nanometers
    wavelength_array_nm : float
        Wavelength of incident light in nanometers
    N_sublayers : int
        Number of sub-layers

    Returns:
    reflectance : float
        Reflectance
    """
    # Collect the discrete refractive indices and thicknesses
    n_array, thickness_sublayers_nm_array = generate_layers_same_thickness_exponentially_growing_indices(n_0, n_s, thickness_nm, N_sublayers)

    # Calculate the permittivities
    eps_array = (n_array ** 2)[:, None, None] * np.eye(3)

    # Define the constant parameters
    n_entry = n_0
    n_exit = n_s
    theta_in_rad = 0

    reflectance_spectrum = []
    for wavelength_nm in wavelength_array_nm:
        # Calculate the reflectance matrix with PyLlama
        model = ll.StackModel(eps_array, thickness_sublayers_nm_array, n_entry, n_exit, wavelength_nm, theta_in_rad)
        reflectance_matrix, transmittance_matrix = model.get_refl_trans(method='SM', circ=False)

        # Extract the p-p reflectance
        reflectance = float(reflectance_matrix[0, 0])

        reflectance_spectrum.append(reflectance)
    return np.array(reflectance_spectrum)


def calculate_and_plot(ax_indices, ax_spectra, n_0, n_s, thickness_inhomogeneous_layer_nm, wavelength_nm_array, N_sublayers):
    # Indices
    plt.sca(ax_indices)
    indices_sublayers, thickness_sublayers_nm = generate_layers_same_thickness_exponentially_growing_indices(n_0, n_s, thickness_inhomogeneous_layer_nm, N_sublayers)
    thickness_sublayers_cumulative = np.concatenate(([0], np.cumsum(thickness_sublayers_nm[:-1])))
    plt.step(thickness_sublayers_cumulative, indices_sublayers, where="post", label=f"{N_sublayers} sub-layers")
    # Reflectance
    plt.sca(ax_spectra)
    refl = reflectance_spectrum_pyllama(n_0, n_s, thickness_inhomogeneous_layer_nm, wavelength_nm_array, N_sublayers)
    plt.plot(thickness_to_wavelength_ratio_array, refl, linestyle="dashed", label=f"{N_sublayers} sub-layers")


if __name__ == "__main__":
    # Initialisation of figures
    fig_indices, ax_indices = plt.subplots()
    fig_spectra, ax_spectra = plt.subplots()

    # Parameters
    n_0 = 1  # index of the entry medium
    n_s = 4  # index of the exit medium
    thickness_inhomogeneous_layer_nm = 100  # total thickness of the layer in nm
    thickness_to_wavelength_ratio_array = np.linspace(0, 1, 100)  # x axis of the spectra similar to Yeh
    wavelength_nm_array = thickness_inhomogeneous_layer_nm / thickness_to_wavelength_ratio_array
    #x_array = np.linspace(0.01, 1, 100)

    # Continuously-varying index
    z_nm_array = np.linspace(0, thickness_inhomogeneous_layer_nm, 30)
    continuous_index = exponential_index(z_nm_array, n_0, n_s, thickness_inhomogeneous_layer_nm)
    plt.sca(ax_indices)
    plt.plot(z_nm_array, continuous_index, label="Continuous", color="lightgray", linewidth=5)

    # Analytical reflection spectrum of the inhomogeneous layer
    refl = reflectance_analytical(n_0, n_s, thickness_inhomogeneous_layer_nm, wavelength_nm_array)
    plt.sca(ax_spectra)
    plt.plot(thickness_to_wavelength_ratio_array, refl, label="Analytical", color="lightgray", linewidth=5)

    # Calculate and plot the spectra and indices for 20 sublayers
    N_sublayers = 20
    calculate_and_plot(ax_indices, ax_spectra, n_0, n_s, thickness_inhomogeneous_layer_nm, wavelength_nm_array, N_sublayers)

    # Calculate and plot the spectra and indices for 4 sublayers
    N_sublayers = 4
    calculate_and_plot(ax_indices, ax_spectra, n_0, n_s, thickness_inhomogeneous_layer_nm, wavelength_nm_array, N_sublayers)

    # Axis layout for the plot with the refractive indices
    plt.sca(ax_indices)
    plt.plot([-thickness_inhomogeneous_layer_nm/3, 0], [n_0, n_0], color='k', linestyle='dashed', label="Half-spaces")
    plt.plot([thickness_inhomogeneous_layer_nm, thickness_inhomogeneous_layer_nm + thickness_inhomogeneous_layer_nm/3], [n_s, n_s], color='k', linestyle='dashed')
    plt.axvline(0, color="gray", linestyle='dotted', label="Inhomogeneous layer thickness")
    plt.axvline(thickness_inhomogeneous_layer_nm, color="gray", linestyle='dotted')
    plt.yticks([1, 4])
    plt.gca().set_yticklabels([r'$n_0$', r'$n_s$'])
    plt.xlabel('z axis')
    plt.ylabel('Refractive index')
    plt.legend()
    plt.tight_layout()

    # Axis layout for the spectra
    plt.sca(ax_spectra)
    plt.xlabel(r'$z / \lambda$')
    plt.ylabel("Reflectance")
    plt.xlim((0, 1))
    plt.legend()
    plt.tight_layout()

    plt.show()

