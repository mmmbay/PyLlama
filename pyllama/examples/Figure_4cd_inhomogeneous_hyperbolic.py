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

def hyperbolic_index(z_nm, n_0, n_s, thickness_inhomogeneous_layer_nm):
    """Calculates the exponentially-varying refractive index at position z in an inhomogeneous layer.

    Args:
        z_nm (float): coordinate where the refractive index is evaluated
        n_0 (float): entry index
        n_s (float): exit index
        thickness_inhomogeneous_layer_nm (float): thickness of the inhomogeneous layer in nm

    Returns:
        _type_: _description_
    """
    return n_0 / (1 - ((n_s - n_0) / n_s) * (z_nm / thickness_inhomogeneous_layer_nm))

def reflectance_analytical(n_0, n_s, thickness_nm, wavelength_nm):
    """Analytically computes the reflection coefficient for a layer with an hyperbolic 
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
    phi = 2 * np.pi * n_0 * n_s * thickness_nm * np.log(n_s / n_0) / (wavelength_nm * (n_s - n_0))
    delta_squared = 0.25 * (np.log(n_s / n_0)) ** 2
    X = phi ** 2 - delta_squared
    reflectance = (1 + (X / (delta_squared * (np.sin(np.sqrt(X))) ** 2))) ** -1
    return reflectance


def generate_layers_same_thickness_hyperbolic_growing_indices(n_0, n_s, thickness_nm, N_sublayers):
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
    n_array = hyperbolic_index(z_array, n_0, n_s, thickness_nm)

    return n_array, thickness_sublayers_array_nm


def generate_layers_same_index_hyperbolic_growing_thickness(n_0, n_s, thickness_nm, wavelength_nm, N_sublayers):
    """Generates discrete layers of the same thickness with exponentially-growing refractive indices

    Args:
        n_0 (float): index of the first layer
        n_s (float): index of the last layer
        thickness_nm (float): length of the stack
        N_sublayers (int): number of discrete layers making up this stack

    Returns:
        _type_: _description_
    """
    # Create an array for layer indices
    layers = np.linspace(0, N_sublayers - 1, N_sublayers)  
    
    # Compute refractive index values
    n_list_optical = n_0 * (n_s / n_0) ** (layers / N_sublayers)

    # Calculate the thickness of the layers
    phi = 1
    thickness_list_1 = wavelength_nm * phi / (2 * np.pi * n_list_optical)
    thickness_list_optical = thickness_list_1 * thickness_nm / np.sum(thickness_list_1)
    
    # Compute the value of the phase correction to be applied
    phi_first_layer = 2 * np.pi * n_list_optical[0] * thickness_list_optical[0] / wavelength_nm

    # Append last layer
    n_list_optical = np.append(n_list_optical, n_s)
    thickness_list_optical = np.append(thickness_list_optical, thickness_nm / np.sum(thickness_list_optical))

    # Compute permittivity tensors
    eps_list_optical = np.array([[[ni**2, 0, 0], [0, ni**2, 0], [0, 0, ni**2]] for ni in n_list_optical])

    return n_list_optical, eps_list_optical, thickness_list_optical, phi_first_layer


def reflectance_spectrum_yeh(n_0, n_s, thickness_nm, wavelength_nm_array, N_sublayers):
    reflectance_spectrum = []
    for wavelength_nm in wavelength_nm_array:
        # Indices and phase shift at the first layer
        n_list_optical, _, _, phi_correc = generate_layers_same_index_hyperbolic_growing_thickness(n_0, n_s, thickness_nm, wavelength_nm, N_sublayers)

        # Calculate the reflection spectrum with Yeh for DISCRETE exponential layers
        resolution_correc = len(n_list_optical) - 2  # don't count the first and last layers
        beta = (n_s / n_0) ** (1 / (resolution_correc + 1))
        y = np.arccos(np.cos(phi_correc) * (1 + beta) / (2 * np.sqrt(beta)))
        C = ((1 - beta) / (2 * np.sqrt(beta))) * np.exp(1j * phi_correc)
        reflectance = (1 + ((np.sin(y)) ** 2) / ((np.abs(C) ** 2) * (np.sin(y * (resolution_correc + 1))) ** 2)) ** -1
        reflectance_spectrum.append(reflectance)
    return np.array(reflectance_spectrum)


def calculate_phase_shift(n, thickness_nm, wavelength_nm):
    """Calculate the phase shift caused by a layer of given refractive index and thickness at a given wavelength"""
    # Compute phase correction
    phi = 2 * np.pi * n * thickness_nm / wavelength_nm
    return phi

def reflectance_spectrum_pyllama_optical_thickness_model(n_0, n_s, thickness_nm, wavelength_nm_array, N_sublayers):
    """Numerically cmputes the reflectance for a layer with a hyperbolic 
    refractive index profile using PyLlama and the StackOpticalThicknessModel class.

    Parameters:
    n_0 : float
        Initial refractive index
    n_s : float
        Final refractive index
    thickness_nm : float
        Thickness of the layer in nanometers
    wavelength_array_nm : array
        Wavelength of incident light in nanometers
    N_sublayers : int
        Number of sub-layers

    Returns:
    reflectance : float
        Reflectance
    """
    n_entry = n_0
    n_exit = n_s
    theta_in_rad = 0

    # Exponential indices
    n_exponential_array = n_0 * (n_s / n_0) ** (np.arange(N_sublayers) / N_sublayers)
    #n_list_optical, eps_list_optical, thickness_list_optical, _ = generate_layers_same_optical_thickness_exponentially_growing_indices(n0_f, ns_f, L_f, wl_f, resolution_f)
    
    reflectance_spectrum = []
    for wavelength_nm in wavelength_nm_array:
        # Calculate the reflection spectrum with PyLlama for layers of optical thickness
        model = ll.StackOpticalThicknessModel(n_exponential_array, thickness_nm, n_entry, n_exit, wavelength_nm, theta_in_rad)

        # Calculate the reflectance matrix with PyLlama
        reflectance_matrix, transmittance_matrix = model.get_refl_trans(method='SM', circ=False)

        # Extract the reflectance
        reflectance = float(reflectance_matrix[0, 0])

        reflectance_spectrum.append(reflectance)
    return np.array(reflectance_spectrum)

def reflectance_spectrum_pyllama_stack_model(n_0, n_s, thickness_nm, wavelength_array_nm, N_sublayers):
    """Numerically cmputes the reflectance for a layer with a hyperbolic 
    refractive index profile using PyLlama and the StackModel class.

    Parameters:
    n_0 : float
        Initial refractive index
    n_s : float
        Final refractive index
    thickness_nm : float
        Thickness of the layer in nanometers
    wavelength_array_nm : array
        Wavelength of incident light in nanometers
    N_sublayers : int
        Number of sub-layers

    Returns:
    reflectance : float
        Reflectance
    """
    # Collect the discrete refractive indices and thicknesses
    n_array, thickness_sublayers_nm_array = generate_layers_same_thickness_hyperbolic_growing_indices(n_0, n_s, thickness_nm, N_sublayers)

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

        # Extract the reflectance
        reflectance = float(reflectance_matrix[0, 0])

        reflectance_spectrum.append(reflectance)
    return np.array(reflectance_spectrum)


def calculate_and_plot(ax_indices, ax_spectra, n_0, n_s, thickness_inhomogeneous_layer_nm, wavelength_nm_array, N_sublayers):
    # Indices
    plt.sca(ax_indices)
    indices_sublayers, thickness_sublayers_nm = generate_layers_same_thickness_hyperbolic_growing_indices(n_0, n_s, thickness_inhomogeneous_layer_nm, N_sublayers)
    thickness_sublayers_cumulative = np.concatenate(([0], np.cumsum(thickness_sublayers_nm[:-1])))
    plt.step(thickness_sublayers_cumulative, indices_sublayers, where="post", label=f"{N_sublayers} sub-layers")
    
    # Reflectance
    plt.sca(ax_spectra)

    # Analytical model for N sublayers
    refl = reflectance_spectrum_yeh(n_0, n_s, thickness_inhomogeneous_layer_nm, wavelength_nm_array, N_sublayers)
    plt.plot(thickness_to_wavelength_ratio_array, refl, linestyle="solid", label=f"{N_sublayers} sub-layers, Yeh")

    # With the optical thickness model (should exactly match the analytical model)
    refl = reflectance_spectrum_pyllama_optical_thickness_model(n_0, n_s, thickness_inhomogeneous_layer_nm, wavelength_nm_array, N_sublayers)
    plt.plot(thickness_to_wavelength_ratio_array, refl, linestyle="dashed", label=f"{N_sublayers} sub-layers, OpticalThicknessModel")

    # With the stack model (should also converge towards the continuous model, but differently)
    refl = reflectance_spectrum_pyllama_stack_model(n_0, n_s, thickness_inhomogeneous_layer_nm, wavelength_nm_array, N_sublayers)
    plt.plot(thickness_to_wavelength_ratio_array, refl, linestyle="dashed", label=f"{N_sublayers} sub-layers, StackModel")

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
    continuous_index = hyperbolic_index(z_nm_array, n_0, n_s, thickness_inhomogeneous_layer_nm)
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

