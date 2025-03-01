import numpy as np
import pyllama.pyllama as ll
import matplotlib.pyplot as plt
from functools import reduce


"""
This script calculates the s and p reflection spectra of a Bragg stack with PyLlama and compare the results with 
formulas from Pochi Yeh. The outputs are used for Figure 4.
"""


def get_dynamic_matrix(n, theta_rad, polarisation):
    """
    Computes the dynamic matrix D for a given refractive index, angle, and polarization type.

    Parameters:
    -----------
    n : float or ndarray
        Refractive index of the medium. Can be a scalar or an array.
    theta_rad : float or ndarray
        Angle of incidence (or refraction) in radians. Can be a scalar or an array.
    polarisation : str
        Polarization type, either 's' (TE mode) or 'p' (TM mode).

    Returns:
    --------
    D : ndarray, shape (2, 2) if scalar inputs, or (N, 2, 2) if `n` and `theta_rad` are arrays
        The 2×2 dynamic matrix (or a batch of such matrices if inputs are arrays).
    
    Notes:
    ------
    - For 's' polarization, the matrix is:
    
        D_s = [[1, 1], 
               [n * cos(theta), -n * cos(theta)]]
    
    - For 'p' polarization, the matrix is:

        D_p = [[cos(theta), cos(theta)],
               [n, -n]]

    - If `n` and `theta_rad` are arrays, the function returns an array of shape (N, 2, 2),
      where N is the number of input values.

    Example:
    --------
    >>> get_dynamic_matrix(1.5, np.radians(45), 's')
    array([[ 1.        ,  1.        ],
           [ 1.06066017, -1.06066017]])
    
    >>> get_dynamic_matrix(np.array([1.5, 1.4]), np.radians([45, 30]), 'p').shape
    (2, 2, 2)
    """
    n = np.asarray(n)
    theta_rad = np.asarray(theta_rad)
    
    if polarisation == 's':
        D = np.array([
            [np.ones_like(n), np.ones_like(n)],
            [n * np.cos(theta_rad), -n * np.cos(theta_rad)]
        ])
    elif polarisation == 'p':
        D = np.array([
            [np.cos(theta_rad), np.cos(theta_rad)],
            [n, -n]
        ])
    else:
        raise ValueError("polarisation must be 's' or 'p'")

    return np.moveaxis(D, 2, 0) if D.ndim == 3 else D


def get_propagation_matrix(n, theta_rad, d, lbd_nm):
    """
    Computes the propagation matrix for a given layer in a multilayer optical system.

    Parameters:
    -----------
    n : float or ndarray
        Refractive index of the layer. Can be a scalar or an array.
    theta_rad : float or ndarray
        Angle of incidence (or refraction) in radians. Can be a scalar or an array.
    d : float or ndarray
        Thickness of the layer in nanometers. Can be a scalar or an array.
    lbd_nm : float or ndarray
        Wavelength of the incident light in nanometers. Can be a scalar or an array.

    Returns:
    --------
    P : ndarray, shape (2, 2) if scalar inputs, or (N, 2, 2) if `n`, `theta_rad`, `d`, and `lbd_nm` are arrays
        The 2×2 propagation matrix (or a batch of such matrices if inputs are arrays).

    Notes:
    ------
    - The propagation matrix is defined as:

        P = [[exp(i * phi), 0],
             [0, exp(-i * phi)]]

      where:

        phi = (2 * pi * n * d * cos(theta)) / lbd_nm

    - If `n`, `theta_rad`, `d`, or `lbd_nm` are arrays, the function returns an array of shape (N, 2, 2),
      where N is the number of input values.

    Example:
    --------
    >>> get_propagation_matrix(1.5, np.radians(45), 100, 500)
    array([[0.80901699+0.58778525j, 0.        +0.j        ],
           [0.        +0.j        , 0.80901699-0.58778525j]])

    >>> get_propagation_matrix(1.5, np.radians([30, 45]), 100, np.array([500, 600])).shape
    (2, 2, 2)
    """
    n = np.asarray(n)
    theta_rad = np.asarray(theta_rad)
    d = np.asarray(d)
    lbd_nm = np.asarray(lbd_nm)

    #phi = 2 * np.pi * n * d * np.cos(theta_rad) / lbd_nm
    # Calculate phi for all combinations of theta_rad and lbd_nm
    phi = 2 * np.pi * n[:, np.newaxis] * d[:, np.newaxis] * np.cos(theta_rad[:, np.newaxis]) / lbd_nm


    P = np.array([
        [np.exp(1j * phi), np.zeros_like(phi)],
        [np.zeros_like(phi), np.exp(-1j * phi)]
    ])

    return np.moveaxis(P, (3, 2), (0, 1)) if P.ndim == 4 else P


def yeh_multilayer_isotropic_stack_reflectance(N_stack, n_entry, n_exit, n_layers_list, thickness_layers_nm_list, theta_in_deg, lbd_nm_list):
    """
    Computes the reflection spectra (Rs, Rp) of a multilayer thin-film structure 
    using Yeh's formalism.

    Parameters:
    -----------
    N_stack : int
        Number of repeated unit cells in the multilayer structure.
    n_entry : float
        Refractive index of the incident medium.
    n_exit : float
        Refractive index of the exit medium.
    n_list_unit : list of floats
        List of refractive indices of each layers in the unit cell.
    thick_nm_list_unit : list of floats
        List of layer thicknesses in nanometers for each layer in the unit cell.
    theta_deg_in : float
        Incident angle in degrees.
    lbd_nm_list : ndarray
        Array of wavelengths in nanometers.

    Returns:
    --------
    R_s : ndarray
        Reflectance for s-polarized light at each wavelength.
    R_p : ndarray
        Reflectance for p-polarized light at each wavelength.

    """
    # Construct refractive index and thickness lists for the full multilayer stack
    n_list_stack = np.array([n_entry] + n_layers_list * N_stack + [n_exit])
    thick_nm_list_stack = np.array([0] + thickness_layers_nm_list * N_stack + [0])

    # Convert incident angle to radians
    theta_in_rad = np.radians(theta_in_deg)

    # Compute angles in each layer using Snell's law
    theta_in_rad_layers = np.arcsin(n_list_stack[0] * np.sin(theta_in_rad) / n_list_stack)

    # Compute dynamic matrices for s- and p-polarization
    D_s = get_dynamic_matrix(n_list_stack, theta_in_rad_layers, 's')
    D_p = get_dynamic_matrix(n_list_stack, theta_in_rad_layers, 'p')

    # Compute propagation matrices (vectorized over wavelengths)
    P = get_propagation_matrix(n_list_stack, theta_in_rad_layers, thick_nm_list_stack, lbd_nm_list)  # Shape: (N_lambda, N_thickness, 2, 2)

    # Reshape D_s and D_p over wavelengths
    def add_axis_front(mat, axis_length):
        mat = np.expand_dims(mat, axis=0)  # Shape: (1, a, b, c)
        mat = np.tile(mat, (axis_length, 1, 1, 1))  # Shape: (axis_length, a, b, c)
        return mat
    D_s = add_axis_front(D_s, len(lbd_nm_list))  # Shape: (N_lambda, N_thickness, 2, 2)
    D_p = add_axis_front(D_p, len(lbd_nm_list))  # Shape: (N_lambda, N_thickness, 2, 2)

    # Compute transfer matrices using batch matrix multiplications
    D_s_inv = np.linalg.inv(D_s) # Shape: (N_lambda, N_thickness, 2, 2)
    D_p_inv = np.linalg.inv(D_p) # Shape: (N_lambda, N_thickness, 2, 2)

    # Intermediate layers (remove the entry and exit media)
    M_s_layers = np.matmul(np.matmul(D_s, P), D_s_inv)[:, 1:-1, :, :]
    M_p_layers = np.matmul(np.matmul(D_p, P), D_p_inv)[:, 1:-1, :, :]

    # Multiply the intermediate layers together (swap the axes to place the material layer axis first, and reduce against it)
    M_s_layers = reduce(np.matmul, M_s_layers.transpose(1, 0, 2, 3))
    M_p_layers = reduce(np.matmul, M_p_layers.transpose(1, 0, 2, 3))

    # Multiply the entry and exit half spaces
    M_s = np.matmul(D_s_inv[:, 0, :, :], M_s_layers)
    M_s = np.matmul(M_s, D_s[:, -1, :, :])
    M_p = np.matmul(D_p_inv[:, 0, :, :], M_p_layers)
    M_p = np.matmul(M_p, D_p[:, -1, :, :])

    # Compute reflectance
    R_s = np.abs(M_s[:, 1, 0] / M_s[:, 0, 0]) ** 2
    R_p = np.abs(M_p[:, 1, 0] / M_p[:, 0, 0]) ** 2

    return R_s, R_p


def pyllama_multilayer_isotropic_stack_reflectance(N_stack, n_entry, n_exit, n_layers_list, thickness_layers_nm_list, theta_in_deg, lbd_nm_list):
    """
    Computes the reflection spectra (Rs, Rp) of a multilayer thin-film structure 
    using the scattering matrix method from PyLlama.

    Parameters:
    -----------
    N_stack : int
        Number of repeated unit cells in the multilayer structure.
    n_entry : float
        Refractive index of the incident medium.
    n_exit : float
        Refractive index of the exit medium.
    n_list_unit : list of floats
        List of refractive indices of each layers in the unit cell.
    thick_nm_list_unit : list of floats
        List of layer thicknesses in nanometers for each layer in the unit cell.
    theta_deg_in : float
        Incident angle in degrees.
    lbd_nm_list : ndarray
        Array of wavelengths in nanometers.

    Returns:
    --------
    R_s : ndarray
        Reflectance for s-polarized light at each wavelength.
    R_p : ndarray
        Reflectance for p-polarized light at each wavelength.

    """
    # Incident angle in radians
    theta_in_rad = theta_in_deg * np.pi / 180

    # Permittivity list
    def permittivity(n):
        eps = np.array([[n**2, 0, 0],
                        [0, n**2, 0],
                        [0, 0, n**2]])
        return eps
    eps_list = [permittivity(n) for n in n_layers_list]

    # Reflectance with PyLlama
    R_p = []
    R_s = []
    for wl in lbd_nm_list:
        model = ll.StackModel(eps_list, thickness_layers_nm_list, n_entry, n_exit, wl, theta_in_rad, N_stack)
        refl, _ = model.get_refl_trans(method="SM", circ=False)
        refl_p_to_p = float(refl[0, 0])
        refl_s_to_s = float(refl[1, 1])
        R_p.append(refl_p_to_p)
        R_s.append(refl_s_to_s)

    return np.array(R_s), np.array(R_p)


if __name__ == "__main__":
    # Parameters to choose manually
    N_stack = 10  # number of cells
    n_entry = 1.0  # refractive index of the entry medium
    n_exit = 2.2  # refractive index of the exit medium
    n0 = 2.2  # refractive index of the first layer
    n1 = 1.0  # refractive index of the second layer
    thick0_nm = 200  # thickness of the first layer in nm
    thick1_nm = 500  # thickness of the second layer in nm
    theta_in_deg = 60  # angle of incidence upon the first layer in degrees
    wl_nm_list = np.arange(380, 500, 1)  # np.arange(400, 801, 1)  # wavelength of light in nm

    # Conversions and repacking
    n_layers_list = [n0, n1]
    thick_period_nm = thick0_nm + thick1_nm
    thickness_layers_nm_list = [thick0_nm, thick1_nm]
    theta_in_rad = theta_in_deg * np.pi / 180

    # Results
    result_pyllama_s, result_pyllama_p = pyllama_multilayer_isotropic_stack_reflectance(N_stack, n_entry, n_exit, n_layers_list, thickness_layers_nm_list, theta_in_deg, wl_nm_list)
    result_reference_s, result_reference_p = yeh_multilayer_isotropic_stack_reflectance(N_stack, n_entry, n_exit, n_layers_list, thickness_layers_nm_list, theta_in_deg, wl_nm_list)

    # Plotting
    fig, ax = plt.subplots()
    plt.plot(wl_nm_list, result_reference_p, color="tab:blue", label="Yeh p")
    plt.plot(wl_nm_list, result_pyllama_p, linestyle='dashed', color="tab:orange", label="PyLlama p")
    plt.plot(wl_nm_list, result_reference_s, color="tab:green", label="Yeh s")
    plt.plot(wl_nm_list, result_pyllama_s, linestyle='dashed', color="tab:red", label="PyLlama s")
    plt.xlabel(r'$\lambda$ (nm)')
    plt.ylabel(r'Reflectance')
    plt.legend(loc=1)
    plt.tight_layout()
    #fig_spec.savefig('figure_4_a_b_Bragg_stack.png', dpi=300)
    plt.show()



