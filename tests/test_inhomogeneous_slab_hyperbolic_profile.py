import numpy as np
from hypothesis import given, settings, strategies as st
import hypothesis.extra.numpy as npst
import pytest
from pyllama.examples.Figure_4cd_inhomogeneous_hyperbolic import *

"""
This script calculates the reflectance (under normal incidence) of an inhomogeneous slab with 
a hyperbolic refractive index profile, discretised into N layers.
Results calculated with PyLlama are compared with results obtained with an 
analytical formula from P. Yeh (Optical Waves in Layered Media, ISBN: 978-0-471-73192-4, chapter 8).
"""

# Define hypothesis strategies for generating inputs
@st.composite
def hyperbolic_inhomogeneous_layer_parameters(draw):
    # Generate the number of sublayers
    N_sublayers = draw(st.integers(min_value=4, max_value=50))
    
    # Generate the refractive index of the entry medium
    n_0 = draw(st.floats(min_value=1.0, max_value=1.5))
    
    # Generate the refractive index and thickness of the exit medium
    n_s = draw(st.floats(min_value=2 * n_0, max_value=4))
    
    # Generate the thickness
    thickness_nm = draw(st.floats(min_value=100, max_value=500))
    
    # Return the generated parameters as a tuple
    return N_sublayers, n_0, n_s, thickness_nm

@pytest.mark.timeout(None)  # Disable timeout for this test
@given(hyperbolic_inhomogeneous_layer_parameters())
@settings(max_examples=10, deadline=None)
def test_multilayer_stack_spectrum(params):
    # Randomly-generated parameters
    N_sublayers, n_0, n_s, thickness_inhomogeneous_layer_nm = params

    # Wavelength range
    thickness_to_wavelength_ratio_array = np.linspace(0.01, 1, 10)  # x axis of the spectra similar to Yeh
    wavelength_nm_array = thickness_inhomogeneous_layer_nm / thickness_to_wavelength_ratio_array

    # Analytical model for N sublayers
    result_reference = reflectance_spectrum_yeh(n_0, n_s, thickness_inhomogeneous_layer_nm, wavelength_nm_array, N_sublayers)

    # With the optical thickness model (should exactly match the analytical model)
    result_pyllama = reflectance_spectrum_pyllama_optical_thickness_model(n_0, n_s, thickness_inhomogeneous_layer_nm, wavelength_nm_array, N_sublayers)

    # Ensure results have the same shape
    assert result_pyllama.shape == result_reference.shape, "Output shapes differ"

    # The analytical formulas can lead to nans, these values should not be considered for testing
    valid_mask = ~np.isnan(result_reference)
    result_reference = result_reference[valid_mask]
    result_pyllama = result_pyllama[valid_mask]
    
    # Check if both results are NaN in the same positions
    nan_mask = np.isnan(result_reference) & np.isnan(result_pyllama)

    # Compare non-NaN numerical results using np.isclose for numerical tolerance
    close_mask = np.isclose(result_reference, result_pyllama, atol=1e-6)

    # Combine the two masks (NaN results pass, others must be close)
    if not np.all(nan_mask | close_mask):
        # Fail and print the parameters that caused the failure
        pytest.fail(f"Test failed for parameters: {params}\nResult PyLlama: {result_pyllama}\nResult Yeh: {result_reference}")



