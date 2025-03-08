import numpy as np
from hypothesis import given, settings, strategies as st
import hypothesis.extra.numpy as npst
import pytest
from pyllama.examples.Figure_4ab_Bragg_stack import *

"""
This script calculates the s and p reflection spectra of a Bragg stack with PyLlama and compare the results with 
formulas from P. Yeh (Optical Waves in Layered Media, ISBN: 978-0-471-73192-4, chapter 8).
"""


# Define hypothesis strategies for generating inputs
@st.composite
def multilayer_stack_parameters(draw):
    # Generate the number of cells
    N_stack = draw(st.integers(min_value=2, max_value=10))
    
    # Generate the refractive index and thickness of the first layer
    n_0 = draw(st.floats(min_value=1.0, max_value=2.0))
    thickness_0 = draw(st.floats(min_value=100.0, max_value=500.0))
    
    # Generate the refractive index and thickness of the second layer
    n_1 = draw(st.floats(min_value=n_0, max_value=2.2))
    thickness_1 = draw(st.floats(min_value=100.0, max_value=500.0))
    
    # Generate a list of wavelengths
    lbd_nm_list = draw(npst.arrays(dtype=np.float64, 
                               shape=(draw(st.integers(min_value=10, max_value=100),)), 
                               elements=st.floats(min_value=380.0, max_value=800.0),
                               unique=True)) 
    
    n_layers_list = [n_0, n_1]
    thickness_layers_nm_list = [thickness_0, thickness_1]
    
    # Return the generated parameters as a tuple
    return N_stack, n_layers_list, thickness_layers_nm_list, lbd_nm_list

@pytest.mark.timeout(None)  # Disable timeout for this test
@given(multilayer_stack_parameters())
@settings(max_examples=100, deadline=None)
def test_multilayer_stack_spectrum(params):
    # Randomly-generated parameters
    N_stack, n_layers_list, thickness_layers_nm_list, lbd_nm_list = params

    # Fixed parameters
    n_entry = 1.0
    n_exit = 2.2
    theta_in_deg = 60

    result_pyllama_s, result_pyllama_p = pyllama_multilayer_isotropic_stack_reflectance(N_stack, n_entry, n_exit, n_layers_list, thickness_layers_nm_list, theta_in_deg, lbd_nm_list)
    result_reference_s, result_reference_p = yeh_multilayer_isotropic_stack_reflectance(N_stack, n_entry, n_exit, n_layers_list, thickness_layers_nm_list, theta_in_deg, lbd_nm_list)

    # Ensure results have the same shape
    assert result_pyllama_s.shape == result_reference_s.shape, "Output shapes differ for s polarisation"
    assert result_pyllama_p.shape == result_reference_p.shape, "Output shapes differ for p polarisation"
    
    # Check if both results are NaN in the same positions
    nan_mask_s = np.isnan(result_pyllama_s) & np.isnan(result_reference_s)
    nan_mask_p = np.isnan(result_pyllama_p) & np.isnan(result_reference_p)

    # Compare non-NaN numerical results using np.isclose for numerical tolerance
    close_mask_s = np.isclose(result_pyllama_s, result_reference_s, atol=1e-6)
    close_mask_p = np.isclose(result_pyllama_p, result_reference_p, atol=1e-6)

    # Combine the two masks (NaN results pass, others must be close)
    if not np.all(nan_mask_s | close_mask_s):
        # Fail and print the parameters that caused the failure
        pytest.fail(f"Test failed (s polarisation) for parameters: {params}\nResult PyLlama: {result_pyllama_s}\nResult Yeh: {result_reference_s}")

    if not np.all(nan_mask_p | close_mask_p):
        # Fail and print the parameters that caused the failure
        pytest.fail(f"Test failed (p polarisation) for parameters: {params}\nResult PyLlama: {result_pyllama_p}\nResult Yeh: {result_reference_p}")


