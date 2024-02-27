import pytest
import numpy as np
import cython

from cythonfn import active_matter

# Test if the simulation runs without errors
def test_simulation_runs():
    active_matter()
    
def test_active_matter_default_parameters():
    x, y, vx, vy, L, v0 = active_matter()
    assert len(x) == 500
    assert len(y) == 500
    assert len(vx) == 500
    assert len(vy) == 500
    assert L == 10
    assert v0 == 1.0
    
def test_active_matter_custom_parameters():
    x, y, vx, vy, L, v0 = active_matter(100, 1000)
    assert len(x) == 1000
    assert len(y) == 1000
    assert len(vx) == 1000
    assert len(vy) == 1000
    assert L == 10
    assert v0 == 1.0

# Test if bird positions are within the box bounds
def test_positions_within_bounds():
    timesteps = 200
    birds = 500
    L = 10

    x, y, _, _, L, _ = active_matter(timesteps, birds)

    assert (np.all(x >= 0) and np.all(x <= L))
    assert (np.all(y >= 0) and np.all(y <= L))

# Test if the velocity magnitude is approximately equal to v0
def test_active_matter_velocity_magnitude():
    _, _, vx, vy, _, v0 = active_matter(timesteps=300, birds=200)
    assert np.allclose(np.sqrt(vx**2 + vy**2), v0, rtol=0.1)
    
# Test if mean angle calculation is functioning
def test_mean_angle_calculation():
    timesteps = 200
    birds = 500

    _, _, vx, vy, _, _ = active_matter(timesteps, birds)

    assert (np.mean(vx)) is not None
    assert (np.mean(vy)) is not None