import numpy as np
from helpers.forest_states import *


# how many cycles a cell remain burning before burnt
BURN_DURATION = {CHAPARRAL: 6, DENSE_FOREST: 15, CANYON: 3, TOWN: 5}  # Lake never burns

# base ignition probability by terrain
IGNITION_PROB = {CHAPARRAL: 0.70, DENSE_FOREST: 0.2, CANYON: 0.90, TOWN: 0.7}

# thresholds for ignition, forest needs more than 2 burning neighbour, not easy to ignite but longer burning cycle
TERRAIN_MIN_NEIGHBOURS = {CHAPARRAL: 1, DENSE_FOREST: 2, CANYON: 1, TOWN: 1}

# factor to cause wind/acceleration of burning spread
WIND_FACTOR = 1.5

# direction of wind - north - south
WIND_DIRECTION = (1, 0)  # (dy, dx)

# Wind speed
WIND_SPEED = 0  # scale of 0-3 (0=no wind, 3=high wind)


def compute_wind_unit_vector() -> np.ndarray:
    """Return the unit wind vector based on global WIND_DIRECTION."""
    wind_vector = np.array(WIND_DIRECTION, dtype=float)
    wind_norm = np.linalg.norm(wind_vector)
    if wind_norm > 0:
        return wind_vector / wind_norm
    return np.array([0.0, 0.0])


def compute_directional_weights(wind_unit) -> dict:
    """
    Compute weight for each neighbour direction based on alignment with the wind.
    Returns a dict mapping direction name -> weight.
    """
    directions = {
        "NW": np.array([-1.0, -1.0]),
        "N": np.array([-1.0, 0.0]),
        "NE": np.array([-1.0, 1.0]),
        "W": np.array([0.0, -1.0]),
        "E": np.array([0.0, 1.0]),
        "SW": np.array([1.0, -1.0]),
        "S": np.array([1.0, 0.0]),
        "SE": np.array([1.0, 1.0]),
    }

    weights = {}
    if WIND_FACTOR <= 0 or WIND_SPEED == 0:
        # no wind effect
        for name in directions:
            weights[name] = 1.0
        return weights
    else:

        beta = np.log(WIND_FACTOR) * WIND_SPEED

        for dir_name, dir_vec in directions.items():
            dir_norm = np.linalg.norm(dir_vec)
            dir_unit = dir_vec / dir_norm if dir_norm > 0 else dir_vec
            # note the minus sign: fire blows downwind
            alignment = np.dot(-dir_unit, wind_unit)
            weights[dir_name] = np.exp(beta * alignment)

    return weights


def compute_weighted_burning_and_wind_field(
    neighbourstates, grid_shape, weights
) -> tuple:
    """
    From neighbour states and directional weights, compute:
      - weighted_burning: weighted count of burning neighbours
      - pw_field: average wind weight of burning neighbours (used to scale pburn)
    """
    NW, N, NE, W, E, SW, S, SE = neighbourstates
    neighbour_arrays = {
        "NW": NW,
        "N": N,
        "NE": NE,
        "W": W,
        "E": E,
        "SW": SW,
        "S": S,
        "SE": SE,
    }

    weighted_burning = np.zeros(grid_shape, dtype=float)
    pw_num = np.zeros(grid_shape, dtype=float)
    pw_den = np.zeros(grid_shape, dtype=float)

    for dir_name, neighbour_array in neighbour_arrays.items():
        is_burning = neighbour_array == BURNING
        w = weights[dir_name]

        weighted_burning += w * is_burning
        pw_num += w * is_burning
        pw_den += is_burning

    pw_field = np.ones(grid_shape, dtype=float)
    has_burning_neigh = pw_den > 0
    pw_field[has_burning_neigh] = pw_num[has_burning_neigh] / pw_den[has_burning_neigh]

    return weighted_burning, pw_field


def update_burning_cells(grid, decay_grid):
    """
    Progress burning cells: decrement their timer, turn them to BURNT when it reaches zero.
    Modifies grid and decay_grid in place.
    """
    grid_copy = grid.copy()
    was_burning = grid_copy == BURNING
    if not was_burning.any():
        return

    decay_grid[was_burning] -= 1
    finished = was_burning & (decay_grid <= 0)
    grid[finished] = BURNT
    decay_grid[finished] = 0


def compute_terrain_masks(grid_copy: np.ndarray) -> tuple:
    """Return boolean masks for each terrain type."""
    is_chaparral = grid_copy == CHAPARRAL
    is_forest = grid_copy == DENSE_FOREST
    is_canyon = grid_copy == CANYON
    is_town = grid_copy == TOWN
    return is_chaparral, is_forest, is_canyon, is_town


def apply_ignition(
    grid,
    decay_grid,
    rng,
    weighted_burning,
    pw_field,
    is_chaparral,
    is_forest,
    is_canyon,
    is_town,
    timestep,
) -> None:
    """
    Apply probabilistic ignition rules based on terrain, neighbours, and wind.
    Modifies grid and decay_grid in place.
    """
    pburn = np.zeros(grid.shape, dtype=float)
    pburn[is_chaparral] = IGNITION_PROB[CHAPARRAL]
    pburn[is_forest] = IGNITION_PROB[DENSE_FOREST]
    pburn[is_canyon] = IGNITION_PROB[CANYON]
    pburn[is_town] = IGNITION_PROB[TOWN]

    # scale by wind field
    pburn *= pw_field
    pburn = np.clip(pburn, 0.0, 1.0)

    # neighbour thresholds using weighted burning counts
    chap_ready = is_chaparral & (weighted_burning >= TERRAIN_MIN_NEIGHBOURS[CHAPARRAL])
    forest_ready = is_forest & (
        weighted_burning >= TERRAIN_MIN_NEIGHBOURS[DENSE_FOREST]
    )
    canyon_ready = is_canyon & (weighted_burning >= TERRAIN_MIN_NEIGHBOURS[CANYON])
    town_ready = is_town & (weighted_burning >= TERRAIN_MIN_NEIGHBOURS[TOWN])

    ignite_chap = chap_ready & (rng < pburn) & is_chaparral
    ignite_forest = forest_ready & (rng < pburn) & is_forest
    ignite_canyon = canyon_ready & (rng < pburn) & is_canyon
    ignite_town = town_ready & (rng < pburn) & is_town

    on_fire = ignite_chap | ignite_forest | ignite_canyon | ignite_town
    if not on_fire.any():
        return

    grid[on_fire] = BURNING

    # initialise burn timers by terrain
    new_chap = on_fire & is_chaparral
    new_forest = on_fire & is_forest
    new_canyon = on_fire & is_canyon
    new_town = on_fire & is_town

    if new_town.any():
        print(f"Timestep {timestep}: Town cells ignited!")

    decay_grid[new_chap] = BURN_DURATION[CHAPARRAL]
    decay_grid[new_forest] = BURN_DURATION[DENSE_FOREST]
    decay_grid[new_canyon] = BURN_DURATION[CANYON]
    decay_grid[new_town] = BURN_DURATION[TOWN]
