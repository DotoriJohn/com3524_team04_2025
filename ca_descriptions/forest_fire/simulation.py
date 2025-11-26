# Name: Forest Fire Simulation - 2D
# Dimensions: 2

import sys
import inspect

this_file_loc = inspect.stack()[0][1]
main_dir_loc = this_file_loc[: this_file_loc.index("ca_descriptions")]
sys.path.append(main_dir_loc)
sys.path.append(main_dir_loc + "capyle")
sys.path.append(main_dir_loc + "capyle/ca")
sys.path.append(main_dir_loc + "capyle/guicomponents")
# ---

from helpers.grids.forest_grid_ingition import Grid2D
import capyle.utils as utils
import numpy as np

from helpers.forest_utils import (
    apply_ignition,
    compute_directional_weights,
    compute_wind_unit_vector,
    compute_weighted_burning_and_wind_field,
    compute_terrain_masks,
    update_burning_cells,
)

# not good to wildcard import, but keeping code clean here
from helpers.forest_states import *


def transition_func(grid, neighbourstates, neighbourcounts, timestep, decay_grid):
    grid_copy = grid.copy()

    # Progress existing fires
    update_burning_cells(grid, decay_grid)

    # Terrain masks
    is_chaparral, is_forest, is_canyon, is_town = compute_terrain_masks(grid_copy)

    # Wind: directional weights + weighted burning + wind field
    wind_unit = compute_wind_unit_vector()
    weights = compute_directional_weights(wind_unit)
    weighted_burning, pw_field = compute_weighted_burning_and_wind_field(
        neighbourstates,
        grid.shape,
        weights,
    )

    # Random field for stochastic ignition
    rng = np.random.random(grid.shape)

    # Apply ignition given terrain, neighbours, wind
    apply_ignition(
        grid=grid,
        decay_grid=decay_grid,
        rng=rng,
        weighted_burning=weighted_burning,
        pw_field=pw_field,
        is_chaparral=is_chaparral,
        is_forest=is_forest,
        is_canyon=is_canyon,
        is_town=is_town,
        timestep=timestep,
    )

    return grid


def setup(args):
    config_path = args[0]
    config = utils.load(config_path)
    config.title = "Wildfire Simulation - 2D"

    config.dimensions = 2

    config.states = [
        BURNING,
        BURNT,
        LAKE,
        CHAPARRAL,
        DENSE_FOREST,
        CANYON,
        TOWN,
    ]

    config.state_colors = [
        (1.0, 0.0, 0.0),  # burning - red
        (0.1, 0.1, 0.1),  # burnt - black
        (0.0, 0.0, 1.0),  # lake - blue
        (0.82, 0.71, 0.55),  # chaparral - tan
        (0.0, 0.39, 0.0),  # dense forest - dark green
        (0.55, 0.27, 0.07),  # canyon - brown
        (0.5, 0.5, 0.5),  # town - gray
    ]
    config.chunk_size = 3
    config.num_generations = 280
    config.grid_dims = (20 * config.chunk_size, 20 * config.chunk_size)
    config.wrap = False

    if len(args) == 2:
        config.save()
        sys.exit()

    return config


def main():
    # Open the config object
    config = setup(sys.argv[1:])

    decay_grid = np.zeros(config.grid_dims)

    # Create grid object
    grid = Grid2D(
        config,
        (transition_func, decay_grid),
    )

    # Run the CA, save grid state every generation to timeline
    timeline = grid.run()

    # save updated config to file
    config.save()
    # save timeline to file
    utils.save(timeline, config.timeline_path)


if __name__ == "__main__":
    main()
