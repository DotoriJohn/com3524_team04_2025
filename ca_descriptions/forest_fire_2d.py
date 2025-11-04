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

from capyle.ca import Grid2D, Neighbourhood, CAConfig, randomise2d
import capyle.utils as utils
import numpy as np


UNBURNT = 0
BURNING = 1
BURNT = 2
LAKE = 3
CHAPARRAL = 4
DENSE_FOREST = 5
CANYON = 6


def transition_func(grid, neighbourstates, neighbourcounts):
    (
        unburnt_neighbours,
        burning_neighbours,
        burnt_neighbours,
        lake_neighbours,
        chaparral_neighbours,
        dense_forest_neighbours,
        canyon_neighbours,
    ) = neighbourcounts

    grid_copy = grid.copy()

    on_fire = (burning_neighbours >= 1) & (grid == UNBURNT)

    # Set cells to BURNING where on_fire condition is met
    grid[on_fire] = BURNING

    # Cells that are BURNING become BURNT
    grid[grid_copy == BURNING] = BURNT

    return grid


def setup(args):
    config_path = args[0]
    config = utils.load(config_path)
    config.title = "Wildfire Simulation - 2D"

    config.dimensions = 2
    # States : 0=unburnt (green), 1=burning (red), 2=burnt (black), 3=lake (blue), 4=chaparral (tan), 5=dense forest (dark green), 6=canyon (brown)

    config.states = [
        UNBURNT,
        BURNING,
        BURNT,
        LAKE,
        CHAPARRAL,
        DENSE_FOREST,
        CANYON,
    ]

    config.state_colors = [
        (0.0, 0.6, 0.0),  # unburnt - green
        (1.0, 0.0, 0.0),  # burning - red
        (0.1, 0.1, 0.1),  # burnt - black
        (0.0, 0.0, 1.0),  # lake - blue
        (0.82, 0.71, 0.55),  # chaparral - tan
        (0.0, 0.39, 0.0),  # dense forest - dark green
        (0.55, 0.27, 0.07),  # canyon - brown
    ]

    config.num_generations = 150
    config.grid_dims = (200, 200)

    if len(args) == 2:
        config.save()
        sys.exit()

    return config


def main():
    # Open the config object
    config = setup(sys.argv[1:])

    # Create grid object
    grid = Grid2D(config, transition_func)

    # Run the CA, save grid state every generation to timeline
    timeline = grid.run()

    # save updated config to file
    config.save()
    # save timeline to file
    utils.save(timeline, config.timeline_path)


if __name__ == "__main__":
    main()
