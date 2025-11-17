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

# state definitions (fire + terrain)
UNBURNT = 0
BURNING = 1
BURNT = 2
LAKE = 3
CHAPARRAL = 4
DENSE_FOREST = 5
CANYON = 6

# how many cycles a cell remain burning before burnt
BURN_DURATION = {CHAPARRAL: 6, DENSE_FOREST: 15, CANYON: 3}  # Lake never burns

# base ignition probability by terrain
IGNITION_PROB = {CHAPARRAL: 0.70, DENSE_FOREST: 0.35, CANYON: 0.90}

# thresholds for ignition, forest needs more than 2 burning neighboeru, not easy to ignite but longer burning cycle
TERRAIN_MIN_NEIGHBOURS = {CHAPARRAL: 1, DENSE_FOREST: 2, CANYON: 1}


# need to make a function that count burning neighbours


def transition_func(grid, neighbourstates, neighbourcounts, decay_grid):
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

    # random grid
    rng = np.random.random(grid.shape)  # random in [0,1) per cell

    # tick burning cells and become burnt when timer ends
    was_burning = grid_copy == BURNING
    if was_burning.any():
        decay_grid[was_burning] -= 1
        finished = was_burning & (decay_grid <= 0)
        grid[finished] = BURNT
        decay_grid[finished] = 0

    # Terrain masks -> treat terrain values as unburnt
    is_lake = grid_copy == LAKE
    is_chaparral = grid_copy == CHAPARRAL
    is_forest = grid_copy == DENSE_FOREST
    is_canyon = grid_copy == CANYON

    # Terrain-specific neighbour thresholds (use burning_neighbours from neighbourcounts)
    # ready as ready to spread/ can spread fire
    # ready only if burning neighbours bigger or equal to minimum neighbours of each terrain
    chap_ready = is_chaparral & (
        burning_neighbours >= TERRAIN_MIN_NEIGHBOURS[CHAPARRAL]
    )
    forest_ready = is_forest & (
        burning_neighbours >= TERRAIN_MIN_NEIGHBOURS[DENSE_FOREST]
    )
    canyon_ready = is_canyon & (burning_neighbours >= TERRAIN_MIN_NEIGHBOURS[CANYON])

    # Probabilistic ignition per terrain
    # ignite only when terrain is ready to ignite and when the ignititon probability is bigger than random grid
    ignite_chap = chap_ready & (rng < IGNITION_PROB[CHAPARRAL])
    ignite_forest = forest_ready & (rng < IGNITION_PROB[DENSE_FOREST])
    ignite_canyon = canyon_ready & (rng < IGNITION_PROB[CANYON])

    on_fire = ignite_chap | ignite_forest | ignite_canyon

    # Set cells to BURNING where on_fire condition is met
    grid[on_fire] = BURNING

    # initialise burn timer
    if on_fire.any():
        chaparral_neighbours = on_fire & is_chaparral
        dense_forest_neighbours = on_fire & is_forest
        canyon_neighbours = on_fire & is_canyon

        decay_grid[chaparral_neighbours] = BURN_DURATION[CHAPARRAL]
        decay_grid[dense_forest_neighbours] = BURN_DURATION[DENSE_FOREST]
        decay_grid[canyon_neighbours] = BURN_DURATION[CANYON]

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
    config.chunk_size = 10
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
    # decay_grid.fill(2)
    # -> lakes and unlit terrain starts with the timer

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
