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

# factor to cause wind/acceleration of burning spread
WIND_FACTOR = 1.5 

# direction of wind - north - south
WIND_DIRECTION = ( 1, 0)  # (dy, dx)

# Wind speed 
WIND_SPEED = 3 # scale of 0-3 (0=no wind, 3=high wind)

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

    # Compute directional weights based on wind
    # normalize wind vector to unit vector (only direction)
    wind_vector = np.array(WIND_DIRECTION, dtype=float)
    wind_norm = np.linalg.norm(wind_vector)
    if wind_norm > 0:
        wind_unit = wind_vector / wind_norm
    else:
        wind_unit = np.array([0.0, 0.0])

    # Direction vectors and their names for all 8 neighbours
    directions = [
        ("NW", np.array([-1.0, -1.0])),
        ("N", np.array([-1.0, 0.0])),
        ("NE", np.array([-1.0, 1.0])),
        ("W", np.array([0.0, -1.0])),
        ("E", np.array([0.0, 1.0])),
        ("SW", np.array([1.0, -1.0])),
        ("S", np.array([1.0, 0.0])),
        ("SE", np.array([1.0, 1.0])),
    ]

    # Compute weight for each direction (exponential alignment-based)
    # weight = exp(ln(WIND_FACTOR) * alignment), where alignment = dot(d_unit, wind_unit)
    weights = {}
    for dir_name, dir_vec in directions:
        dir_norm = np.linalg.norm(dir_vec)

        #compute unit vector for neighbour direction
        if dir_norm > 0:
            dir_unit = dir_vec / dir_norm
        else:
            dir_unit = dir_vec
        alignment = np.dot(-dir_unit, wind_unit)
        beta = (np.log(WIND_FACTOR) * WIND_SPEED) if WIND_FACTOR > 0 else 0
        weight = np.exp(beta * alignment)
        weights[dir_name] = weight #southern neighbours burn more

    # Apply weights to neighbour burning counts
    # neighbourstates is tuple: (NW, N, NE, W, E, SW, S, SE)
    NW, N, NE, W, E, SW, S, SE = neighbourstates
    neighbour_arrays = {"NW": NW, "N": N, "NE": NE, "W": W, "E": E, "SW": SW, "S": S, "SE": SE}

    weighted_burning = np.zeros(grid.shape, dtype=float) 
    pw_num = np.zeros(grid.shape, dtype=float)  # numerator for pw (sum of wind weights at burning neighbours)
    pw_den = np.zeros(grid.shape, dtype=float)  # denominator (number of burning neighbours)

    for dir_name, neighbour_array in neighbour_arrays.items():
        is_burning = neighbour_array == BURNING

        weighted_burning += weights[dir_name] * is_burning

        # for pw (wind probability): accumulate weight only where neighbour is burning
        pw_num += weights[dir_name] * is_burning
        pw_den += is_burning

    # Compute pw field: average wind weight of burning neighbours, default 1 when no burning neighbours
    pw_field = np.ones(grid.shape, dtype=float)
    has_burning_neigh = pw_den > 0
    pw_field[has_burning_neigh] = pw_num[has_burning_neigh] / pw_den[has_burning_neigh]

    # Start with base (no-wind) ignition probability per terrain
    pburn = np.zeros(grid.shape, dtype=float)

    pburn[is_chaparral] = IGNITION_PROB[CHAPARRAL]
    pburn[is_forest]    = IGNITION_PROB[DENSE_FOREST]
    pburn[is_canyon]    = IGNITION_PROB[CANYON]

    # Multiply by wind factor pw: pburn = p0(1+pveg)(1+pden)*ps * pw
    pburn *= pw_field

    # Ensure probabilities stay within [0,1]
    pburn = np.clip(pburn, 0.0, 1.0)

    # Terrain-specific neighbour thresholds using weighted count
    # ready as ready to spread / can spread fire
    # ready only if weighted burning neighbours >= minimum neighbours of each terrain
    chap_ready = is_chaparral & (
        weighted_burning >= TERRAIN_MIN_NEIGHBOURS[CHAPARRAL]
    )
    forest_ready = is_forest & (
        weighted_burning >= TERRAIN_MIN_NEIGHBOURS[DENSE_FOREST]
    )
    canyon_ready = is_canyon & (weighted_burning >= TERRAIN_MIN_NEIGHBOURS[CANYON])

    # Probabilistic ignition per terrain
    # ignite only when terrain is ready to ignite and when the ignititon probability is bigger than random grid
    ignite_chap = chap_ready & (rng < pburn) & is_chaparral
    ignite_forest = forest_ready & (rng < pburn) & is_forest
    ignite_canyon = canyon_ready & (rng < pburn) & is_canyon

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
