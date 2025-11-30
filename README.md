# Wildfire Cellular Automaton – COM3524 Team 04 

This project is a 2D cellular automaton model of a forest fire built on top of the CAPyLE framework.
It simulates a fire spreading over different terrain types (lake, chaparral, dense forest, canyon, town) under configurable wind conditions.

---

**Requirements**
The project uses Python and small set of libraries
- Python 3.10 + (any recent version should work)
- numpy
- pyyaml

## 0. Environment setup

It is easiest to run the project from an isolated virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate   # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the commands above from the project root only once (or whenever you recreate the environment). Every new terminal session needs the `activate` step before running the simulator.

## 1. How to run the simulation

From the project root (where `main.py` is):

```bash
python main.py
```

Then in the CAPyLE window : 
1. Go to File -> Open
2. Navigate to : ca_descriptions/forest_fire/simulation.py
3. Click Open
4. On the left panel, you can adjust the number of generations
5. Click Apply configuration & run CA.
6. Use the playback controls at the top to step through the simulation.

The grid will show:
- Red - burning cells
- Black - burnt cells
- Blue - lake
- Tan - chapparal
- Dark green - dense forest
- Brown - canyon
- Grey - town

## 2. Changing Model Parameters (Wind, Burning Duration, Ignition Probabilites)

The parameters are stored in:
ca_description/forest_fire/settings.yaml

Besides the wind and ignition settings, you can also point to a different grid map through the same `settings.yaml` file (see the `grid` entry) to experiment with alternative terrain layouts.

After editing the YAML file, simply re-run python main.py and run the simulation again.
