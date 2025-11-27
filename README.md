# Wildfire Cellular Automaton – COM3524 Team 04 

This project is a 2D cellular automaton model of a forest fire built on top of the CAPyLE framework.
It simulates a fire spreading over different terrain types (lake, chaparral, dense forest, canyon, town) under configurable wind conditions.

---

**Requirements**
The project uses Python and small set of libraries
- Python 3.10 + (any recent version should work)
- numpy
- pyyaml

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

