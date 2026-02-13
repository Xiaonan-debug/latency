# 9.3.25 (c) Brassil
# Simulation configuration
#

import uuid
import init

# File management items
import os
filePath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results") + os.sep 
myuuid = uuid.uuid4()   # Make a unique string fot the simulation log filename
current_scenario = init.current_scenario

# Simulation setup
# Items that may be set
SCORE_TYPE = "consequentialHours"
# Score Types can be pH, pO2, glucose, PFI, vectorLength, hours, hyperHours, consequentialHours
STATE_TYPE = "discrete"
# State type can also be unitFloat, an alternate state type that results in a 0 to 1 float value for each state element
# CHECK that unitFloat is actually supported in the code

ACTION_DIMENSION = 9 # This number is how many types of action may be taken
action_value = 0
action = [0] * ACTION_DIMENSION
out_24_list =[""]* 28  # CHECK THIS 28

if init.current_scenario == "VCA":
    # Action setup
    action_step =      [1,  5,   .1,  2.776, 200,   1, 1.2, 10, 10] # This is the standard step
    action_limit_max = [40, 120, .95,  50,   10000, 50, 60, 60, 60]
    action_limit_min = [20, 30,   .2,   0,    0,    0,   0,  0,  0]

    # State scoring limits: L HH, H, L, LL (Only 13 of the 19 states are scored)
    criticalDepletion = [10, 20, 25,   .26, 6.99, 50, 0,   0,  0,  2,     1,   0,  1,   10]
    depletion =         [19, 30, 50,   .4,  7.2, 100, 30, .3,  10, 3,     15,  0, 10,  15] 
    excess =            [38, 120, 200, 1.6, 7.5, 500, 500, .7, 50, 9,     45,  10, 60,  50]
    criticalExcess =    [41, 150, 300, 3.2, 7.6, 700, 700,  1, 60, 30,    50,  20, 100, 100] # DOUBLE CHECK THIS

elif init.current_scenario ==  "EYE": # NEED REVISION TO EYE VALUES THESE ARE ALL ESTIMATES
    # Action setup
    action_step =      [1,  5,   .1,  .2776, 20,   .1, 1.2, .0134, .0134] # This is the standard step
    action_limit_max = [40, 120, .95,  13.88,   1000, 5, 60, .134, .134]
    action_limit_min = [20, 30,   .2,   0,    0,    0,   0,  0,  0]

    # State scoring limits: L HH, H, L, LL (Only 13 of the 19 states are scored)
    criticalDepletion = [10, 20, .1,   12.5, 6.99, 50, 0,   0,  0,  2,     1,    0,  1,   10]
    depletion =         [19, 30, .2,   20,   7.2, 100, 30, .3,  10, 3,     15,   0, 10,  15] 
    excess =            [38, 120, 1.5, 600,  7.5, 500, 500, .7, 50, 9,     45,  10, 60,  50]
    criticalExcess =    [41, 150, 1.6, 1500, 7.6, 700, 700,  1, 60, 30,    50,  20, 100, 100] # DOUBLE CHECK THIS

else:
    raise ValueError(f"Unknown scenario {current_scenario}")

# Big State Vector, 19D [0 to 18].
# 0. Temperature C (state0) *
# 1. Pressure mmHg *
# 2. Flow mL/min *
# 3. VR (REVISED) (state1) *
# 4. pH (state2) *
# 5. pO2 mmHg *
# 6. pvO2 mmHg (state3) *
# 7. svO2 fraction *
# 8. pCO2 mmHg (REVISED) *
# 9. Glucose mM (state4) *
# 10. Insulin mU (state5) *
# 11. Lactate mM *
# 12. Hematocrit % *
# 13. Bicarb mM *
# 14. FiO2 (REVISED) *
# 15. Dialysis in mL per minute (REVISED) *
# 16. Hours *
# 17. Dialysis out mL per minute (REVISED) *
# 18. Volume mL (REVISED) *

# NOTE: This is revised since RT240040 to represent the observations from RT210029
# Changes:
# Action 1 changes from Gas Flow lpm to Pressure
# Action 2 changes from Gas Richness % to FiO2
# Action 3 changes from Glucose mM to Glucose millimoles
# Action 5 changes from Bicarb mM to Bicarb millimoles
# Action 7 new is Dialysis in mL/min
# Action 8 new is Dialysis out mL/min

# Action Vector, 9D, [0 to 8]. This is a single number encoding 8 ternary components (0 to 3 recoded at Step() to -1 to 1)
# 0. Temperature C (simulated in 1 C steps) *
# 1. Pressure mmHg (MAP) (simulated in 5 mmHg steps) *
# 2. FiO2 (adjustable in 0.1 increments) (air mixed with carbogen) *
# 3. Glucose millimoles (2.776 mmol per mL unit injection) *
# 4. Insulin mU (200 mU per mL injection) *
# 5. Bicarb millimoles (1 mmol per mL unit injection) *
# 6. Vasodilator (Papavarine 1.2 mg per mL) *
# 7. Dialysis in mL/min (up to 60 mL per minute) *
# 8. Dialysis out mL/min (up to 60 mL per minute) *



