# 9.3.25 (c) Brassil
# Simulation initial conditions
# Init Rev 1 presents 2 scenarios: EYE and VCA as customizable plus maintains all the unchanging values as constants across all scenarios

import random

# SELECT the scenario for the graft type
# current_scenario = "VCA" # VCA simulation
current_scenario = "EYE" # Eye simulation

#Constants 
GLUCOSE_MW_G_PER_MOLE = 180.156
CO2_SOLUBILITY_mM_PER_MMHG = .03
MOLAR_VOLUME_STP_GAS_L = 22.4
BICARB_IN_KREBS_mM = 25
H_H_PK = 6.1  # Henderson-Hasselbalch pK, physical constant
LACTATES_PER_GLUCOSE = 2
OXYGENS_PER_GLUCOSE = 6
CO2S_PER_GLUCOSE = 6
INSULIN_mUNIT_PER_mMOLE_GLUCOSE = 100
ATMOSPHERIC_PRESSURE_MMHG = 760  # Physical constant
CARBOGEN_OXYGEN_FRACTION = 0.95
CARBOGEN_CO2_TO_O2_FRACTION = 0.053 # (5% / 95%)

# Empirical
TRANSFUSION_TIME_HOURS = 12
BIG_STATE_DIMENSION = 19

if current_scenario == "VCA":
    STEP_DURATION_HR = 1
    GRAFT_GRAMS = 300
    MINUTES_PER_STEP = 60
    VOLUME_MINIMUM_ML = 100
    dVR_dHR_MMHG_PER_ML_PER_MINUTE_PER_HOUR = .0208
    FLOW_INIT_VARIATION = .25
    VR_STOCHASTIC_FACTOR = 1.0052
    ANAEROBIC_FRACTION = 0.7
    HEMATOCRIT_CHANGE_PER_HOUR = 1
    GLUCOSE_CONSUMPTION_MMOLE_PER_GRAM_HOUR = .00258 # Based on 2021 2022 experiments
    VR_ORGAN_FACTOR = 1 #Empirical
    temperature_C = 36
    pressure_mmHg = 80
    base_flow = 80# variable 
    pH = 7.35  # Lets just say we titrate to a perfect starting pH
    pO2_mmHg = 380 # Corresponds to FiO2 = 50%
    pvO2_mmHg = 40 # Arbitrary good value for mixed venous
    svO2 = .75 # More or less matches the pvO2
    glucose_mM = 6 # Middle of 4 to 8 feeling great - set perfectly ~108 mg/dL
    insulin_mU = 160
    lactate_mM = 0
    hematocrit = 13.75
    dialysis_in_mL_per_min = 0
    hours = 0
    dialysis_out_mL_per_min = 0
    volume_mL = 650
    
elif current_scenario ==  "EYE": # NEED REVISION TO EYE VALUES
    STEP_DURATION_HR = 1
    GRAFT_GRAMS = 1
    MINUTES_PER_STEP = 60
    VOLUME_MINIMUM_ML = 10
    dVR_dHR_MMHG_PER_ML_PER_MINUTE_PER_HOUR = 5 # Estimate
    FLOW_INIT_VARIATION = .25
    VR_STOCHASTIC_FACTOR = 1.0052
    ANAEROBIC_FRACTION = 0.7
    HEMATOCRIT_CHANGE_PER_HOUR = 1
    GLUCOSE_CONSUMPTION_MMOLE_PER_GRAM_HOUR = .00258 # Based on 2021 2022 experiments
    VR_ORGAN_FACTOR = 23000 # Estimate
    temperature_C = 36
    pressure_mmHg = 80
    base_flow = 1 # variable 
    pH = 7.35  # Lets just say we titrate to a perfect starting pH
    pO2_mmHg = 380 # Corresponds to FiO2 = 50%
    pvO2_mmHg = 40 # Arbitrary good value for mixed venous
    svO2 = .75 # More or less matches the pvO2
    glucose_mM = 6 # Middle of 4 to 8 feeling great - set perfectly ~108 mg/dL
    insulin_mU = 16 #Estimate
    lactate_mM = 0
    hematocrit = 13.75
    dialysis_in_mL_per_min = 0 #134 uL/minute is teh normal pump speed
    hours = 0
    dialysis_out_mL_per_min = 0
    volume_mL = 150   


# Calculations relying on the above values
HOURS_PER_STEP = MINUTES_PER_STEP / 60 # 60 minutes per hour!
AEROBIC_FRACTION = 1 - ANAEROBIC_FRACTION
flow_variation = random.uniform(-FLOW_INIT_VARIATION, FLOW_INIT_VARIATION)
flow_mL_per_min = base_flow + base_flow * flow_variation
VR = pressure_mmHg / flow_mL_per_min
pCO2_mmHg = pO2_mmHg * CARBOGEN_CO2_TO_O2_FRACTION
bicarb_mM = (10**(pH - H_H_PK)) * (CO2_SOLUBILITY_mM_PER_MMHG * pCO2_mmHg) # Satisfies the H-H equation
fiO2 = pO2_mmHg / ATMOSPHERIC_PRESSURE_MMHG

def initial_big_state():
    big_state_list = [0] * BIG_STATE_DIMENSION

    big_state_list = [
        temperature_C,
        pressure_mmHg,
        round (flow_mL_per_min, 2),
        round (VR, 3),
        pH,
        pO2_mmHg,
        pvO2_mmHg,
        svO2,
        pCO2_mmHg,
        glucose_mM,
        insulin_mU,
        lactate_mM,
        hematocrit,
        round(bicarb_mM, 2),
        fiO2,
        dialysis_in_mL_per_min,
        hours,
        dialysis_out_mL_per_min,
        volume_mL
    ]
    
    return big_state_list

def get_scenario_type():
    return current_scenario
