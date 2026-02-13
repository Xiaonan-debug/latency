# 4.17.25   CONFIDENTIAL
#
# This script estimates the perfusate chemistry and hemodynaamic response of a graft to perfusion conditions.
# Adapted to simulate perfusion of a single eye

# Revision 10 Eye: 4.16.25 (c) Brassil 2023
# Added SCORE_TYPE centerLineHours
# Added causeOfDeath output to filename+Final2s
# Added reduced mass for the eye and volume of perfusate
# Removed requirement to stop at out of bounds so that the agent may better learn

# Need to check all the chamistry and make sure all teh calculations reverence actual perfusate volume
# i.e., dont assume 1 liter

import math
import random
import uuid
import os


# Hard coded as follows:
# 1 hour per step

# Items that may be set
SCORE_TYPE = "vectorLength"
# Score Types can be pH, pO2, glucose, PFI, vectorLength, hours, hyperHours, consequentialHours, centerLineHours
CENTERLINE_SCORE_WEIGHT = [0, .03, .03, .22, .23, .49] # Based on probability that the enumerated state would predicate failure
STATE_TYPE = "discrete"
# STATE_TYPE = "unitFloat" # This is an alternate state type that results in a 0 to 1 float value for each state element
myuuid = uuid.uuid4()   # Make a unique string foor teh simulation log filename
filePath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results") + os.sep

HOURLY_VASCULAR_RESISTANCE_CHANGE = 1 #v10.1 Starts at 70 and increases by 1 every hour to become 94 at 24 hours -> 0.74 mL/min at 24h EMPIRICAL
VASCULAR_RESISTANCE_STOCHASTIC_FACTOR = 0.25 # VR can change stochastically an additional +25%
VASCULAR_RESISTANCE_DELTA_T_BASE = .981 # VR decreases almost 2% per degree increase EMPIRICAL
PERFUSATE_LITERS = 0.15
GRAFT_GRAMS = 0.326 # Normal estimate Werkmeister
ANAEROBIC_METABOLIC_FRACTION = 0.7
aerobicMetabolicFraction = 1 - ANAEROBIC_METABOLIC_FRACTION
hematocritInitial = 30 #v10.1 matches Env
transfusionHours = 12
HOURLY_HEMATOCRIT_CHANGE = 1   # Usually it is 1
GLUCOSE_CONSUMPTION_MMOLE = .0043 # From our 2021 and 2022 experimental data (per minute per 100g)
INITIAL_PRESSURE_MMHG = 70
ONE_LPM_ARTERY_GAS_PRESSURE_FRACTION = .5 # Empirical. Could be as low as 0.2 v10.1 still good

# CHECK
HOURLY_CO2_LOAD_MMHG = 13.4 # Empirical at 0 lpm sweep gas. declines to 0 at 1lpm v10.1 Not used
PCO2_EQULIBRIUM_AT_ONELPM = 38 # Empirical v10.1

# Initialize constants
GLUCOSE_MOLECULAR_WT = 180.156
SOLUBILITY_CO2 = .03 # mM/mmHg
PK = 6.1 # Used in equation to convert CO2 and bicarb to pH
LACTATES_PER_GLUCOSE = 2
OXYGENS_PER_GLUCOSE = 6
CO2S_PER_GLUCOSE = 6
INSULIN_PER_MMOLE_GLUCOSE = 19.5 # mUnits
ACTION_DIMENSION = 7

# Action Vector, 7D, [0 to 6]. This is a single number encoding 8 ternary components (0 to 3 recoded at Step() to -1 to 1)
# 0. Temperature C
# 1. Gas Flow lpm
# 2. Gas Richness %
# 3. Glucose mM
# 4. Insulin mU
# 5. Bicarb mM
# 6. Vasodilator (mL concentration TBD)

# Action setup
actionStep = [1, .1, .1, 1, 20, 1, 1] # This is the standard step
actionLimitMax = [38, 1, 1]  # Changed from 40 to 38 for temperature
actionLimitMin = [34, 0, 0]  # Changed from 20 to 34 for temperature
actionValue = 0
action = [0] * ACTION_DIMENSION
out24List =[""]* 25

# Initialize endpoint
causeOfDeath = 99 # Still OK

# Big State Vector, 18D [0 to 17].
# 0. Temperature C
# 1. Pressure mmHg
# 2. Flow mL/min
# 3. PFI
# 4. pH
# 5. pO2 mmHg
# 6. pvO2 mmHg
# 7. svO2 fraction
# 8. pvCO2 mmHg
# 9. Glucose mM
# 10. Insulin mU
# 11. Lactate mM
# 12. Hematocrit %
# 13. Bicarb mM
# 14. Gas Flow lpm
# 15. Gas Richness %
# 16. Hours

# State scoring limits: L HH, H, L, LL (Only 13 of the 17 states are scored)
criticalDepletion = [34, 20, 0, .1, 6.9, 70, 0, 0, 0, 2, 1, 0, 1]  # Changed from 10 to 34 for temperature
depletion = [35, 40, .1, .5, 7.1, 100, 20, .3, 20, 3, 15, 0, 10]  # Changed from 19 to 35 for temperature
excess = [37, 100, 150, 20, 7.5, 600, 500, .9, 50, 12, 45, 15, 100]  # Changed from 38 to 37 for temperature
criticalExcess = [38, 120, 3, 50, 7.7, 700, 700, 1, 60, 33, 80, 30, 100]  # Changed from 41 to 38 for temperature

stateIndices = [0,3,4,6,9,10]
stateExcess = [criticalExcess[i] for i in stateIndices]
stateDeplete = [criticalDepletion[i] for i in stateIndices]

# Substitute the linearized pH as cH in millimoles (reversing excess and deplete due to - exponent)
# stateDeplete[2] = pow(10,-criticalExcess[4])*1000
# stateExcess[2] = pow(10,-criticalDepletion[4])*1000

stateCentroid = [float (x + y) / 2 for x, y in zip (stateDeplete,stateExcess)]
stateHalfWidth = [(x - y) for x,y in zip (stateCentroid,stateDeplete)]


#------ FUNCTION ------
def SingleStep(actionCombo):

    # Parse the actionCombo into actionValue and bigState
    actionValue = actionCombo[0]
    bigState = actionCombo[1]

    # Sort out the initial hemodynamics for this step
    pressuremmHg = bigState[1]
    perfusionFlowIndex = bigState[3]

    # CHECK
    vascularResistance = 100 / (perfusionFlowIndex * GRAFT_GRAMS)
    perfusionFlowmLpm = pressuremmHg / vascularResistance
    status = "ok"

      
    # Decode the actionValue
    action_mapping = {0: -1, 1: -0.5, 2: 0, 3: 0.5, 4: 1}
    
    for a in range(ACTION_DIMENSION):
        encoded_action = actionValue % 5  # Changed from 3 to 5 for 5-level actions
        action[a] = action_mapping[encoded_action]
        actionValue = actionValue // 5  # Changed from 3 to 5 for 5-level actions

    # print(action)
        
    # Enforce the actions that can only go up
    for i in range(3, ACTION_DIMENSION):
      if action[i] < 0:
          action[i] = 0

    # for constant P, ignore Flow actions, because flow is an output variable 5.5.23
    # action[1] = 0 

    # action is the 8-D action vector R:-1 to 1 <temp, flow, gasLPM, gasRich%, gluc, insul, NaCO3, vasodilator>
    # bigState is 17-D practical values = state.append(NaCO3, gasLPM, gasRich%, hours)
    # state is 6-D <temperatureCelsius, perfusionFlowIndex, pH, pvO2, glucosemMolar, insulinmUnits>

    # Set local variables to initial zeros, and trim bigState to get state       
    scoreValue = 0
    scoreCard = [0] * 13
    state = bigState[:13] # Only use first 13 elements of the bigState vector
    localStep = [0] * 8 # This are the practical steps taken predicated on the action vector (initally steps are all 0)
    annex = [0] * 3 # Added to the state
    over = False
    causeOfDeath = 99 # Still OK = Not dead yet


    # ----- PERFUSION AND PHYSIOLOGY FUNCTIONS -----
    # ----- A. HEMATOCRIT -----
    # Calculate new Hct on the basis of hourly "corrosion"
    def NewHct(Hct):
        Hct = Hct - HOURLY_HEMATOCRIT_CHANGE
        if Hct < 0:
            Hct = 0
        return Hct

    # ----- B. GLUCOSE -----
    # Calculate glucose level based on temperature derated consumption
    def ConsumedGlucose(mass, temperature):
        glumMolesConsumed = GLUCOSE_CONSUMPTION_MMOLE * (mass / 100) * 60 * pow(2,(temperature - 37) / 10)                                                                            
        return glumMolesConsumed

    # ----- C. LACTATE -----
    def NewLactate(glumMolesCons):
        lacmMoles = glumMolesCons * ANAEROBIC_METABOLIC_FRACTION * LACTATES_PER_GLUCOSE
        return lacmMoles

    # ----- D. ARTERIAL OXYGEN -----

    # CHECK : New gas mixer
    # D1. Calculate new pO2 - assumes partial pressure of oxygen equilbrates between gas and perfusate i.e oxygenator has significantly excess capacity
    def NewPO2(gasFlow, richness):
        FiO2 = (.95 * richness) + (.2 * (1 - richness))
        gasPO2 = FiO2 * 760
        OneLpmArtPO2 = gasPO2 * ONE_LPM_ARTERY_GAS_PRESSURE_FRACTION
        pO2 = OneLpmArtPO2 * gasFlow # Max concentration of gas in blood at 1 lpm, declinining linearly to 0 at 0 lpm
        return pO2

    # D2. Calculate oxygen saturation as a function of pO2(see Serianni on Hill, human, 37C, pH 7.4)
    def SaturationO2(pO2):
        SaOxygen = (pow((.13534 * pO2),2.62)) / ((pow((.13534 * pO2),2.62)) + 27.4)
        return SaOxygen

    # D3. Calculate oxygen content in blood considering SaO2, pO2 and Hct
    def ConcentrationaO2(pO2, Hct, SaOxygen):
        HbGperdL = 0.34 * Hct 
        CaOxygen = (SaOxygen * HbGperdL * 1.36) + ( .0031 * pO2) # in mL/dL
        return CaOxygen

    # ----- E. VENOUS OXYGEN -----
    # E1. Calculate immediate venous oxygen concentration
    def ConcentrationvO2 (o2RateOut):
        CvOxygen = o2RateOut * 1000 / (perfusionFlowmLpm) # This is millimolar
        #print(CvOxygen)
        if CvOxygen < 0:
            CvOxygen = 0   
        return CvOxygen

    # E2. Estimate immediate venous oxygen saturation (see Madan)
    # If pO2 is below 40 then simply multiply pO2 by 2 to estimate SvO2
    # Using this relationship we can estimate SvO2 from CvO2 assuming SvO2 < 80%
    # Calculate the inverse Concentration equation
    def VsaturationO2(CvOxygen, Hct):
        HbGperdL = 0.34 * Hct
        SvOxygen = (CvOxygen /((HbGperdL * 1.34) + (.0031 / 2)))
        if SvOxygen > 1:
            SvOxygen = 1
        return SvOxygen


    # ----- F. HEMODYNAMICS -----

    # CHECK : revise to new empirical
    # Calculate new vascular resistance
    def NewVR(oldVR, temperature, vasodilator):
        # Step 1: Age the VR unless vasodilator is given in which case keep VR the same. Then allow VR to change stochastically 25% of hourly change
        localVR = oldVR + HOURLY_VASCULAR_RESISTANCE_CHANGE * int(not(vasodilator)) * random.uniform(1, VASCULAR_RESISTANCE_STOCHASTIC_FACTOR)
        # Step 2: Adjust VR due to temperature
        localVR = localVR * pow(VASCULAR_RESISTANCE_DELTA_T_BASE,(lastTemperature - temperature)) # VR = VR * base^(-deltaT) as T increases VR decreases
        return localVR


    # ----- G. ARTERIAL CO2 -----
    def NewPCO2(gasFlow, richness):
        FiCO2 = .05 * richness
        pCO2 = FiCO2 * 760 + PCO2_EQULIBRIUM_AT_ONELPM # Different frrom oxygen: always blow-off excess CO2 and get back to FiCO2 level
        return pCO2

    # ----- I. VENOUS PH -----
    def NewpH(bcb,pCO2):
        localpH = PK + math.log((bcb/(SOLUBILITY_CO2 * pCO2)),10)
        return localpH

    # ----- MAIN SIMULATION-----

    # Convert <action> into <practical steps>
    for x in range(0, ACTION_DIMENSION):
        localStep[x] = action[x] * actionStep[x] # practical step = elementwise R:-1 to 1 <action> * <constant practical action step>

    # Assign values to each of the practical elements of <bigState> based on <localStep>                  
    # Limit the controls to keep them within Max/Min bounds
    # Results in 8 practical sctions resilting from the 8-D action vector
    lastTemperature = bigState[0]
    temperatureCelsius = bigState[0] + localStep[0]
    if temperatureCelsius > actionLimitMax[0]:
      temperatureCelsius = actionLimitMax[0]
    if temperatureCelsius < actionLimitMin[0]:
      temperatureCelsius = actionLimitMin[0]

    gasFlowLPM = bigState[14] + localStep[1]
    if gasFlowLPM > actionLimitMax[1]:
      gasFlowLPM = actionLimitMax[1]
    if gasFlowLPM < actionLimitMin[1]:
      gasFlowLPM = actionLimitMin[1]          

    gasRichness = bigState[15] + localStep[2]
    if gasRichness > actionLimitMax[2]:
      gasRichness = actionLimitMax[2]
    if gasRichness < actionLimitMin[2]:
      gasRichness = actionLimitMin[2]
              
    # Increment chemical concentrations responsive to infusion
    glucosemMoles = bigState[9] * PERFUSATE_LITERS + localStep[3] #v10.1
    insulinmUnits = bigState[10] + localStep[4]
    bicarbMmoles = bigState[13] + localStep[5]
    vasodilator = localStep[6]
    hematocrit = bigState[12]
    lactatemMoles = bigState[11] * PERFUSATE_LITERS
    hours = bigState[16] + 1

    # Apply the hourly calculated adjustments


    # F. HEMODYNAMICS (VR, PFI, FLOW)
    # Start with hemodynamics because the new flow determines A-V biochemical concentration changes     
    pO2mmHg = NewPO2(gasFlowLPM,gasRichness)
    vascularResistance = NewVR(vascularResistance, temperatureCelsius, vasodilator)
    perfusionFlowmLpm = pressuremmHg / vascularResistance # Flow now responds to VR at constant pressure
    perfusionFlowIndex = 100 / (vascularResistance * GRAFT_GRAMS)

   
    # A. HEMATOCRIT 
    # Hematocrit declines monotonically, independently 
    hematocrit = NewHct(hematocrit)
    if hours == transfusionHours:
        hematocrit = hematocritInitial

    # B. GLUCOSE AND INSULIN
    glucosemMolesConsumed = ConsumedGlucose(GRAFT_GRAMS, temperatureCelsius)
    glucosemMoles = glucosemMoles - glucosemMolesConsumed
    if glucosemMoles < 0:
      glucosemMoles = 0
    glucosemMolar = glucosemMoles / PERFUSATE_LITERS
    insulinmUnits = insulinmUnits - glucosemMolesConsumed * INSULIN_PER_MMOLE_GLUCOSE

    # C. LACTATE
    lactatemMoles = lactatemMoles + NewLactate(glucosemMolesConsumed)
    lactatemMolar = lactatemMoles / PERFUSATE_LITERS

    # D. ARTERIAL OXYGEN
    # Arterial Oxygen (Determined by FiO2 and gas flow conditions in the oxygenator, i.e., oxygenator capacity >> O2 demand)
    saO2 = SaturationO2(pO2mmHg)
    caO2 = ConcentrationaO2(pO2mmHg, hematocrit, saO2)
    o2mLperLiter = caO2 * 10 # convert fom dL to L
    o2milliMolar = o2mLperLiter / 22.4 # Universal gas law
    o2RateInmMolePerMin = o2milliMolar * perfusionFlowmLpm / 1000
    o2DemandmMolePerMin = (glucosemMolesConsumed / 60) * (1 - ANAEROBIC_METABOLIC_FRACTION) * OXYGENS_PER_GLUCOSE
    o2DemandmLperMinPer100g = o2DemandmMolePerMin * 22.4 *100 / GRAFT_GRAMS
    o2RateOutmMolePerMin = o2RateInmMolePerMin - o2DemandmMolePerMin
    
    #CHECK : This is a patch. Variable doesnt seem to do anything
    #fractionO2removedByGraft = o2DemandmMolePerMin / o2RateInmMolePerMin


    # E. VENOUS OXYGEN
    cvO2 = ConcentrationvO2(o2RateOutmMolePerMin)
    cvO2mLperdL = cvO2 * 22.4 / 10
    svO2 = VsaturationO2(cvO2mLperdL, hematocrit)
    pvO2 = (cvO2mLperdL - (svO2 * hematocrit * .34 * 1.36))/.0031

    # Saturation correction to sigmoid
    if pvO2 > 100:
      svO2 = 1
    elif pvO2 > 60:
      svO2 = .9 + .25 * (pvO2 - 60)/ 100
    elif pvO2 > 40:
      svO2 = .8 + .5 * (pvO2 - 40) / 100            
    elif pvO2 > 0:
      svO2 = pvO2 * 2 / 100              
    else:
      svO2 = 0
                            
    # G. ARTERIAL CO2
    pCO2mmHg = NewPCO2(gasFlowLPM, gasRichness)
    # Empirical alternte for physiology below: pCO2mmHg = pCO2mmHg + HOURLY_CO2_LOAD_MMHG - HOURLY_CO2_LOAD_MMHG * bigState[14]
    co2concentrationmMolar = pCO2mmHg * SOLUBILITY_CO2
    co2rateInmMolesPerMin = co2concentrationmMolar * perfusionFlowmLpm / 1000
    co2productionmMolePerMin = CO2S_PER_GLUCOSE * glucosemMolesConsumed / 60
    co2RateOutmMolePerMin = co2productionmMolePerMin + co2rateInmMolesPerMin

    # H. VENOUS CO2
    cvCO2mMolar = 1000 * co2RateOutmMolePerMin / perfusionFlowmLpm
    pvCO2mmHg = cvCO2mMolar / SOLUBILITY_CO2

    # I. VENOUS PH
    pH = NewpH(bicarbMmoles, pCO2mmHg)


    # Update the state
    state = [temperatureCelsius, pressuremmHg, perfusionFlowmLpm, perfusionFlowIndex, pH, pO2mmHg, pvO2, svO2, pvCO2mmHg, glucosemMolar, insulinmUnits, lactatemMolar, hematocrit]
    annex = [bicarbMmoles, gasFlowLPM, gasRichness, hours]
    bigState = state + annex
    
    # Generate Score. Note that test for done occurs in the calling progran
    scalarSum = 0
    newScoreCard = [0] * 6

    if True: # For testing
    # if STATE_TYPE == "unitFloat":
        for w in range (0,13):
            scoreCard[w] = (state[w] - criticalDepletion[w])/(criticalExcess[w] - criticalDepletion[w])
            if scoreCard[w] < 0:
                scoreCard[w] = 0   # ReLU thinking            

    # print (scoreCard)   # For testing

    if STATE_TYPE == "discrete":
        for y in range(0, 13):
          if state[y] < criticalDepletion[y]:
              scoreCard[y] = -2
          elif state[y] < depletion[y]:
              scoreCard[y] = -1
          elif state[y] > criticalExcess[y]:
              scoreCard[y] = 2
          elif state[y] > excess[y]:
              scoreCard[y] = 1
          else:
              scoreCard[y] = 0

          scalarSum = scalarSum + pow(scoreCard[y],2)

    # Scoring - 8 bit score requires that we bracket the scores between 0 to 255
    # All scores are initially lower-the-better then REVERSED by multiplyimg by -1

    if SCORE_TYPE == "pH": #CPk 
        pHScore = (abs(pH - 7.3 ) / .2 ) * 120 # allocate 120 of 255 points to pH
        scoreValue = int(-hours * 5 + pHScore) # allocate 120 of 255 points to hours

    if SCORE_TYPE == "glucose":   #CPk
        if glucosemMolar > 4 or glucosemMolar < 8:
            glucoseScore = (abs(glucosemMolar - 6) / 2) * 120
        else:
            glucoseScore = 0
        scoreValue = int((24 - hours) * 5 + glucoseScore) # allocate 120 of 255 points to hours

    if SCORE_TYPE == "pO2": #CPk
        pO2Score = (abs(pO2mmHg - 300) / 200) * 120
        scoreValue = int((24 - hours) * 5 + pO2Score) # allocate 120 of 255 points to hours

    if SCORE_TYPE == "PFI": #More is better Sept 1, 2023 version changed this to a positive number
        scoreValue = int(hours * 5 + int(perfusionFlowIndex * 120 / .33333))
        # scoreValue = int((24 - hours) * 5 - int(perfusionFlowIndex * 33.75)) Previous 

    if SCORE_TYPE == "vectorLength":  #CPk
        scoreValue = int(pow(scalarSum, .5) * 30 + (24 - hours) * 5)

    if SCORE_TYPE == "hours": #reward is soley based on hours elapsed
        scoreValue = hours * -10 #0 to 240

    if SCORE_TYPE == "hyperHours":
        scoreValue = int(pow (hours, 1.74) * -1) #24^1.74 = 252.1
         
    if SCORE_TYPE != "PFI":
        scoreValue = -1 * scoreValue # Make the higher number more rewarding:i.e., REVERSED          

    # littleState is the actual values for th 6-D state
    littleState = [temperatureCelsius, perfusionFlowIndex, pH, pvO2, glucosemMolar, insulinmUnits]
    halfClosenessFromCentroid = [1 - (abs(x - y) / z) for x, y, z in zip (littleState, stateCentroid, stateHalfWidth)] 
    #print(halfClosenessFromCentroid)
    #exit()




    # state is the 13-D practical value vector of <temp, press, flow, VR, pH, pO2, pvO2, SvO2, pvCO2, gluc, insul, lact, Hct>
    # scoreCard is the 13-D observation vector R:-2 to 2, same order as state
    # scoreValue is the length of the score vector (root sum of the squares) * -1 to make higher better
    # Convert scoreCard from original 13D to new 6D

    newScoreCard[0] = scoreCard[0]
    newScoreCard[1] = scoreCard[3]
    newScoreCard[2] = scoreCard[4]
    newScoreCard[3] = scoreCard[6]
    newScoreCard[4] = scoreCard[9]
    newScoreCard[5] = scoreCard[10]

    if SCORE_TYPE == "consequentialHours":
        scoreValue = hours

    if hours == 24:
        scoreValue = 255

    for x in range (0,6):
            if abs(newScoreCard[x]) == 2:
                scoreValue = -255

    if scoreValue > 255:
        scoreValue = 255  

    if SCORE_TYPE == "centerLineHours":
        commonWeight = 60 # score will be 60 if all closenesses = 1
        scoreValue = hours + sum(x * y for x, y in zip(halfClosenessFromCentroid, CENTERLINE_SCORE_WEIGHT)) * commonWeight
        # print(scoreValue)

    if hours == 24:
        scoreValue = 255

    for x in range (0,6):
            if abs(newScoreCard[x]) == 2:
                causeOfDeath = x
                scoreValue = -255

    if scoreValue > 255:
        scoreValue = 255  


    # print(hours, scoreValue, sum(halfClosenessFromCentroid))
    
    # Finally record the incoming action, score, and state into the log file 
    actionString = [str(i) for i in action]
    scoreString = [str(j) for j in newScoreCard]
    stateString = [str(round(k,2)) for k in state] 


    if hours == 1:
        outstring = "\n" + str(hours) + "\t " + ", ".join(actionString) + "; " + ", ".join(scoreString) + "; " + str(round(scoreValue,2)) + "; " + ", ".join(stateString) + "\n"
  
    else:
        outstring = str(hours) + "\t " + ", ".join(actionString) + "; " + ", ".join(scoreString) + "; " + str(round(scoreValue,2)) + "; " + ", ".join(stateString) + "\n"

    with open (filePath + str(myuuid) + " " + SCORE_TYPE + ".txt",'a') as myfile:
        myfile.write (outstring)

    out24List[hours] =  outstring

    if (hours == 24) and (scoreValue > -255):
        with open (filePath + str(myuuid) + " " + SCORE_TYPE + "24s.txt",'a') as my24file:
            for x in range (0,25):
                my24file.write (out24List[x])
                #print (x, out24List[x])

    if (scoreValue == -255):
        with open (filePath + str(myuuid) + " " + SCORE_TYPE + "Final2s.txt",'a') as Final2sfile:
            for x in range ((hours - 1),(hours + 1)):
                Final2sfile.write (str(round(causeOfDeath, 0)) + "; " + out24List[x])
                # print (x, out24List[x])           
            Final2sfile.write ("\n")
            
    roundState = [ round(elem, 2) for elem in bigState ]
    bigState = roundState

   
    answer = [bigState, newScoreCard, scoreValue]           

    return answer





          

            
