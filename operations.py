import init
import config
import random
import math

action = [0] * config.ACTION_DIMENSION
big_state = [0] * 19


def single_step(action_combo):
    """
    Execute a single simulation step.
    
    Args:
        action_combo: Tuple of (action_value, big_state)
    
    Returns:
        [big_state, newScoreCard, scoreValue]
    """

    scoreValue = 0
    # PARSE the actionCombo into action_value and big_state
    action_value = action_combo[0]
    #print(action_value)
    big_state = action_combo[1]
    #print(big_state)
      
    # DECODE  the action_value. Action value is a single int of the range to 19683 that is a 9-digit base
    # 3 encoding all action
    for a in range(config.ACTION_DIMENSION):
        action[a] = action_value % 3 
        if action[a] > 1: # turn the code into +/- 1 by converting all 2s to -1s THIS IS TRICKY: read it again
            action[a] = -1
        action_value = action_value // 3
        
    # ENFORCE the actions that can only go up
    # 4 items starting at 3: Glu, Ins, Bic, Vas
    for i in range(3, 6):
      if action[i] < 0:
          action[i] = 0

    # action is the 9-D action vector R:-1 to 1 
    # bigState is 18-D practical values in natural numbers (see config)
    # state is 6-D <temperatureCelsius, VR, pH, pvO2, glucosemMolar, insulinmUnits>
    # score_value = the calculated reward returned to the ML
    
    # INITIALIZE
    score_value = 0  
    score_card = [0] * init.BIG_STATE_DIMENSION # all the -2, -1, 0, 1, 2 scores for each state
    #state = big_state[:13] # Only use first 13 elements of the bigState vector are scoreable in the sim 
    local_step = [0] * config.ACTION_DIMENSION # This are the practical steps taken predicated on the action vector (initally steps are all 0)
    annex = [0] * (config.ACTION_DIMENSION - 13 - 1) # Added the fine non-scorable states to the state
    over = False

    # ***** THE EQUATIONS *****

    # The index callouts allow us to use the big state values explicitly
    def new_volume(big_state):
        #indices: volume 18. dialysis in 15, dialysis out 17
        volume = big_state[18] + big_state[15] - big_state[17]
        if volume < init.VOLUME_MINIMUM_ML:
            volume = init.VOLUME_MINIMUM_ML
        return volume # returns in mL

    def new_hematocrit(bigState, preStepVolumemL):
        #index: hematocrit 12
        hemoglobin_milliliter = preStepVolumemL * bigState[12] / 100
        hematocrit = hemoglobin_milliliter * 100 / bigState[18] # new volume adjustment
        # hemolysis calculation
        hematocrit = hematocrit - init.HEMATOCRIT_CHANGE_PER_HOUR * init.HOURS_PER_STEP
        if hematocrit < 0:
            hematocrit = 0
        return hematocrit # returns a percentage
    
    def glucose_millimole_consumed(bigState):
        glucose_mmole_consumed = (
            init.GLUCOSE_CONSUMPTION_MMOLE_PER_GRAM_HOUR
            * init.GRAFT_GRAMS
            * init.HOURS_PER_STEP
            * pow(2, (bigState[0] - 37) /10) # Q10 or Arrhenius equation 
            )
        return glucose_mmole_consumed

    def new_glucose(bigState, localStep, preStepVolumemL):
        glucose_millimole = 0
        # indices: glucose 9, temperature 0, volume 18
        try: # accomodate any possible divide by zero 
            glucose_millimole = (bigState[9] * preStepVolumemL / 1000) +  localStep[3]
        except: glucose_millimole = .001   
        # Glucose consumed by tissue
        glucose_millimole = glucose_millimole - glucose_millimole_consumed(bigState) 
        gluc_mM = glucose_millimole / (bigState[18] / 1000 ) 
        return gluc_mM

    def new_insulin(bigState, localStep):
        # indices: insulin 10
        insulinmIU = bigState[10] + localStep[4] - glucose_millimole_consumed(bigState) * init.INSULIN_mUNIT_PER_mMOLE_GLUCOSE
        return insulinmIU

    def new_lactate(bigState, preStepVolumemL):
        lactate_millimole = (bigState[11] * preStepVolumemL / 1000) + glucose_millimole_consumed(bigState) * init.LACTATES_PER_GLUCOSE
        return lactate_millimole / (bigState[18] / 1000 ) # millimolar relative to new volume
        
    def new_VR(bigState, vasod):
        specificVR = (.0107 - .0002 * big_state[0]) * init.VR_ORGAN_FACTOR
    
        # slight 1-time 90% reduction on vasodilator dose
        VR = specificVR * init.GRAFT_GRAMS * (1 - .1 * vasod) * random.uniform(1, init.VR_STOCHASTIC_FACTOR)
        return VR

    def new_flow(bigState):
        # big_state[1] = pressure
        try:
            return bigState[1]/big_state[3]
        except:
            return 99

    def new_bicarb(bigState, localStep, preStepVolumemL):
        # bicarb mM = bicarb dialysis * dialysis flow + perfusate bicarb * perfusion flow) / (dialysis + perfusion flow)
        flow_weighted_mM = (init.BICARB_IN_KREBS_mM * bigState[15] + bigState[13] * bigState[2]) / (bigState[15] + bigState[2])
        bicarb_millimole = (flow_weighted_mM * preStepVolumemL / 1000) +  localStep[5]
        bicarb_millimolar = bicarb_millimole / (bigState[18] / 1000 )
        return bicarb_millimolar

    def new_pH(bigState): 
        # bigState 13 is bicarb, bigState [8] is CO2
        localpH = init.H_H_PK + math.log((bigState[13]/(init.CO2_SOLUBILITY_mM_PER_MMHG * bigState[8])),10)
        return localpH

    def new_svO2(bigState):
        # See Serianni on Hill, human, 37C, pH 7.4
        # bigState[5] is pO2, bigState[12] is Hct, bigState[2] is flow, bigState[18] is volume(mL), bigState[12] is hemotocrit
        saO2 = (pow((.13534 * bigState[5]),2.62)) / ((pow((.13534 * bigState[5]),2.62)) + 27.4)
        Hgb_g_per_dL = 0.34 * bigState[12]
        CaO2_mL_per_dL = (saO2 * Hgb_g_per_dL * 1.36) + (.0031 * bigState[5])
        CaO2_mL_per_L = CaO2_mL_per_dL * 10
        CaO2_mM = CaO2_mL_per_L / init.MOLAR_VOLUME_STP_GAS_L
        O2_rate_in_millimole_per_min = CaO2_mM * bigState[2] / 1000
        O2_millimole_consumed_per_minute = (glucose_millimole_consumed(bigState) / init.MINUTES_PER_STEP) * (init.AEROBIC_FRACTION * init.OXYGENS_PER_GLUCOSE)
        O2_rate_out_millimole_per_min = O2_rate_in_millimole_per_min - O2_millimole_consumed_per_minute
        CvO2_mM = (O2_rate_out_millimole_per_min / bigState[2]) * 1000 # the factor 1000 converts mL to L
        CvO2_mL_per_dL = CvO2_mM * init.MOLAR_VOLUME_STP_GAS_L / 10
        v_Hgb_g_per_dL = 0.34 * bigState[12]
        svO2 = (CvO2_mL_per_dL / ((v_Hgb_g_per_dL * 1.34)+ (.0031 / 2)))
        if svO2 > 1:
            svO2 = 1
        return svO2, CvO2_mL_per_dL

# Action Vector, 9D, [0 to 8]. This is a single number encoding 8 ternary components (0 to 3 recoded at Step() to -1 to 1)
# 0. Temperature C (simulated in 1 C steps)
# 1. Pressure mmHg (MAP) (simulated in 5 mmHg steps)
# 2. FiO2 (adjustable in 0.1 increments) (air mixed with carbogen)
# 3. Glucose millimoles (2.776 mmol per mL unit injection)
# 4. Insulin mU (200 mU per mL injection)
# 5. Bicarb millimoles (1 mmol per mL unit injection)
# 6. Vasodilator (Papavarine 1.2 mg per mL)
# 7. Dialysis in mL/min (up to 60 mL per minute)
# 8. Dialysis out mL/min (up to 60 mL per minute)


    # ***** THE SIMULATION *****    

    # Grab pre-step values for some initial calculations

    pre_step_volume_mL = big_state[18]

    # Convert <action> into <practical steps>
    for x in range(0, config.ACTION_DIMENSION):
        local_step[x] = action[x] * config.action_step[x] # Here is where the +/- 1 action turns into physical units                

    # Apply the steps below in order as some laters depend on some earliers:
    # Step big states while limiting to keep them within controlled physical Max/Min bounds
    temperature_C = big_state[0] + local_step[0]
    if temperature_C > config.action_limit_max[0]:
      temperature_C = config.action_limit_max[0]
    if temperature_C < config.action_limit_min[0]:
      temperature_C = config.action_limit_min[0]

    pressure_mmHg = big_state[1] + local_step[1]
    if pressure_mmHg > config.action_limit_max[1]:
      pressure_mmHg = config.action_limit_max[1]
    if pressure_mmHg < config.action_limit_min[1]:
      pressure_mmHg = config.action_limit_min[1]

    fiO2 = big_state[14] + local_step[2]
    if fiO2 > config.action_limit_max[2]:
      fiO2 = config.action_limit_max[2]
    if fiO2 < config.action_limit_min[2]:
      fiO2 = config.action_limit_min[2]

    dialysis_in_mL_per_min = big_state[15] + local_step[7]
    if dialysis_in_mL_per_min > config.action_limit_max[7]:
      dialysis_in_mL_per_min = config.action_limit_max[7]
    if dialysis_in_mL_per_min < config.action_limit_min[7]:
      dialysis_in_mL_per_min = config.action_limit_min[7]

    dialysis_out_mL_per_min = big_state[17] + local_step[8]
    if dialysis_out_mL_per_min > config.action_limit_max[8]:
      dialysis_out_mL_per_min = config.action_limit_max[8]
    if dialysis_out_mL_per_min < config.action_limit_min[8]:
      dialysis_out_mL_per_min = config.action_limit_min[8]
         
    # Grab pre-step values for some initial calculations
    pre_step_volume_mL = big_state[18]

    # We start repopulating big state one by one, right away
    big_state[0] = temperature_C
    big_state[1] = pressure_mmHg
    big_state[14] = fiO2
    big_state[15] = dialysis_in_mL_per_min
    big_state[17] = dialysis_out_mL_per_min

    # Step the volume and assign the value to big state [18]
    # This calculation precedes any concentrartion calculations
    big_state[18] = new_volume(big_state)
    
    # Glucose
    big_state[9] = new_glucose(big_state, local_step, pre_step_volume_mL) #Glucose
    glucose_mM = big_state[9]

    # Lactate
    big_state[11] = new_lactate(big_state, pre_step_volume_mL)
    lactate_mM = big_state[11]

    # Insulin 
    big_state[10] = new_insulin(big_state, local_step)
    insulin_mU = big_state[10]

    # Vasodilator
    vasodilator = local_step[6] # vasodilartor is not a state variable - it gets used up upon infusion

    # Hematocrit
    big_state[12] = new_hematocrit(big_state, pre_step_volume_mL)
    hematocrit = big_state[12]

    # Hours
    big_state[16] = big_state[16] + init.HOURS_PER_STEP
    hours = big_state[16]

    # pO2
    big_state[5] = fiO2 * init.ATMOSPHERIC_PRESSURE_MMHG # Full saturation of perfusate to oxygenator pO2
    pO2_mmHg = big_state[5]

    # pCO2
    big_state[8] = pO2_mmHg * init.CARBOGEN_CO2_TO_O2_FRACTION #Full saturation assumed same as O2
    pCO2_mmHg = big_state[8]

    # VR (a function of time, temperature, graft mass, vasodilator)
    big_state[3] = new_VR(big_state,vasodilator)
    VR = big_state[3]

    # Flow
    big_state[2] = new_flow(big_state)
    flow_mL_per_min = big_state[2]

    # Bicarb
    big_state[13] = new_bicarb(big_state, local_step, pre_step_volume_mL)
    bicarb_mM = big_state[13]

    # pH
    big_state[4] = new_pH(big_state)
    pH = big_state[4]

    # svO2
    big_state[7], cvO2_mL_per_dL = new_svO2(big_state)
    svO2 = big_state[7]

    # pvO2
    big_state[6] = (cvO2_mL_per_dL - (svO2 * hematocrit * 0.34 * 1.36)) / .0031
    if big_state[6] < 0:
        big_state[6] = 0
    pvO2_mmHg = big_state[6]

    # Sigmoid correction to svO2
    if pvO2_mmHg > 100:
        svO2 = 1
    elif pvO2_mmHg > 60:
        svO2 = .9 + .25 * (pvO2_mmHg - 60)/ 100
    elif pvO2_mmHg > 40:
        svO2 = .8 + .5 * (pvO2_mmHg - 40) / 100            
    elif pvO2_mmHg > 0:
        svO2 = pvO2_mmHg * 2 / 100              
    else:
        svO2 = 0
    if svO2 < 0:
        svO2 = 0 # Cant go below zero
    big_state[7] = svO2 # revise big_state[7] to match sigmoid svO2 and under-zero prevention

    #print(big_state)

    # ***** THE SCORING *****

    # Generate Score. Note that test for done occurs in the calling progran
    scalarSum = 0
    newScoreCard = [0] * 6
    scoreCard = [0] * 14 # The first 14 big state values are scored Temperature (0) to Bicarb (13_

    if config.STATE_TYPE == "discrete":
        for y in range(0, 14):
          if big_state[y] < config.criticalDepletion[y]:
              scoreCard[y] = -2
          elif big_state[y] < config.depletion[y]:
              scoreCard[y] = -1
          elif big_state[y] > config.criticalExcess[y]:
              scoreCard[y] = 2
          elif big_state[y] > config.excess[y]:
              scoreCard[y] = 1
          else:
              scoreCard[y] = 0

   
    for z in range (0,6):
        newScoreCard[0] = scoreCard[0] # Temperature
        newScoreCard[1] = scoreCard[3] # VR
        newScoreCard[2] = scoreCard[4] # pH
        newScoreCard[3] = scoreCard[6] # pvO2
        newScoreCard[4] = scoreCard[9] # Glucose
        newScoreCard[5] = scoreCard[10] # Insulin

        scalarSum = scalarSum + pow(newScoreCard[z],2)

    scoreValue = scalarSum
    scoreValue = (127 - 10 * scoreValue) # all zeros give you a 127

    # Perfect score if done
    if hours >= 24:
        scoreValue = 255

    # Any out of range by 2 levels then big fail
    bad_score_flag = 0
    for x in range (0,6):
            if abs(newScoreCard[x]) == 2:
                bad_score_flag = 1 # any out of bounds negates a good score
    if bad_score_flag == 1:
        scoreValue = -255
    
   
    # ***** THE OUTPUT *****

    # Finally record the incoming action, score, and state into the log file 
    actionString = [str(i) for i in action]
    scoreString = [str(j) for j in newScoreCard]
    stateString = [str(round(k,2)) for k in big_state]


    if hours == 1:
        outstring = "\n" + str(hours) + "\t " + ", ".join(actionString) + "; " + ", ".join(scoreString) + "; " + str(round(scoreValue,2)) + "; " + ", ".join(stateString) + "\n"
  
    else:
        outstring = str(hours) + "\t " + ", ".join(actionString) + "; " + ", ".join(scoreString) + "; " + str(round(scoreValue,2)) + "; " + ", ".join(stateString) + "\n"

    with open (config.filePath + str(config.myuuid) + " " + config.SCORE_TYPE + ".txt",'a') as myfile:
        myfile.write (outstring)

    out24List =  outstring

    if (hours == 24) and (scoreValue > -255):
        with open (config.filePath + str(config.myuuid) + " " + config.SCORE_TYPE + "24s.txt",'a') as my24file:
            for x in range (0,25):
                my24file.write (out24List[x])
                #print (x, out24List[x])

    # ***** RETURN VALUE TO THE AGENT *****
         
    roundState = [ round(elem, 2) for elem in big_state ]
    big_state = roundState

    answer = [big_state, newScoreCard, scoreValue]           

    return answer





        


    
        
        

