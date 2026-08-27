import os
import sys
import traci
import numpy as np
from agent import DQNAgent 
import time

if 'SUMO_HOME' in os.environ:
     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
     sys.path.append(tools)
else:
     sys.exit("please declare environment variable 'SUMO_HOME'")

# --- Helper Functions ---
def get_state(traffic_light_id):
    incoming_lanes = traci.trafficlight.getControlledLanes(traffic_light_id)
    incoming_lanes = list(set(incoming_lanes)) 
    vehicle_counts = [traci.lane.getLastStepVehicleNumber(lane_id) for lane_id in incoming_lanes]
    return np.array(vehicle_counts)

def get_total_fuel_consumption(traffic_light_id):
    incoming_lanes = traci.trafficlight.getControlledLanes(traffic_light_id)
    incoming_lanes = list(set(incoming_lanes)) 
    total_fuel = 0
    for lane in incoming_lanes:
        total_fuel += traci.lane.getFuelConsumption(lane)
    return total_fuel

def calculate_reward(old_state, new_state, fuel_consumed):
    queue_reduction = np.sum(old_state) - np.sum(new_state)
    fuel_penalty = fuel_consumed * 0.00001 
    reward = queue_reduction - fuel_penalty
    return reward

def check_ambulance_location():
    # Check W1_to_J1
    v_ids = traci.edge.getLastStepVehicleIDs("W1_to_J1")
    for vid in v_ids:
        if traci.vehicle.getTypeID(vid) == "ambulance": return "APPROACHING_J1_WEST"
    # Check J1_to_J2
    v_ids = traci.edge.getLastStepVehicleIDs("J1_to_J2")
    for vid in v_ids:
        if traci.vehicle.getTypeID(vid) == "ambulance": return "BETWEEN_J1_J2"
    # Check J2_to_J4
    v_ids = traci.edge.getLastStepVehicleIDs("J2_to_J4")
    for vid in v_ids:
        if traci.vehicle.getTypeID(vid) == "ambulance": return "BETWEEN_J2_J4"
    return None

def explain_decision(step, agent_id, action, state):
    total_cars = np.sum(state)
    if action == 0: decision = "Green EW"
    else: decision = "Green NS"
    
    if total_cars > 10: explanation = f"High Congestion ({total_cars} cars). Clearing queue."
    elif total_cars < 5: explanation = "Light traffic. Maintaining flow."
    else: explanation = "Balanced traffic flow strategy."
    
    return f"{step},{agent_id},{decision},{explanation}\n"

# --- Main Simulation Logic ---
def run_simulation(gui=True, episodes=1):
    
    junction_ids = ["J1", "J2", "J3", "J4"]
    agents = {} 
    sumo_binary = "sumo-gui" if gui else "sumo"
    
    # FILES TO SAVE DATA
    log_file = open("decision_log.csv", "w")
    log_file.write("Step,Agent,Action,Reason\n")
    log_file.flush()
    
    stats_file = open("simulation_stats.csv", "w")
    stats_file.write("Step,Junction,QueueLength,FuelConsumption,Reward\n")
    stats_file.flush()
    
    for e in range(episodes):
        print(f"--- Starting Episode {e+1}/{episodes} ---")
        sumo_command = [sumo_binary, "-c", "quad.sumocfg"]
        try:
            traci.start(sumo_command)
        except traci.TraCIException as err:
            print(f"Error: {err}")
            break

        # Warm up
        for _ in range(50): traci.simulationStep()

        # Init Agents
        for j_id in junction_ids:
            if j_id not in agents:
                initial_state = get_state(j_id)
                agents[j_id] = DQNAgent(len(initial_state), 2)

        PHASE_EW_GREEN = 2 
        PHASE_NS_GREEN = 0
        
        current_states = {j_id: get_state(j_id) for j_id in junction_ids}
        step_counter = 0

        while traci.simulation.getMinExpectedNumber() > 0:
            step_counter += 1
            
            # 1. EMERGENCY CHECK
            ambulance_loc = check_ambulance_location()
            if ambulance_loc:
                if ambulance_loc in ["APPROACHING_J1_WEST", "BETWEEN_J1_J2"]:
                    traci.trafficlight.setPhase("J1", PHASE_EW_GREEN)
                    traci.trafficlight.setPhase("J2", PHASE_EW_GREEN)
                elif ambulance_loc == "BETWEEN_J2_J4":
                    traci.trafficlight.setPhase("J4", PHASE_NS_GREEN)
                    
                log_file.write(f"{step_counter},SYSTEM,OVERRIDE,Ambulance detected at {ambulance_loc}\n")
                log_file.flush()
                traci.simulationStep()
                continue 

            # 2. NORMAL AI LOGIC
            current_actions = {}
            for j_id in junction_ids:
                agent = agents[j_id]
                current_state = current_states[j_id]
                action = agent.act(current_state)
                current_actions[j_id] = action
                
                # Log Explanation
                if step_counter % 10 == 0:
                    log_file.write(explain_decision(step_counter, j_id, action, current_state))
                    log_file.flush()

                if action == 0:
                    traci.trafficlight.setPhase(j_id, PHASE_EW_GREEN)
                else:
                    traci.trafficlight.setPhase(j_id, PHASE_NS_GREEN)

            # Simulation Step
            fuel_usage = {j_id: 0 for j_id in junction_ids}
            for _ in range(5):
                if traci.simulation.getMinExpectedNumber() > 0:
                    traci.simulationStep()
                    for j_id in junction_ids:
                        fuel_usage[j_id] += get_total_fuel_consumption(j_id)
                else:
                    break

            # Learn & Save Stats
            for j_id in junction_ids:
                next_state = get_state(j_id)
                reward = calculate_reward(current_states[j_id], next_state, fuel_usage[j_id])
                
                # --- SAVE STATS TO CSV ---
                queue_len = np.sum(next_state)
                stats_file.write(f"{step_counter},{j_id},{queue_len},{fuel_usage[j_id]},{reward:.2f}\n")
                stats_file.flush()
                # -------------------------

                agents[j_id].remember(current_states[j_id], current_actions[j_id], reward, next_state)
                agents[j_id].replay(32)
                current_states[j_id] = next_state

        print(f"Episode {e+1} Finished.")
        traci.close()
        time.sleep(0.5)
    
    log_file.close()
    stats_file.close()
    print("✅ Data Saved: 'decision_log.csv' and 'simulation_stats.csv'")

if __name__ == "__main__":
    run_simulation(gui=True, episodes=1)