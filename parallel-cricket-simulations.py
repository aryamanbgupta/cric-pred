import multiprocessing
from simulation_v2 import CricketSimulation
import random
import torch
import numpy as np

def run_simulation(params):
    sim_number, team1, team2, ground = params
    
    # Set a unique random seed for this simulation
    random_seed = random.randint(1, 1000000)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Paths to required files (adjust these as needed)
    model_path = "cricket_transformer_model(1).pth"
    label_encoder_path = "label_encoders(1).pkl"
    scaler_path = "standard_scaler(1).pkl"
    names_csv_path = "names_v2.csv"
    players_info_path = "data/players_info.csv"
    ground_avg_scores_path = "data/ground_average_scores.json"
    
    # Initialize the simulation
    sim = CricketSimulation(model_path, label_encoder_path, scaler_path, names_csv_path, players_info_path, ground_avg_scores_path)
    
    # Run the simulation with the fixed teams and ground
    first_innings, second_innings, team1_score, team2_score = sim.simulate_match(team1, team2, ground)
    
    # Determine the winner
    if team1_score > team2_score:
        winner = "Team 1"
    elif team2_score > team1_score:
        winner = "Team 2"
    else:
        winner = "Tie"
    
    return {
        "simulation_number": sim_number,
        "team1_score": team1_score,
        "team2_score": team2_score,
        "winner": winner,
        "random_seed": random_seed
    }

def main(team1, team2, ground, num_simulations):
    # Determine the number of CPU cores to use
    num_cores = multiprocessing.cpu_count()
    print(f"Running {num_simulations} simulations on {num_cores} cores")
    print(f"Teams: Team 1 vs Team 2")
    print(f"Ground: {ground}")
    
    # Prepare parameters for each simulation
    simulation_params = [(i, team1, team2, ground) for i in range(num_simulations)]
    
    # Create a pool of worker processes
    with multiprocessing.Pool(processes=num_cores) as pool:
        # Run the simulations in parallel
        results = pool.map(run_simulation, simulation_params)
    
    # Process and print the results
    for result in results:
        print(f"\nSimulation {result['simulation_number'] + 1}:")
        print(f"Team 1 Score: {result['team1_score']}")
        print(f"Team 2 Score: {result['team2_score']}")
        print(f"Winner: {result['winner']}")
        print(f"Random Seed: {result['random_seed']}")

if __name__ == "__main__":
    # Define the fixed teams and ground
    TEAM1 = ["7fca84b7", "09a9d073", "3241e3fd", "650d5e49", "bbd41817", "d014d5ac", "c5aef772", "3feda4fa", "4d7f517e", "b0946605", "97bdec3d"]
    TEAM2 = ["3d284ca3", "99b75528", "bb351c23", "abb83e27", "4ae1755b", "50c6bc2b", "e94915e6", "5574750c", "249d60c9", "8d92a2c3", "8db7f47f"]
    GROUND = "M Chinnaswamy Stadium"

    num_simulations = 10  # Adjust this number as needed
    main(TEAM1, TEAM2, GROUND, num_simulations)