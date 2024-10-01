import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import csv
import json
from collections import defaultdict
from pred_transformer_v2_2 import CricketTransformer
import random


# Set a different seed each time
random_seed = random.randint(1, 10000)
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

class CricketSimulation:
    def __init__(self, model_path, label_encoder_path, scaler_path, names_csv_path, players_info_path, ground_avg_scores_path, num_overs=20):
        self.num_overs = num_overs
        self.reset_match()
        self.load_label_encoders(label_encoder_path)
        self.load_scaler(scaler_path)
        self.load_player_names(names_csv_path)
        self.load_model(model_path)
        self.load_players_info(players_info_path)
        self.load_ground_average_scores(ground_avg_scores_path)
        self.player_stats = defaultdict(lambda: {'batting': [], 'bowling': []})

    def load_model(self, model_path):
        self.model = CricketTransformer(
            input_dim=19,
            n_players=len(self.le_player.classes_),
            n_batting_styles=len(self.le_batting_style.classes_),
            n_playing_roles=len(self.le_playing_role.classes_),
            n_bowling_styles=len(self.le_bowling_style.classes_),
            num_heads=8,
            num_layers=6,
            ff_dim=256,
            dropout=0.1
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def load_label_encoders(self, label_encoder_path):
        with open(label_encoder_path, 'rb') as f:
            encoders = pickle.load(f)
            self.le_player = encoders['player']
            self.le_batting_style = encoders['batting_style']
            self.le_playing_role = encoders['playing_role']
            self.le_bowling_style = encoders['bowling_style']

    def load_scaler(self, scaler_path):
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

    def load_player_names(self, csv_file):
        self.player_names = defaultdict(set)
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                identifier, name = row
                self.player_names[identifier].add(name)
        
        # Choose the longest name for each player as the "full name"
        self.player_full_names = {id: max(names, key=len) for id, names in self.player_names.items()}

    def get_player_name(self, identifier):
        return self.player_full_names.get(identifier, identifier)

    def load_players_info(self, players_info_path):
        self.players_info = {}
        with open(players_info_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.players_info[row['identifier']] = {
                    'batting_style': row['batting_styles'],
                    'bowling_style': row['bowling_styles'],
                    'playing_role': row['playing_roles']
                }

    def load_ground_average_scores(self, ground_avg_scores_path):
        with open(ground_avg_scores_path, 'r') as f:
            self.ground_average_scores = json.load(f)

    def get_player_info(self, player_id):
        info = self.players_info.get(player_id, {})
        return {
            'batting_style': info.get('batting_style', 'Unknown'),
            'bowling_style': info.get('bowling_style', 'Unknown'),
            'playing_role': info.get('playing_role', 'Unknown')
        }

    def get_player_form(self, player_id, role):
        performances = self.player_stats[player_id][role]
        if len(performances) < 3:
            return 0  # Not enough data
        return sum(performances[-3:]) / 3  # Average of last 3 performances

    def update_player_stats(self, player_id, role, performance):
        self.player_stats[player_id][role].append(performance)
        self.player_stats[player_id][role] = self.player_stats[player_id][role][-5:]  # Keep only last 5 performances

    def generate_ball_input(self):
        input_array = np.zeros((1, self.balls + 1, 19))  # +1 to include the current ball

        for i in range(self.balls + 1):
            batter_info = self.get_player_info(self.current_batter)
            non_striker_info = self.get_player_info(self.current_non_striker)
            bowler_info = self.get_player_info(self.current_bowler)

            input_array[0, i, 0] = self.inning
            input_array[0, i, 1] = self.score
            input_array[0, i, 2] = self.wickets
            input_array[0, i, 3] = i
            input_array[0, i, 4] = self.safe_transform(self.le_player, [self.current_batter])[0]
            input_array[0, i, 5] = self.safe_transform(self.le_player, [self.current_non_striker])[0]
            input_array[0, i, 6] = self.safe_transform(self.le_player, [self.current_bowler])[0]
            input_array[0, i, 7] = self.get_batting_position(self.current_batter)
            input_array[0, i, 8] = self.safe_transform(self.le_batting_style, [batter_info['batting_style']])[0]
            input_array[0, i, 9] = self.safe_transform(self.le_playing_role, [batter_info['playing_role']])[0]
            input_array[0, i, 10] = self.safe_transform(self.le_batting_style, [non_striker_info['batting_style']])[0]
            input_array[0, i, 11] = self.safe_transform(self.le_playing_role, [non_striker_info['playing_role']])[0]
            input_array[0, i, 12] = self.safe_transform(self.le_bowling_style, [bowler_info['bowling_style']])[0]
            input_array[0, i, 13] = self.get_player_form(self.current_batter, 'batting')
            input_array[0, i, 14] = self.get_player_form(self.current_non_striker, 'batting')
            input_array[0, i, 15] = self.get_player_form(self.current_bowler, 'bowling')
            input_array[0, i, 16] = 0  # is_noball, set to 0 for now
            input_array[0, i, 17] = 0  # is_wide_or_noball, set to 0 for now
            input_array[0, i, 18] = self.target

        #print("Raw input shape:", input_array.shape)
        # Apply the scaler to the numerical features
        numerical_features = [1, 2, 3, 7, 13, 14, 15, 18]
        flat_numerical = input_array[:, :, numerical_features].reshape(-1, len(numerical_features))
        normalized_numerical = self.scaler.transform(flat_numerical)
        input_array[:, :, numerical_features] = normalized_numerical.reshape(input_array.shape[0], input_array.shape[1], -1)

        #print("Scaled input shape:", input_array.shape)

        return torch.FloatTensor(input_array)

    def safe_transform(self, encoder, values):
        try:
            return encoder.transform(values)
        except ValueError:
            return np.array([-1])  # Return -1 for unknown values

    def get_batting_position(self, batter):
        return self.batting_order.index(batter) + 1 if batter in self.batting_order else len(self.batting_order) + 1

    def predict_ball_outcome(self, ball_input):
        with torch.no_grad():
            runs_pred = self.model(ball_input)
            #print("Runs pred shape:", runs_pred.shape)
            runs_prob = torch.softmax(runs_pred.squeeze(), dim=-1)
            #print("Runs probabilities shape:", runs_prob.shape)
            print("Runs probabilities:", runs_prob[-1])
            
            if runs_prob.dim() == 1:
                outcome = np.random.choice(range(9), p=runs_prob.numpy())
            else:
                outcome = np.random.choice(range(9), p=runs_prob[-1].numpy())  # Use the last ball's prediction
        return outcome

    def simulate_ball(self):
        ball_input = self.generate_ball_input()
        outcome = self.predict_ball_outcome(ball_input)
        self.update_match_state(outcome)

        batter_name = self.get_player_name(self.current_batter)
        bowler_name = self.get_player_name(self.current_bowler)

        print(f"Ball {self.balls}: {batter_name} faces {bowler_name}")
        print(f"Outcome: {self.outcome_to_string(outcome)} | Score: {self.score}/{self.wickets}")
        
        return outcome

    def update_match_state(self, outcome):
        #print(f"Updating match state with outcome: {outcome}")
        if outcome == 8:  # Wicket
            self.wickets += 1
            self.update_player_stats(self.current_bowler, 'bowling', 1)  # 1 wicket
            self.out_batters.append(self.current_batter)
            self.current_batter = self.get_next_batter()
        else:
            self.score += outcome
            self.update_player_stats(self.current_batter, 'batting', outcome / 6)  # Run rate per ball
            self.update_player_stats(self.current_bowler, 'bowling', -outcome / 6)  # Negative run rate for bowler

        self.balls += 1
        #(f"Updated state: Score: {self.score}, Wickets: {self.wickets}, Balls: {self.balls}")
        if self.balls % 6 == 0:
            self.current_batter, self.current_non_striker = self.current_non_striker, self.current_batter

        # Check if target is reached in second innings
        if self.inning == 2 and self.score > self.target:
            return True  # Indicates the innings should end

        return False

    def reset_match(self):
        self.inning = 1
        self.score = 0
        self.wickets = 0
        self.balls = 0
        self.batting_team = []
        self.bowling_team = []
        self.current_batter = None
        self.current_non_striker = None
        self.current_bowler = None
        self.batting_order = []
        self.bowler_overs = {}
        self.last_bowler = None
        self.target = None
        self.out_batters = []

    def initialize_teams(self, team1, team2, ground):
        self.batting_team = team1
        self.bowling_team = team2
        self.batting_order = team1.copy()
        self.current_batter = self.batting_order[0]
        self.current_non_striker = self.batting_order[1]
        self.current_bowler = self.get_next_bowler()
        self.target = self.ground_average_scores.get(ground, 160)

    def simulate_match(self, team1, team2, ground):
        self.reset_match()
        self.initialize_teams(team1, team2, ground)
        
        print("First Innings:")
        first_innings = self.simulate_innings()
        first_innings_score = self.score

        # Reset for second innings
        self.inning = 2
        self.score = 0
        self.wickets = 0
        self.balls = 0
        self.batting_team, self.bowling_team = team2, team1
        self.batting_order = team2.copy()
        self.current_batter = self.batting_order[0]
        self.current_non_striker = self.batting_order[1]
        self.current_bowler = self.get_next_bowler()
        self.target = first_innings_score + 1

        print("\nSecond Innings:")
        second_innings = self.simulate_innings()
        second_innings_score = self.score

        return first_innings, second_innings, first_innings_score, second_innings_score

    # ... (other methods like simulate_over, simulate_innings, etc.)
    def simulate_over(self):
        over_outcomes = []
        initial_bowler = self.current_bowler
        for ball in range(6):
            if self.wickets < 10:
                outcome = self.simulate_ball()
                over_outcomes.append(outcome)
            else:
                break
        
        self.bowler_overs[initial_bowler] = self.bowler_overs.get(initial_bowler, 0) + 1
        self.last_bowler = initial_bowler
        self.current_bowler = self.get_next_bowler()
        
        print(f"Over completed by {self.get_player_name(initial_bowler)}")
        print(f"Bowler overs: {', '.join([f'{self.get_player_name(k)}: {v}' for k, v in self.bowler_overs.items()])}")
        
        return over_outcomes

    def simulate_innings(self):
        innings_outcomes = []
        for over in range(self.num_overs):
            print(f"\nOver {over + 1}:")
            if self.wickets < 10:
                over_outcomes = self.simulate_over()
                innings_outcomes.append(over_outcomes)
                if self.inning == 2 and self.score > self.target:
                    print("Target reached!")
                    break
            else:
                print("All out!")
                break
        return innings_outcomes

    def outcome_to_string(self, outcome):
        if outcome == 8:
            return "Wicket"
        elif outcome == 7:
            return "7+ runs"
        else:
            return f"{outcome} run{'s' if outcome != 1 else ''}"

    def get_next_batter(self):
        for player in self.batting_order:
            if player not in [self.current_batter, self.current_non_striker] and player not in self.out_batters:
                return player
        return None

    def get_next_bowler(self):
        available_bowlers = [b for b in self.bowling_team 
                            if b != self.last_bowler 
                            and self.bowler_overs.get(b, 0) < 4]
        return np.random.choice(available_bowlers) if available_bowlers else None

# Usage example
if __name__ == "__main__":
    model_path = "cricket_transformer_model(1).pth"
    label_encoder_path = "label_encoders(1).pkl"
    scaler_path = "standard_scaler(1).pkl"
    names_csv_path = "names_v2.csv"
    players_info_path = "data/players_info.csv"
    ground_avg_scores_path = "data/ground_average_scores.json"
    
    sim = CricketSimulation(model_path, label_encoder_path, scaler_path, names_csv_path, players_info_path, ground_avg_scores_path)
    
    team1 = ["7fca84b7", "09a9d073", "3241e3fd", "650d5e49", "bbd41817", "d014d5ac", "c5aef772", "3feda4fa", "4d7f517e", "b0946605", "97bdec3d"]
    team2 = ["3d284ca3", "99b75528", "bb351c23", "abb83e27", "4ae1755b", "50c6bc2b", "e94915e6", "5574750c", "249d60c9", "8d92a2c3", "8db7f47f"]
    ground = "M Chinnaswamy Stadium"
    
    first_innings, second_innings, team1_score, team2_score = sim.simulate_match(team1, team2, ground)
    
    print("\nMatch Result:")
    print(f"Team 1: {team1_score}")
    print(f"Team 2: {team2_score}")
    if team1_score > team2_score:
        print("Team 1 wins!")
    elif team2_score > team1_score:
        print("Team 2 wins!")
    else:
        print("It's a tie!")