import numpy as np
from tensorflow.keras.models import load_model
import pickle

class CricketSimulation:
    def __init__(self, model_path, num_overs=20):
        self.model = load_model(model_path)
        self.num_overs = num_overs
        self.reset_match()
        self.load_label_encoders(label_encoder_path)
    
    def load_label_encoders(self, label_encoder_path):
        if label_encoder_path:
            with open(label_encoder_path, 'rb') as f:
                encoders = pickle.load(f)
                self.le_batter = encoders['batter']
                self.le_bowler = encoders['bowler']
        else:
            self.le_batter = LabelEncoder()
            self.le_bowler = LabelEncoder()

    def reset_match(self):
        self.inning = 1
        self.score = 0
        self.wickets = 0
        self.balls = 0
        self.batting_team = []
        self.bowling_team = []
        self.current_batter = None
        self.current_bowler = None

    def safe_transform(self, encoder, values):
        try:
            return encoder.transform(values)
        except ValueError:
            return np.array([encoder.transform([value])[0] if value in encoder.classes_ else -1 for value in values])

    def initialize_teams(self, team1, team2):
        self.batting_team = self.safe_transform(self.le_batter, team1)
        self.bowling_team = self.safe_transform(self.le_bowler, team2)
        self.current_batter = self.batting_team[0]
        self.current_bowler = self.bowling_team[0]

        # Print warnings for unknown players
        unknown_batters = [team1[i] for i, val in enumerate(self.batting_team) if val == -1]
        unknown_bowlers = [team2[i] for i, val in enumerate(self.bowling_team) if val == -1]
        
        if unknown_batters:
            print(f"Warning: Unknown batters (set to -1): {', '.join(unknown_batters)}")
        if unknown_bowlers:
            print(f"Warning: Unknown bowlers (set to -1): {', '.join(unknown_bowlers)}")

    def generate_ball_input(self):
        return [
            np.array([self.inning]),
            np.array([self.score]),
            np.array([self.wickets]),
            np.array([self.balls]),
            np.array([max(self.current_batter, 0)]),  # Use 0 instead of -1 for unknown batters
            np.array([max(self.current_bowler, 0)]),  # Use 0 instead of -1 for unknown bowlers
            np.array([(self.balls // 6) % 11 + 1])  # Simple batting position based on balls faced
        ]

    def predict_ball_outcome(self, ball_input):
        try:
            prediction = self.model.predict(ball_input, verbose=0)[0]
            return np.random.choice(range(9), p=prediction)
        except ValueError as e:
            print(f"Error in prediction: {e}")
            print(f"Input shape: {[x.shape for x in ball_input]}")
            raise

    def get_next_batter(self):
        available_batters = [b for b in self.batting_team if b != -1]
        return available_batters[self.wickets] if self.wickets < len(available_batters) else -1

    def get_next_bowler(self):
        available_bowlers = [b for b in self.bowling_team if b != -1]
        return np.random.choice(available_bowlers) if available_bowlers else -1

    def update_match_state(self, outcome):
        if outcome == 8:  # Wicket
            self.wickets += 1
            self.current_batter = self.get_next_batter()
        else:
            self.score += outcome

        self.balls += 1
        
        # add check to ensure that its not an extra
        if self.balls % 6 == 0:
            self.current_bowler = self.get_next_bowler()

    

    def simulate_ball(self):
        ball_input = self.generate_ball_input()
        outcome = self.predict_ball_outcome(ball_input)
        self.update_match_state(outcome)
        return outcome

    def simulate_over(self):
        over_outcomes = []
        for _ in range(6):
            if self.wickets < 10:
                outcome = self.simulate_ball()
                over_outcomes.append(outcome)
            else:
                break
        return over_outcomes

    def simulate_innings(self):
        innings_outcomes = []
        for _ in range(self.num_overs):
            if self.wickets < 10:
                over_outcomes = self.simulate_over()
                innings_outcomes.append(over_outcomes)
            else:
                break
        return innings_outcomes

    def simulate_match(self, team1, team2):
        self.reset_match()
        self.initialize_teams(team1, team2)
        
        # First innings
        first_innings = self.simulate_innings()
        first_innings_score = self.score

        # Reset for second innings
        self.inning = 2
        self.score = 0
        self.wickets = 0
        self.balls = 0
        self.batting_team, self.bowling_team = self.bowling_team, self.batting_team
        self.current_batter = self.batting_team[0]
        self.current_bowler = self.bowling_team[0]

        # Second innings
        second_innings = self.simulate_innings()

        return first_innings, second_innings, first_innings_score, self.score

# Usage example
if __name__ == "__main__":
    model_path = "pred_v1.h5"
    label_encoder_path = "label_encoders.pkl"
    sim = CricketSimulation(model_path)
    
    team1 = ["7fca84b7", "09a9d073", "3241e3fd", "650d5e49", "bbd41817", "d014d5ac", "c5aef772", "3feda4fa", "4d7f517e", "b0946605", "97bdec3d"]  # Replace with actual player IDs
    team2 = ["3d284ca3", "99b75528", "bb351c23", "abb83e27", "4ae1755b", "50c6bc2b", "e94915e6", "5574750c", "249d60c9", "8d92a2c3", "8db7f47f"]  # Replace with actual player IDs
    
    first_innings, second_innings, team1_score, team2_score = sim.simulate_match(team1, team2)
    
    print(f"Team 1 Score: {team1_score}")
    print(f"Team 2 Score: {team2_score}")
    if team1_score > team2_score:
        print("Team 1 wins!")
    elif team2_score > team1_score:
        print("Team 2 wins!")
    else:
        print("It's a tie!")