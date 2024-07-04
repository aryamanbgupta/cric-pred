import numpy as np
from tensorflow.keras.models import load_model

class CricketSimulation:
    def __init__(self, model_path, num_overs=20):
        self.model = load_model(model_path)
        self.num_overs = num_overs
        self.reset_match()

    def reset_match(self):
        self.inning = 1
        self.score = 0
        self.wickets = 0
        self.balls = 0
        self.batting_team = []
        self.bowling_team = []
        self.current_batter = None
        self.current_bowler = None

    def initialize_teams(self, team1, team2):
        self.batting_team = team1
        self.bowling_team = team2
        self.current_batter = self.batting_team[0]
        self.current_bowler = self.bowling_team[0]

    def generate_ball_input(self):
        return np.array([
            self.inning,
            self.score,
            self.wickets,
            self.balls,
            self.current_batter,
            self.current_bowler,
            (self.balls // 6) % 11 + 1  # Simple batting position based on balls faced
        ]).reshape(1, -1)

    def predict_ball_outcome(self, ball_input):
        prediction = self.model.predict(ball_input)
        return np.argmax(prediction)

    def update_match_state(self, outcome):
        if outcome == 8:  # Wicket
            self.wickets += 1
            # Logic to change batter
        else:
            self.score += outcome

        self.balls += 1

        # Logic to change bowler at the end of an over
        if self.balls % 6 == 0:
            self.current_bowler = (self.current_bowler + 1) % len(self.bowling_team)

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
    sim = CricketSimulation(model_path)
    
    team1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # Replace with actual player IDs
    team2 = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]  # Replace with actual player IDs
    
    first_innings, second_innings, team1_score, team2_score = sim.simulate_match(team1, team2)
    
    print(f"Team 1 Score: {team1_score}")
    print(f"Team 2 Score: {team2_score}")
    if team1_score > team2_score:
        print("Team 1 wins!")
    elif team2_score > team1_score:
        print("Team 2 wins!")
    else:
        print("It's a tie!")