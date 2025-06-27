import tkinter as tk
from tkinter import messagebox
import pandas as pd
import joblib

# Load model and columns
model = joblib.load('score_model.pkl')
model_columns = joblib.load('model_columns.pkl')

teams = ['India', 'Australia', 'England', 'South Africa', 'Pakistan', 'New Zealand', 'Sri Lanka', 'Bangladesh']

root = tk.Tk()
root.title("Cricket Score Predictor")
root.geometry("500x450")
root.resizable(False, False)

tk.Label(root, text="Cricket Score Predictor", font=("Arial", 16, "bold")).pack(pady=10)

batting_team_var = tk.StringVar(value=teams[0])
bowling_team_var = tk.StringVar(value=teams[1])

def create_dropdown(label, var):
    frame = tk.Frame(root)
    frame.pack(pady=5)
    tk.Label(frame, text=label, font=("Arial", 12)).pack(side=tk.LEFT)
    tk.OptionMenu(frame, var, *teams).pack(side=tk.LEFT)

def create_input_field(label):
    frame = tk.Frame(root)
    frame.pack(pady=5)
    tk.Label(frame, text=label, font=("Arial", 12)).pack(side=tk.LEFT)
    entry = tk.Entry(frame)
    entry.pack(side=tk.LEFT)
    return entry

create_dropdown("Batting Team:", batting_team_var)
create_dropdown("Bowling Team:", bowling_team_var)

entry_runs = create_input_field("Runs:")
entry_wickets = create_input_field("Wickets:")
entry_overs = create_input_field("Overs:")
entry_last5 = create_input_field("Runs in last 5 overs:")

result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.pack(pady=10)

def predict_score():
    try:
        runs = float(entry_runs.get())
        wickets = int(entry_wickets.get())
        overs = float(entry_overs.get())
        runs_last5 = float(entry_last5.get())
        bat_team = batting_team_var.get()
        bowl_team = bowling_team_var.get()

        if bat_team == bowl_team:
            messagebox.showerror("Error", "Batting and Bowling teams must be different.")
            return

        if wickets >= 10 or overs >= 20:
            result_label.config(text=f"Innings Over. Final Score: {int(runs)}")
            return

        input_dict = {
            'runs': runs,
            'wickets': wickets,
            'overs': overs,
            'runs_last_5': runs_last5
        }

        for team in teams:
            input_dict[f'batting_team_{team}'] = 1 if team == bat_team else 0
            input_dict[f'bowling_team_{team}'] = 1 if team == bowl_team else 0

        input_df = pd.DataFrame([input_dict])

        for col in model_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[model_columns]

        prediction = model.predict(input_df)[0]

        if wickets > 5:
            penalty = 0.05 * (wickets - 5)
            prediction *= (1 - penalty)

        if prediction < runs:
            prediction = runs + 10

        low = int(prediction - 10)
        high = int(prediction + 10)
        result_label.config(text=f"Predicted Final Score: {low} - {high}")
    except ValueError:
        messagebox.showerror("Error", "Enter valid numeric inputs.")

tk.Button(root, text="Predict Final Score", command=predict_score).pack(pady=10)

root.mainloop()
