import pandas as pd
import numpy as np
from kloppy import metrica, sportscode
from games import asset_paths, half_video_blocks


# SETUP: Load Data
paths = asset_paths()
print("Loading tracking data... (This may take 30-60 seconds)")
dataset = metrica.load_tracking_epts(
    meta_data=paths["atd_meta"], 
    raw_data=paths["atd_raw"]
)
df = dataset.to_df()

# Initialize our final possession label column with NaNs
df['possession_label'] = np.nan 


# STEP 1: The Macro Filter (Smart Tagging)
print("Applying Step 1: Macro Filter (Loading SportsCode XML)...")
df['macro_phase'] = 'Unknown'

# Load the smart tagging using the hackathon's built-in kloppy method
events_df = sportscode.load(paths["smart_tagging"]).to_df()

# Kloppy returns pandas timedeltas, so we convert them to total seconds
for _, row in events_df.iterrows():
    start_sec = row['timestamp'].total_seconds()
    end_sec = row['end_timestamp'].total_seconds()
    label = row['code']  # This correctly extracts 'SET PIECES', 'BUILD UP', etc.
    
    # Convert seconds to video frame (25 fps), add 1 to match Kloppy's 1-indexed frame_id
    start_frame = round(start_sec * 25) + 1
    end_frame = round(end_sec * 25) + 1
    
    df.loc[(df['frame_id'] >= start_frame) & (df['frame_id'] <= end_frame), 'macro_phase'] = label


# STEP 2: Candidate Frame Extraction (ELASTIC Filter)
print("Applying Step 2: Candidate Extraction...")
df['ball_x_diff'] = df['ball_x'].diff()
df['ball_y_diff'] = df['ball_y'].diff()
df['ball_velocity'] = np.sqrt(df['ball_x_diff']**2 + df['ball_y_diff']**2)
df['ball_acceleration'] = df['ball_velocity'].diff()
valid_ball_mask = df['ball_x'].notna()


# STEP 3: The Micro Finder (Distance & Trajectory Scoring)
print("Applying Step 3: Micro Finder (Scaling to true meters)...")
player_ids = [str(i) for i in range(1, 47)]
dist_columns = []

# Actual pitch dimensions from games.py
PITCH_LENGTH = 105.3 
PITCH_WIDTH = 68.0

for pid in player_ids:
    if f'{pid}_x' in df.columns:
        dist_col = f'dist_to_{pid}'
        
        # Convert normalized differences into real-world meters BEFORE squaring
        dx_meters = (df['ball_x'] - df[f'{pid}_x']) * PITCH_LENGTH
        dy_meters = (df['ball_y'] - df[f'{pid}_y']) * PITCH_WIDTH
        
        df[dist_col] = np.sqrt(dx_meters**2 + dy_meters**2)
        dist_columns.append((dist_col, pid))

dist_cols = [c[0] for c in dist_columns]

df['min_dist'] = np.nan
df['closest_player'] = np.nan

# ONLY calculate the minimum and idxmin on rows where the ball is actually visible
df.loc[valid_ball_mask, 'min_dist'] = df.loc[valid_ball_mask, dist_cols].min(axis=1)
df.loc[valid_ball_mask, 'closest_player'] = df.loc[valid_ball_mask, dist_cols].idxmin(axis=1).str.replace('dist_to_', '').astype(float)

# Apply ELASTIC 3-meter hard cutoff using real-world meters
DISTANCE_THRESHOLD = 3.0 
close_enough_mask = df['min_dist'] <= DISTANCE_THRESHOLD

# Map player IDs to Home (H) or Away (A) teams (Barcelona is 1-24, Atletic is 25-46)
df['math_team'] = np.where(df['closest_player'].astype(float) <= 24, 'H', 'A')

# Unlock the column to accept text
df['possession_label'] = df['possession_label'].astype(object)

# Assign possession ONLY if the ball is visible AND a player is within 3 meters
df.loc[valid_ball_mask & close_enough_mask, 'possession_label'] = df['math_team']


# STEP 4: The Smoother (Majority Vote / Rolling Window)
print("Applying Step 4: The Smoother...")
label_map = {'H': 1, 'A': -1}
df['numeric_label'] = df['possession_label'].map(label_map).fillna(0)

# FIX 1: Apply a 75-frame (3 seconds at 25fps) rolling mean to stop fragmentation
smoothed_signal = df['numeric_label'].rolling(window=75, center=True, min_periods=1).mean()

df['smoothed_label'] = np.nan
df['smoothed_label'] = df['smoothed_label'].astype(object)

df.loc[smoothed_signal > 0.3, 'smoothed_label'] = 'H'
df.loc[smoothed_signal < -0.3, 'smoothed_label'] = 'A'


# STEP 5: The Bridge (Label Forward-Fill & Buffer)
print("Applying Step 5: The Bridge...")

# FIX 2: The Dead-Ball Buffer
# Create a temporary column that is True during the actual XML set piece, NaN otherwise
df['is_dead_ball'] = np.where(df['macro_phase'] == 'SET PIECES', True, np.nan)

# Backward-fill the True values up to 125 frames (5 seconds) into the past
df['is_dead_ball'] = df['is_dead_ball'].bfill(limit=125)

# Forward-fill a little bit too (25 frames / 1 sec) just in case the tag cuts off before the ball actually moves
df['is_dead_ball'] = df['is_dead_ball'].ffill(limit=25)

# Override the math and apply 'D' wherever our expanded mask is True
df.loc[df['is_dead_ball'] == True, 'smoothed_label'] = 'D'

# CRITICAL RESTORE: Transfer the smoothed labels and fill the empty gaps
df['final_possession'] = df['smoothed_label'].ffill()
df['final_possession'] = df['final_possession'].bfill()

# EXPLICIT OUT-OF-BOUNDS MASKING ('X')
# Fetch official graded frame boundaries dynamically from games.py
halves = half_video_blocks("BAR-ATH") 
h1_start, h1_end = halves[0][1], halves[0][2]
h2_start, h2_end = halves[1][1], halves[1][2]

# Apply the 'X' mask using the dynamic variables
df.loc[df['frame_id'] < h1_start, 'final_possession'] = 'X' 
df.loc[(df['frame_id'] > h1_end) & (df['frame_id'] < h2_start), 'final_possession'] = 'X' 
df.loc[df['frame_id'] > h2_end, 'final_possession'] = 'X' 


# STEP 6: Export Debug File
output_path = "data/tactical/BAR-ATH_possession_output.csv"
print(f"Saving debug pipeline to {output_path}...")
export_columns = ['period_id', 'timestamp', 'frame_id', 'macro_phase', 'closest_player', 'min_dist', 'final_possession']
df[export_columns].to_csv(output_path, index=False)


# Export Local Scorer File (0-Indexed)
print("Formatting for local scorer...")
# Convert Kloppy's 1-indexed frame_id to the 0-indexed video frame required by the scorer
df['frame'] = df['frame_id'] - 1

# Rename and isolate the required columns
df['label'] = df['final_possession']
submission_df = df[['frame', 'label']]

submission_path = "404_TNF.csv"
submission_df.to_csv(submission_path, index=False)

print("Save complete!")