# BBLF AI Selector: Part 1: Pre Tournament Optimal Squad - Data Prep 

'''
Round Begins 15th December 2024!!!

BBL 24/25 Rules:
    1. 9 rounds with 4 or 5 games
    2. Three emergency players
    3. Captain gets double points, Vice Captain will get double points if the captain does not play
    4. Trades: 3 trades per round. Trades are not transferable round to round. Two trade boost (allowing for 4 trades in 2 rounds)
    5. Select 12 starting players but recieve points for top 11 performances
    6. Selection requirements: 7 batters, 7 bowlers, 2 wicketkeepers, 1 flex player
    7. Budget is 2 million
    8. When a team plays twice in one round, they recieve the cumulative number of points
    9. 

    New Changes
    1. Flex Position: Any position, scores will be best top 11 players after emergency players are subbed in.
    2.  

BBL 23/24 Rules:
    1. Point scoring is the same as the scoring in BBL 24.25
'''

# 0. Prerequistes
import pandas as pd
import numpy as np
import os

os.getcwd()
directory = 'C:/Users/dilan/OneDrive/Documents/Data Science Projects/Big Bash Fantasy AI'

# 1. Data Extraction 
    # Pull in all_matches csv file 

raw_df = pd.read_csv(os.path.join(directory,'data/all_matches.csv'), low_memory=False)
raw_df.head
raw_df_clean = raw_df
raw_df_season = raw_df_clean["season"].str.split("/", n=1, expand = True)
raw_df_clean["season"] = raw_df_season[0].astype(int) - 2010

print(raw_df_clean)
# 2. Create Fantasy Points of each player per game
    # a. Create function to aggregate bowl by bowl data to overall innings scorecard summary

# List of all the unique BBL match ids    
match_id_list = raw_df_clean.match_id.unique()

# Create empty dataframe to hold fantasy point outputs from the for loop
fant_pts_table = pd.DataFrame()
fant_pts_table["match_id"] = []
fant_pts_table["player"] = []
fant_pts_table["fantasy_point_bat"] = []
fant_pts_table["fantasy_point_bowl"] = []
fant_pts_table["fantasy_point"] = []

# For loop to collect the player fantasy points for each game and add to fantasy points table
# For loop start ---------------------------------------

for x in match_id_list:

    # select specific match id from all ids
    game_df = raw_df[raw_df["match_id"] == x]

    #game_df.to_csv('data/python_datasets/bbl_23_24_final_data.csv')

    game_df_clean = game_df[["match_id", "innings", "ball", "striker", "non_striker","bowler", "runs_off_bat","wides","noballs","wicket_type","player_dismissed","batting_team", "bowling_team"]]

    # 2a. Batting (1. Total Runs, 2. Strike Rate, 3. Run Bonus (50 & 100))
        # 1. Total Runs
    game_bat_df = game_df_clean[["match_id", "striker", "runs_off_bat", "batting_team", "bowling_team"]]
    game_bat_df = game_bat_df.rename(columns = {"batting_team": "team", "bowling_team": "opposition"})
    game_bat_df_agg = game_bat_df.groupby(['match_id', 'striker', "team", "opposition"], as_index=False).agg(
    total_runs=('runs_off_bat',"sum"),
    total_balls=("runs_off_bat","count"))

        # 2. Strike Rate
    game_bat_df_agg["strike_rate"] = game_bat_df_agg["total_runs"]/game_bat_df_agg["total_balls"]*100

    sr_conditions = [
        (game_bat_df_agg['strike_rate'] < 120),
        (game_bat_df_agg['strike_rate'] >= 120) & (game_bat_df_agg['strike_rate'] < 130),
        (game_bat_df_agg['strike_rate'] >= 130) & (game_bat_df_agg['strike_rate'] < 140),
        (game_bat_df_agg['strike_rate'] >= 140) & (game_bat_df_agg['strike_rate'] < 150),
        (game_bat_df_agg['strike_rate'] >= 150) & (game_bat_df_agg['strike_rate'] < 160),
        (game_bat_df_agg['strike_rate'] >= 160)
    ]

    sr_group = [0,1,2,3,4,5]

    game_bat_df_agg["strike_rate_group"] = np.select(sr_conditions, sr_group)

        # 3. Run Bonus
    rb_conditions = [
        (game_bat_df_agg['total_runs'] < 50),
        (game_bat_df_agg['total_runs'] >= 50) & (game_bat_df_agg['total_runs'] < 100),
        (game_bat_df_agg['total_runs'] >= 100)
    ]

    rb_group = [0,1,2]

    game_bat_df_agg["run_bonus_group"] = np.select(rb_conditions, rb_group)

        # 4. Batting Fantasy Points

    game_bat_df_agg["fantasy_point_bat"] = game_bat_df_agg["total_runs"] + game_bat_df_agg["strike_rate_group"]*5 + game_bat_df_agg["run_bonus_group"]*10 

    # 2b. Bowling (1. Wickets, 2. Economy, 3. Wicket Bonus (3), 4. Maiden , 5. Dot Ball 6. Extras)

        # 1. Wickets, Dot Balls, Extras
    game_bowl_df = game_df[["match_id", "innings", "ball", "striker", "non_striker","bowler", "runs_off_bat","wides","noballs","wicket_type","player_dismissed","batting_team", "bowling_team"]]
    game_bowl_df = game_bowl_df.rename(columns = {"batting_team": "opposition", "bowling_team": "team"})
    game_bowl_df["ball"] = game_bowl_df["ball"].astype(str) 

    game_bowl_df_over = game_bowl_df["ball"].str.split(".", n=1, expand = True)
    game_bowl_df["over"] = game_bowl_df_over[0]
    game_bowl_df["ball"] = game_bowl_df_over[1]

    game_bowl_df["bowl_wicket"] = np.where((game_bowl_df["wicket_type"] == "runout" ) |(game_bowl_df["wicket_type"].isnull()) , 0 , 1)
    game_bowl_df["dot_ball_f"] = np.where(game_bowl_df["runs_off_bat"] == 0, 1, 0)
    game_bowl_df["wides"] = game_bowl_df["wides"].fillna(0)
    game_bowl_df["noballs"] = game_bowl_df["noballs"].fillna(0)
    game_bowl_df["bowl_extra"] = (game_bowl_df["wides"] + game_bowl_df["noballs"])
    game_bowl_df["elig_bowl"] = np.where((game_bowl_df["bowl_extra"] == 0) ,1, 0)

    game_bowl_df_agg = game_bowl_df.groupby(["match_id", "bowler", "team", "opposition"], as_index=False).agg(
    wickets = ("bowl_wicket", "sum"),
    runs = ("runs_off_bat", "sum"),
    ball_cnt = ("elig_bowl", "sum"),
    dot_ball_cnt = ("dot_ball_f", "sum"),
    extra_cnt = ("bowl_extra", "sum")
    )

        # 2. Economy
    game_bowl_df_agg["econ_elig_f"] = np.where((game_bowl_df_agg["ball_cnt"] >= 18), 1, 0)
    game_bowl_df_agg["econ"] = (game_bowl_df_agg["runs"]/game_bowl_df_agg["ball_cnt"]*6)

    econ_conditions = [
        (game_bowl_df_agg['econ'] <= 4),
        (game_bowl_df_agg['econ'] > 4) & (game_bowl_df_agg['econ'] <= 5),
        (game_bowl_df_agg['econ'] > 5) & (game_bowl_df_agg['econ'] <= 6),
        (game_bowl_df_agg['econ'] > 6) & (game_bowl_df_agg['econ'] <= 7),
        (game_bowl_df_agg['econ'] > 7) & (game_bowl_df_agg['econ'] <= 8),
        (game_bowl_df_agg['econ'] > 8)
    ]

    econ_group = [5,4,3,2,1,0]
    game_bowl_df_agg["econ_bonus_group"] = np.select(econ_conditions, econ_group)

        # 3. Wicket Bonus
    wb_conditions = [
        (game_bowl_df_agg['wickets'] < 3),
        (game_bowl_df_agg['wickets'] >= 3) & (game_bowl_df_agg['wickets'] < 6),
        (game_bowl_df_agg['wickets'] >= 6) & (game_bowl_df_agg['wickets'] < 9),
        (game_bowl_df_agg['wickets'] >= 9)
    ]

    wb_group = [0,1,2,3]
    game_bowl_df_agg["wicket_bonus_group"] = np.select(wb_conditions, wb_group)

        # 4. Maiden
    game_bowl_over_df_agg = game_bowl_df.groupby(["match_id", "bowler", "over"]).agg(
    dot_ball_cnt = ("dot_ball_f", "sum")
    )

    game_bowl_over_df_agg["maiden_f"] = np.where((game_bowl_over_df_agg["dot_ball_cnt"] == 6), 1, 0)

    game_bowl_maiden_agg = game_bowl_over_df_agg.groupby(["match_id", "bowler"]).agg(
    maiden_cnt = ("maiden_f", "sum")    
    )

    game_bowl_df_agg = pd.merge(game_bowl_df_agg, game_bowl_maiden_agg, left_on = "bowler", right_on = "bowler", how = "left")

        # 5. Bowling Fantasy Points 
    game_bowl_df_agg["fantasy_point_bowl"] = game_bowl_df_agg["wickets"]*20 + game_bowl_df_agg["dot_ball_cnt"] + game_bowl_df_agg["extra_cnt"]*(-1) + (game_bowl_df_agg["econ_bonus_group"]*game_bowl_df_agg["econ_elig_f"]*5) + game_bowl_df_agg["wicket_bonus_group"]*10 + game_bowl_df_agg["maiden_cnt"]*10 

    # 2c. Overall game player fantasy points
    game_bat_play_sum = game_bat_df_agg[["match_id", "striker", "team", "opposition", "fantasy_point_bat"]].rename(columns={"striker": "player"})
    game_bowl_play_sum = game_bowl_df_agg[["match_id", "bowler", "team", "opposition", "fantasy_point_bowl"]].rename(columns={"bowler": "player"})
    game_play_fantasy_pts_df = pd.merge(game_bat_play_sum, game_bowl_play_sum, left_on = ["player", "match_id"], right_on = ["player", "match_id"], how = "outer")
    game_play_fantasy_pts_df = game_play_fantasy_pts_df.fillna(0)
    game_play_fantasy_pts_df["fantasy_point"] = game_play_fantasy_pts_df["fantasy_point_bat"] + game_play_fantasy_pts_df["fantasy_point_bowl"]
    #print(game_play_fantasy_pts_df)

    fant_pts_table = pd.concat([fant_pts_table, game_play_fantasy_pts_df])

# For loop end ---------------------------------------

# 3. Add key index columns to fantasy point table
# fantasy points raw dataframe
fant_pts_df_raw = fant_pts_table

# Create match ids primary index table
match_primary_index = raw_df_clean[["match_id", "start_date", "season", "venue", "innings", "batting_team", "bowling_team"]]
match_primary_index = match_primary_index[match_primary_index.innings == 1]

match_primary_index_agg = match_primary_index.groupby(["match_id", "start_date", "season", "venue", "innings", "batting_team", "bowling_team"], as_index=False).agg(
    count = ("season", "count"),
    )

match_primary_index_agg =  match_primary_index_agg[["match_id", "start_date", "season", "venue", "batting_team", "bowling_team"]].rename(columns={"batting_team": "first_bat_team", "bowling_team": "first_bowl_team"})

# Join primary index and fantasy points table
fant_pts_df = pd.merge(fant_pts_df_raw, match_primary_index_agg, left_on = ["match_id"], right_on = ["match_id"], how = "left")
fant_pts_df["team"] = np.where(fant_pts_df["team_x"] == 0, fant_pts_df["team_y"], fant_pts_df["team_x"]) 
fant_pts_df["opp"] = np.where(fant_pts_df["opposition_x"] == 0, fant_pts_df["opposition_y"], fant_pts_df["opposition_x"]) 
fant_pts_df = fant_pts_df.drop(['team_x', 'team_y', 'opposition_x', 'opposition_y'], axis=1)
print(fant_pts_df)

fant_pts_df.to_csv('data/python_datasets/fantasy_point_player_table.csv')

# Code End (Created dataframe with all players fantasy points for every BBL game) ---------------------------