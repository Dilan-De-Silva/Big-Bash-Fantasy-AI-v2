# BBLF AI Selector: Part 2: During Tournament Optimal Squad - Data Prep 

'''
Data preparation for during campaign features to add to fantasy_point_player_table

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

# 1. Overs Bowled 
# a. Pull in all_matches csv file 

raw_df = pd.read_csv(os.path.join(directory,'data/all_matches.csv'), low_memory=False)
raw_df.head
raw_df_clean = raw_df
raw_df_season = raw_df_clean["season"].str.split("/", n=1, expand = True)
raw_df_clean["season"] = raw_df_season[0].astype(int) - 2010

# b. During game over bowled 
raw_df_clean["ball"] = raw_df_clean["ball"].astype(str) 
raw_df_clean[["over", "balls"]] = raw_df_clean['ball'].str.split('.', expand=True)
raw_df_clean["over"] = raw_df_clean["over"].astype(int) + 1
raw_df_clean["balls"] = raw_df_clean["balls"].astype(int)

raw_df_clean = pd.get_dummies(raw_df_clean, columns= ['over'])

raw_df_clean.loc[:,"over_1":"over_20"] = raw_df_clean.loc[:,"over_1":"over_20"].astype(int)
raw_df_clean.loc[:,"over_1":"over_20"] = raw_df_clean.loc[:,"over_1":"over_20"].astype(object)

match_feat_df = raw_df_clean.loc[:,"match_id":"over_20"]
match_feat_df = match_feat_df.drop(columns = match_feat_df.loc[:, 'start_date':'non_striker'].columns)
match_feat_df = match_feat_df.drop(columns = match_feat_df.loc[:, 'runs_off_bat':'balls'].columns)

print(match_feat_df)

# c. Calculate in game features for each player per game
match_feat_df_agg = match_feat_df.groupby(['match_id', "bowler"], as_index=False).agg(
    over_1_f =('over_1',"sum"),
    over_2_f =("over_2","sum"),
    over_3_f =("over_3","sum"),
    over_4_f =("over_4","sum"),
    over_5_f =("over_5","sum"),
    over_6_f =("over_6","sum"),
    over_7_f =("over_7","sum"),
    over_8_f =("over_8","sum"),
    over_9_f =("over_9","sum"),
    over_10_f =("over_10","sum"),
    over_11_f =("over_11","sum"),
    over_12_f =("over_12","sum"),
    over_13_f =("over_13","sum"),
    over_14_f =("over_14","sum"),
    over_15_f =("over_15","sum"),
    over_16_f =("over_16","sum"),
    over_17_f =("over_17","sum"),
    over_18_f =("over_18","sum"),
    over_19_f =("over_19","sum"),
    over_20_f =("over_20","sum"))

print(match_feat_df_agg)

match_feat_df_agg.loc[:,"over_1_f":"over_20_f"] = np.where(match_feat_df_agg.loc[:,"over_1_f":"over_20_f"] > 0, 1, 0) 

# d. Save data set
match_feat_df_agg.to_csv('data/python_datasets/during_tourny_overs_bowl.csv')

# 3. During Season prior fantasy stats

# a. loop through each season, each player of the season and each match they played
full_fbbl_play_table = pd.read_csv(os.path.join(directory,'data/python_datasets/fantasy_point_player_table.csv'), low_memory=False)  
full_fbbl_play_table = full_fbbl_play_table[["season","match_id", "player", "fantasy_point"]]

# Create empty dataframe to hold fantasy point outputs from the for loop
curr_seas_fant_pts_table = pd.DataFrame()
curr_seas_fant_pts_table["season"] = []
curr_seas_fant_pts_table["match_id"] = []
curr_seas_fant_pts_table["player"] = []
curr_seas_fant_pts_table["prev_game_fp"] = []
curr_seas_fant_pts_table["last_2g_avg_fp"] = []
curr_seas_fant_pts_table["curr_seas_total_fp"] = []
curr_seas_fant_pts_table["curr_seas_avg_fp"] = []
curr_seas_fant_pts_table["curr_seas_max_fp"] = []
curr_seas_fant_pts_table["curr_seas_min_fp"] = []
curr_seas_fant_pts_table["curr_seas_std_fp"] = []

# List of all the past BBL season    
season_list = full_fbbl_play_table.season.unique()
print(season_list)

# For loop for all past BBL seasons
for i in season_list:

    full_fbbl_play_table_season =  full_fbbl_play_table[full_fbbl_play_table.season == i]

    # List of all the unique players in the season    
    season_play_list = full_fbbl_play_table_season.player.unique()

    # For loop for all BBL players in particular season
    for j in season_play_list:
        full_fbbl_play_seas_df = full_fbbl_play_table_season[full_fbbl_play_table_season.player == j]
        full_fbbl_play_seas_df = full_fbbl_play_seas_df.sort_values(by = ['match_id'])
        full_fbbl_play_seas_df["match_num"] = np.arange(len(full_fbbl_play_seas_df)) + 1  

        # List of all the matches played by individual    
        match_num_list = full_fbbl_play_seas_df.match_num.unique()
        # match_num_list = match_num_list[match_num_list > 2]

        # For loop for all individual games by individual in particular season
        for k in match_num_list:
            # Current Game
            match_fbbl_play_seas_df = full_fbbl_play_seas_df[full_fbbl_play_seas_df.match_num == k]

            # Previous Game Attributes
            match_fbbl_play_seas_df_prev = full_fbbl_play_seas_df[full_fbbl_play_seas_df.match_num == k - 1].rename(columns={"fantasy_point":"prev_game_fp"})
            match_fbbl_play_seas_df_prev = match_fbbl_play_seas_df_prev.drop(columns=["match_id", "match_num"], axis = 1)
            match_fbbl_play_seas_df = pd.merge(match_fbbl_play_seas_df, match_fbbl_play_seas_df_prev, left_on = ["player", "season"], right_on = ["player", "season"], how = "left")

            # Prior 2 Games in the season aggregate attributes
            match_fbbl_play_seas_df_prior_2g = full_fbbl_play_seas_df[(full_fbbl_play_seas_df.match_num < k) & (full_fbbl_play_seas_df.match_num >= k - 2)]
            match_fbbl_play_seas_df_prior_2g = match_fbbl_play_seas_df_prior_2g.drop(columns=["match_id", "match_num"], axis = 1)
            match_fbbl_play_seas_df_prior_agg_2g = match_fbbl_play_seas_df_prior_2g.groupby(["season", "player"], as_index=False).agg(
            last_2g_avg_fp = ('fantasy_point', "mean"),
            )

            match_fbbl_play_seas_df = pd.merge(match_fbbl_play_seas_df, match_fbbl_play_seas_df_prior_agg_2g, left_on = ["player", "season"], right_on = ["player", "season"], how = "left")

            # Prior 3 Games in the season aggregate attributes
            match_fbbl_play_seas_df_prior_3g = full_fbbl_play_seas_df[(full_fbbl_play_seas_df.match_num < k) & (full_fbbl_play_seas_df.match_num >= k - 3)]
            match_fbbl_play_seas_df_prior_3g = match_fbbl_play_seas_df_prior_3g.drop(columns=["match_id", "match_num"], axis = 1)
            match_fbbl_play_seas_df_prior_agg_3g = match_fbbl_play_seas_df_prior_3g.groupby(["season", "player"], as_index=False).agg(
            last_3g_avg_fp = ('fantasy_point', "mean"),
            )

            match_fbbl_play_seas_df = pd.merge(match_fbbl_play_seas_df, match_fbbl_play_seas_df_prior_agg_3g, left_on = ["player", "season"], right_on = ["player", "season"], how = "left")

            # All prior games in the season aggregate attributes
            match_fbbl_play_seas_df_prior = full_fbbl_play_seas_df[full_fbbl_play_seas_df.match_num < k]
            match_fbbl_play_seas_df_prior = match_fbbl_play_seas_df_prior.drop(columns=["match_id", "match_num"], axis = 1)
            match_fbbl_play_seas_df_prior_agg = match_fbbl_play_seas_df_prior.groupby(["season", "player"], as_index=False).agg(
            curr_seas_total_fp =('fantasy_point',"sum"),
            curr_seas_avg_fp = ('fantasy_point', "mean"),
            curr_seas_max_fp = ('fantasy_point', 'max'),
            curr_seas_min_fp = ('fantasy_point', 'min'),
            curr_seas_std_fp = ('fantasy_point', 'std')
            )

            match_fbbl_play_seas_df = pd.merge(match_fbbl_play_seas_df, match_fbbl_play_seas_df_prior_agg, left_on = ["player", "season"], right_on = ["player", "season"], how = "left")
            match_fbbl_play_seas_df = match_fbbl_play_seas_df.drop(columns = ["fantasy_point", "match_num"], axis = 1)

            # Add all during season attributes to empty table
            curr_seas_fant_pts_table = pd.concat([curr_seas_fant_pts_table, match_fbbl_play_seas_df])

# b. Save data set
curr_seas_fant_pts_table.to_csv('data/python_datasets/dur_tourny_fant_pts_attr_table.csv')

# 4. Batting Order Flag
# a. Pull in all_matches csv file 
match_full_df = raw_df_clean[["match_id", 'season', 'innings', 'ball', 'striker']]
match_full_df["ball"] = match_full_df["ball"].astype(str) 
match_full_df[["over", "balls"]] = match_full_df['ball'].str.split('.', expand=True)
match_full_df["over"] = match_full_df["over"].astype(int) + 1
match_full_df["balls"] = match_full_df["balls"].astype(int)

print(match_full_df)

# b. Loop through all match ids and find batters order in each innings

# Create empty dataframe to hold fantasy point outputs from the for loop
bat_position_table = pd.DataFrame()
bat_position_table["player"] = []
bat_position_table["match_id"] = []
bat_position_table["innings"] = []
bat_position_table["bat_position"] = []

# List of all the past BBL match_ids    
match_id_list = match_full_df.match_id.unique()
innings_list = [1,2]

print(match_id_list)
print(innings_list)

# For loop for all past match ids
for a in match_id_list:
    
    # For loop for each innings of the game
    for b in innings_list:
        match_batting_df = match_full_df[match_full_df.match_id == a]
        match_batting_df = match_batting_df.sort_values(by = ['innings', 'over','balls'])
        match_inn_bat_df = match_batting_df[match_batting_df.innings == b]
        match_bat_order = match_inn_bat_df['striker'].unique()

        inning_bat_order_df = pd.DataFrame(match_bat_order)
        inning_bat_order_df.columns = ['player']
        inning_bat_order_df['match_id'] = a
        inning_bat_order_df['innings'] = b
        inning_bat_order_df["bat_position"] = np.arange(len(inning_bat_order_df)) + 1 
        
        # Add all batting order attributes to empty table
        bat_position_table = pd.concat([bat_position_table, inning_bat_order_df])

# c. Save data set
bat_position_table.to_csv('data/python_datasets/all_match_bat_order.csv')

# Code End (Created dataframe with all players fantasy points for every BBL game) ---------------------------