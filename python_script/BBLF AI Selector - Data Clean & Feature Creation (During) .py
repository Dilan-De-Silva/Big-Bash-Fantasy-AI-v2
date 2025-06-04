# BBLF AI Selector: Part 2: During Tournament Optimal Squad - Data Clean & Feature Creation

# 0. Prerequistes
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder

os.getcwd()
directory = 'C:/Users/dilan/OneDrive/Documents/Data Science Projects/Big Bash Fantasy AI'

# 1. Data Extraction 
# 1a. Pull in all_matches csv file 

data_prep_df = pd.read_csv(os.path.join(directory,'data/python_datasets/fantasy_point_player_table.csv'), low_memory=False)

print(data_prep_df)

# 1b. Group & fix mis match venue names
unique_venues = data_prep_df['venue'].unique()
print(unique_venues)

data_prep_df['venue'] = data_prep_df['venue'].replace('Western Australia Cricket Association Ground', 'WACA')
data_prep_df['venue'] = data_prep_df['venue'].replace('W.A.C.A. Ground', 'WACA')
data_prep_df['venue'] = data_prep_df['venue'].replace('Brisbane Cricket Ground, Woolloongabba', 'GABBA')
data_prep_df['venue'] = data_prep_df['venue'].replace('Brisbane Cricket Ground', 'GABBA')
data_prep_df['venue'] = data_prep_df['venue'].replace('Brisbane Cricket Ground, Woolloongabba, Brisbane', 'GABBA')
data_prep_df['venue'] = data_prep_df['venue'].replace('Aurora Stadium', 'Launceston')
data_prep_df['venue'] = data_prep_df['venue'].replace('University of Tasmania Stadium, Launceston', 'Launceston')
data_prep_df['venue'] = data_prep_df['venue'].replace('Aurora Stadium, Launceston', 'Launceston')
data_prep_df['venue'] = data_prep_df['venue'].replace('Manuka Oval', 'Manuka')
data_prep_df['venue'] = data_prep_df['venue'].replace('Manuka Oval, Canberra', 'Manuka')
data_prep_df['venue'] = data_prep_df['venue'].replace('Docklands Stadium, Melbourne', 'Marvel')
data_prep_df['venue'] = data_prep_df['venue'].replace('Docklands Stadium', 'Marvel')
data_prep_df['venue'] = data_prep_df['venue'].replace('International Sports Stadium, Coffs Harbour', 'Coffs Harbour')
data_prep_df['venue'] = data_prep_df['venue'].replace('International Sports Stadium', 'Coffs Harbour')
data_prep_df['venue'] = data_prep_df['venue'].replace('GMHBA Stadium, South Geelong, Victoria', 'Geelong')
data_prep_df['venue'] = data_prep_df['venue'].replace('Geelong Cricket Ground', 'Geelong')
data_prep_df['venue'] = data_prep_df['venue'].replace('Simonds Stadium, South Geelong, Victoria', 'Geelong')
data_prep_df['venue'] = data_prep_df['venue'].replace('Sydney Cricket Ground', 'SCG')
data_prep_df['venue'] = data_prep_df['venue'].replace('Lavington Sports Oval, Albury', 'Albury')
data_prep_df['venue'] = data_prep_df['venue'].replace('North Sydney Oval, Sydney', 'North Sydney Oval')
data_prep_df['venue'] = data_prep_df['venue'].replace("Cazaly's Stadium, Cairns", 'Cairns')
data_prep_df['venue'] = data_prep_df['venue'].replace("Junction Oval, Melbourne", 'Junction Oval')
data_prep_df['venue'] = data_prep_df['venue'].replace("Sydney Showground Stadium", 'Sydney Showground')
data_prep_df['venue'] = data_prep_df['venue'].replace("Stadium Australia", 'Sydney Showground')
data_prep_df['venue'] = data_prep_df['venue'].replace("Traeger Park", 'Alice Springs')
data_prep_df['venue'] = data_prep_df['venue'].replace("Bellerive Oval, Hobart", 'Hobart')
data_prep_df['venue'] = data_prep_df['venue'].replace("Bellerive Oval", 'Hobart')
data_prep_df['venue'] = data_prep_df['venue'].replace("Ted Summerton Reserve", 'Moe')
data_prep_df['venue'] = data_prep_df['venue'].replace("Carrara Oval", 'Gold Coast')

data_prep_df['venue'] = data_prep_df['venue'].replace('WACA', "Perth Stadium")
data_prep_df['venue'] = data_prep_df['venue'].replace('Launceston', "Hobart")
data_prep_df['venue'] = data_prep_df['venue'].replace('Coffs Harbour', "Other")
data_prep_df['venue'] = data_prep_df['venue'].replace('Geelong', "Other")
data_prep_df['venue'] = data_prep_df['venue'].replace('Albury', "Other")
data_prep_df['venue'] = data_prep_df['venue'].replace('North Sydney Oval', "SCG")
data_prep_df['venue'] = data_prep_df['venue'].replace('Cairns', "Other")
data_prep_df['venue'] = data_prep_df['venue'].replace('Junction Oval', "Other")
data_prep_df['venue'] = data_prep_df['venue'].replace('Alice Springs', "Other")
data_prep_df['venue'] = data_prep_df['venue'].replace('Moe', "Other")
data_prep_df['venue'] = data_prep_df['venue'].replace('Gold Coast', "GABBA")
data_prep_df['venue'] = data_prep_df['venue'].replace('Manuka', "Other")

unique_venues = data_prep_df['venue'].unique()
print(unique_venues)

unique_venues = data_prep_df['team'].unique()
print(unique_venues)

# 1c. Team Home and Away Flag

print(data_prep_df)

home_conditions = [
        (data_prep_df['venue'] == "Other"),
        (data_prep_df['venue'] == "GABBA") & (data_prep_df['team'] == "Brisbane Heat"),
        (data_prep_df['venue'] == "Melbourne Cricket Ground") & (data_prep_df['team'] == "Melbourne Stars"),
        (data_prep_df['venue'] == "Hobart") & (data_prep_df['team'] == "Hobart Hurricanes"),
        (data_prep_df['venue'] == "Sydney Showground") & (data_prep_df['team'] == "Sydney Thunder"),
        (data_prep_df['venue'] == "Adelaide Oval") & (data_prep_df['team'] == "Adelaide Strikers"),
        (data_prep_df['venue'] == "SCG") & (data_prep_df['team'] == "Sydney Sixers"),
        (data_prep_df['venue'] == "Marvel") & (data_prep_df['team'] == "Melbourne Renegades"),
        (data_prep_df['venue'] == "Perth Stadium") & (data_prep_df['team'] == "Perth Scorchers")
    ]

home_group = [0,1,1,1,1,1,1,1,1]

data_prep_df["Home_f"] = np.select(home_conditions, home_group)

# 2. Create Season FP Stats

season_df = data_prep_df[["player", "fantasy_point", "season"]]
season_df_agg = season_df.groupby(['player', 'season'], as_index=False).agg(
season_fp =('fantasy_point',"sum"),
avg_season_fp = ('fantasy_point', "mean"),
max_season_fp = ('fantasy_point', 'max'),
min_season_fp = ('fantasy_point', 'min'),
med_season_fp = ('fantasy_point', 'median'),
sd_season_fp = ('fantasy_point', 'std'),
match_cnt = ('fantasy_point', 'count'))

# 3. Select Response Variable
# 3a. Current Response Variable: Average Season FP

# resp_avg_df = season_df_agg[["player", "season", "avg_season_fp"]].rename(columns={"avg_season_fp": "resp_var"})
# print(resp_avg_df)
# resp_df = resp_avg_df

# 3b. Other Ideas: Expected Fantasy Points (Calculate for each game and add them)
resp_ind_game_df = data_prep_df[["match_id", "player", "season", "fantasy_point", "team", "opp"]].rename(columns = {"fantasy_point": "resp_var"})
resp_df = resp_ind_game_df

print(resp_df)

# 4. Feature Creation
    # 4a. Previous Season/s Fantasy Point Summary Stats 
    # Lag 1
season_df_lag1_stat = season_df_agg.copy()
season_df_lag1_stat['season'] = season_df_lag1_stat['season'] + 1
season_df_lag1_stat = season_df_lag1_stat.rename(columns={c:c+'_lag1' for c in season_df_lag1_stat.columns if c not in ['player', 'season']})
print(season_df_lag1_stat)

fant_model_df = pd.merge(resp_df, season_df_lag1_stat, left_on = ["player", "season"], right_on = ["player", "season"], how = "left")
print(fant_model_df)

    # Lag 2
season_df_lag2_stat = season_df_agg.copy()
season_df_lag2_stat['season'] = season_df_lag2_stat['season'] + 2
season_df_lag2_stat = season_df_lag2_stat.rename(columns={c:c+'_lag2' for c in season_df_lag2_stat.columns if c not in ['player', 'season']})
print(season_df_lag2_stat)

fant_model_df = pd.merge(fant_model_df, season_df_lag2_stat, left_on = ["player", "season"], right_on = ["player", "season"], how = "left")
print(fant_model_df)

    # Lag 3
season_df_lag3_stat = season_df_agg.copy()
season_df_lag3_stat['season'] = season_df_lag3_stat['season'] + 3
season_df_lag3_stat = season_df_lag3_stat.rename(columns={c:c+'_lag3' for c in season_df_lag3_stat.columns if c not in ['player', 'season']})
print(season_df_lag3_stat)

fant_model_df = pd.merge(fant_model_df , season_df_lag3_stat, left_on = ["player", "season"], right_on = ["player", "season"], how = "left")
print(fant_model_df)

    # 4b. Individual Match Features
    # Match Venue
match_fact_df = data_prep_df[["match_id","player", "venue", "Home_f"]]
fant_model_df = pd.merge(fant_model_df , match_fact_df, left_on = ["match_id", "player"], right_on = ["match_id", "player"], how = "left")

print(fant_model_df)

    # Overs Bowled
over_bowled_df = pd.read_csv(os.path.join(directory,'data/python_datasets/during_tourny_overs_bowl.csv'), low_memory=False).rename(columns = {"bowler":"player"}).drop(["Unnamed: 0"], axis=1)
fant_model_df = pd.merge(fant_model_df , over_bowled_df, left_on = ["match_id", "player"], right_on = ["match_id", "player"], how = "left")

print(fant_model_df)

    # Batting Position
bat_pos_df = pd.read_csv(os.path.join(directory,'data/python_datasets/all_match_bat_order.csv'), low_memory=False).drop(["Unnamed: 0"], axis=1)
fant_model_df = pd.merge(fant_model_df , bat_pos_df, left_on = ["match_id", "player"], right_on = ["match_id", "player"], how = "left")

print(fant_model_df)

    # During Tourny fantasy pts features
dur_tourny_fp_df = pd.read_csv(os.path.join(directory,'data/python_datasets/dur_tourny_fant_pts_attr_table.csv'), low_memory=False).drop(["Unnamed: 0"], axis=1)
fant_model_df = pd.merge(fant_model_df , dur_tourny_fp_df, left_on = ["match_id", "player","season"], right_on = ["match_id", "player","season"], how = "left")

    # 4c. Opponent Season Power Ranking
team_rank_df = pd.read_csv(os.path.join(directory,'data/python_datasets/team_season_rank.csv'), low_memory=False).rename(columns = {"team": "opp"})
fant_model_df = pd.merge(fant_model_df , team_rank_df, left_on = ["season", "opp"], right_on = ["season", "opp"], how = "left")
print(fant_model_df)

# 5. One Hot Encoding of factors
fant_model_df = pd.get_dummies(fant_model_df, columns= ['opp','venue','rank_group','bat_position'])

fant_model_df.loc[:,"opp_Adelaide Strikers":"opp_Sydney Thunder"] = fant_model_df.loc[:,"opp_Adelaide Strikers":"opp_Sydney Thunder"].astype(int)
fant_model_df.loc[:,"opp_Adelaide Strikers":"opp_Sydney Thunder"] = fant_model_df.loc[:,"opp_Adelaide Strikers":"opp_Sydney Thunder"].astype(object)

fant_model_df.loc[:,"venue_Adelaide Oval":"venue_Sydney Showground"] = fant_model_df.loc[:,"venue_Adelaide Oval":"venue_Sydney Showground"].astype(int)
fant_model_df.loc[:,"venue_Adelaide Oval":"venue_Sydney Showground"] = fant_model_df.loc[:,"venue_Adelaide Oval":"venue_Sydney Showground"].astype(object)

# fant_model_df.loc[:,"rank_1":"rank_8"] = fant_model_df.loc[:,"rank_1":"rank_8"].astype(int)
# fant_model_df.loc[:,"rank_1":"rank_8"] = fant_model_df.loc[:,"rank_1":"rank_8"].astype(object)

fant_model_df.loc[:,"rank_group_High":"rank_group_Middle"] = fant_model_df.loc[:,"rank_group_High":"rank_group_Middle"].astype(int)
fant_model_df.loc[:,"rank_group_High":"rank_group_Middle"] = fant_model_df.loc[:,"rank_group_High":"rank_group_Middle"].astype(object)

fant_model_df.loc[:,"bat_position_1.0":"bat_position_11.0"] = fant_model_df.loc[:,"bat_position_1.0":"bat_position_11.0"].astype(int)
fant_model_df.loc[:,"bat_position_1.0":"bat_position_11.0"] = fant_model_df.loc[:,"bat_position_1.0":"bat_position_11.0"].astype(object)
fant_model_df['bat_f'] = fant_model_df.loc[:,"bat_position_1.0":"bat_position_11.0"].sum(axis=1)
fant_model_df['bat_position_DNB'] = np.where(fant_model_df['bat_f'] == 1, 0, 1)
fant_model_df = fant_model_df.drop(["bat_f"], axis = 1)

print(fant_model_df)

# 6. Interaction Variables
    # 6a. Venue & Home Flag (See if the certain teams have bigger home ground advantage)
fant_model_df["Home_Adelaide Strikers"] = np.where((fant_model_df["venue_Adelaide Oval"] == 1) & (fant_model_df["Home_f"] == 1) , 1 , 0)
fant_model_df["Home_Melbourne Stars"] = np.where((fant_model_df["venue_Melbourne Cricket Ground"] == 1) & (fant_model_df["Home_f"] == 1) , 1 , 0)
fant_model_df["Home_Melbourne Renegades"] = np.where((fant_model_df["venue_Marvel"] == 1) & (fant_model_df["Home_f"] == 1) , 1 , 0)
fant_model_df["Home_Brisbane Heat"] = np.where((fant_model_df["venue_GABBA"] == 1) & (fant_model_df["Home_f"] == 1) , 1 , 0)
fant_model_df["Home_Perth Scorchers"] = np.where((fant_model_df["venue_Perth Stadium"] == 1) & (fant_model_df["Home_f"] == 1) , 1 , 0)
fant_model_df["Home_Sydney Sixers"] = np.where((fant_model_df["venue_SCG"] == 1) & (fant_model_df["Home_f"] == 1) , 1 , 0)
fant_model_df["Home_Sydney Thunder"] = np.where((fant_model_df["venue_Sydney Showground"] == 1) & (fant_model_df["Home_f"] == 1) , 1 , 0)
fant_model_df["Home_Hobart Hurricanes"] = np.where((fant_model_df["venue_Hobart"] == 1) & (fant_model_df["Home_f"] == 1) , 1 , 0)

fant_model_df.loc[:,"Home_Adelaide Strikers":"Home_Hobart Hurricanes"] = fant_model_df.loc[:,"Home_Adelaide Strikers":"Home_Hobart Hurricanes"].astype(object)

print(fant_model_df)
fant_model_df.dtypes

# 7. Final Model Dataset
# Adjust 0 points to very small non 0 number 

#fant_model_df["resp_var"] = fant_model_df["resp_var"].replace(0, 0.01)

# Remove first three season (due to lag variables up to 3 seasons prior. Also remove as over 10 years old)
# fant_model_df = fant_model_df[fant_model_df["season"] > 3] 
# Remove first ten season (due to power surge being introduced in season 10)
fant_model_df = fant_model_df[fant_model_df["season"] > 9]    

fant_model_df.to_csv('data/python_datasets/bblf_during_FS_PowerSurge.csv')

# 8. Create BBL14 Data Features for Prediction
    # 8a. Lag Variables (incl latest BBL data for players who have played BBL but the previous season)
player_df = pd.read_csv(os.path.join(directory,'data/python_datasets/player_price.csv'), low_memory=False)
player_df = player_df[["player", "Full_Name"]]
print(player_df)

    # Latest BBL player records
 # 1. For loop for each distinct player to find max season
 
player_list = season_df_lag1_stat.player.unique()
play_last_season = []

for x in player_list:

    # select individual player records 
    ind_play_lag_df = season_df_lag1_stat[season_df_lag1_stat["player"] == x]
    ind_play_last_season = ind_play_lag_df[["season"]].sort_values(by='season', ascending=False).iloc[0]
    ind_play_last_season = max(ind_play_last_season)
    play_last_season = play_last_season + [ind_play_last_season]

join_list = {'player': player_list, 'season': play_last_season}
print(join_list)

bbl14_play_lags = pd.DataFrame(join_list)

all_play_latest_season_lags = pd.merge(bbl14_play_lags , season_df_lag1_stat, left_on = ["player","season"], right_on = ["player","season"], how = "left")
all_play_latest_season_lags = pd.merge(all_play_latest_season_lags , season_df_lag2_stat, left_on = ["player","season"], right_on = ["player","season"], how = "left")
all_play_latest_season_lags = pd.merge(all_play_latest_season_lags , season_df_lag3_stat, left_on = ["player","season"], right_on = ["player","season"], how = "left")
all_play_latest_season_lags = all_play_latest_season_lags.drop(["season"], axis = 1)
print(all_play_latest_season_lags)

# Join all_player_latest_season_lags to the bbl14 player list
season_df_lag_14 = pd.merge(player_df, all_play_latest_season_lags, left_on = ["player"], right_on = ["player"], how = "left")

season_df_lag_14.to_csv('data/python_datasets/bbl14_lags.csv')
