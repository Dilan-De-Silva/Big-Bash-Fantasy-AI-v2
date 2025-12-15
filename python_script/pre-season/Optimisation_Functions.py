# Optimisation Functions
# Imports
import pandas as pd
import numpy as np
import os
import random

# Player Rolling Price Function
def roll_rnd_price_fn(player_df_init, price_df, current_rnd, 
                      price_model_obj_1, price_model_obj_2, price_model_obj_3):
    
    # Imports
    import pandas as pd
    import numpy as np

    # Copy player init df to avoid modifying original
    player_df_init_2 = player_df_init.copy()

    # a. Derive Features required for Pricing Models
    # Aggregate by player name per round
    player_df = player_df_init_2.groupby(['Name', 'Price', "Team", "Round", "Wk_f", "Bat_f", "Bowl_f", "Role", "weight","Available", "In_Team"], as_index=False).agg(
    exp_rnd_points=('exp_points',"sum"),
    games_in_round=('Round',"count"))

    # Create game_num variable
    pts_per_game = player_df_init_2[['Name', 'game_num', 'exp_points']].sort_values(['Name', 'game_num'])
    
    # Vectorized feature calculation using groupby and transform
    grouped = pts_per_game.groupby('Name')['exp_points']
    
    # Calculate all rolling features at once
    pts_per_game['curr_game_pts'] = pts_per_game['exp_points']
    pts_per_game['prev_game_pts'] = grouped.shift(1)
    pts_per_game['two_prev_game_pts'] = grouped.shift(2)
    pts_per_game['last_2_games_ma_pts'] = grouped.transform(lambda x: x.rolling(2, min_periods=1).mean())
    pts_per_game['last_3_games_ma_pts'] = grouped.transform(lambda x: x.rolling(3, min_periods=1).mean())
    pts_per_game['seas_avg_games_pts'] = grouped.transform(lambda x: x.expanding().mean())
    
    # Drop the original exp_points column and keep calculated features
    bbl15_game_pts_table_pre = pts_per_game.drop('exp_points', axis=1)

    # Add team and round info
    player_team = player_df_init_2[['Name', 'Team']].drop_duplicates()
    game_rnd_team_df = player_df_init_2[['Team','Round','game_num']].drop_duplicates()
    
    bbl15_game_pts_table = bbl15_game_pts_table_pre.merge(player_team, on='Name').merge(game_rnd_team_df, on=['Team', 'game_num']).drop('Team', axis=1)

    # For player double gameweek rounds, only return the second game row
    # Keep only max game_num per player per round
    bbl15_game_pts_table = bbl15_game_pts_table.loc[
        bbl15_game_pts_table.groupby(['Name', 'Round'])['game_num'].idxmax()
    ]

    bbl15_game_pts_table['Round'] = bbl15_game_pts_table['Round'].astype("Int64")
    bbl15_game_pts_table = bbl15_game_pts_table.sort_values(['Name','Round'])

    # Add player price
    bbl15_game_pts_table = pd.merge(bbl15_game_pts_table, price_df[['Name', 'Price']], on = 'Name', how = 'left').rename(columns={"Price":"price_pre"})

    # Price prediction - OPTIMIZED with bulk predictions
    player_df_lags = bbl15_game_pts_table[['Name', 'Round', 'game_num', 'price_pre', 'seas_avg_games_pts', 'last_2_games_ma_pts', 'last_3_games_ma_pts']]
    
    # Pre-compute all predictions in bulk
    player_df_lags['Price_Pred'] = np.nan
    
    # Game 1 predictions
    mask_g1 = player_df_lags['game_num'] == 1
    if mask_g1.any():
        player_df_lags.loc[mask_g1, 'Price_Pred'] = price_model_obj_1.predict(
            player_df_lags.loc[mask_g1, ['price_pre', 'seas_avg_games_pts']]
        )
    
    # Game 2 predictions
    mask_g2 = player_df_lags['game_num'] == 2
    if mask_g2.any():
        player_df_lags.loc[mask_g2, 'Price_Pred'] = price_model_obj_2.predict(
            player_df_lags.loc[mask_g2, ['price_pre', 'last_2_games_ma_pts']]
        )
    
    # Game 3+ predictions
    mask_g3 = player_df_lags['game_num'] >= 3
    if mask_g3.any():
        player_df_lags.loc[mask_g3, 'Price_Pred'] = price_model_obj_3.predict(
            player_df_lags.loc[mask_g3, ['price_pre', 'last_3_games_ma_pts']]
        )
    
    # b. Build price dataframe with rolling predictions
    player_df_new_list = []
    
    for player in player_df['Name'].unique():
        player_lags = player_df_lags[player_df_lags['Name'] == player].sort_values('Round')
        player_rounds = player_lags['Round'].values
        
        # Current round - use actual price
        curr_mask = player_df['Name'] == player
        curr_data = player_df[curr_mask & (player_df['Round'] == current_rnd)].copy()
        if not curr_data.empty:
            player_df_new_list.append(curr_data)
            last_known_price = curr_data['Price'].values[0]
        else:
            # If no current round data, get price from player_df
            last_known_price = player_df[curr_mask]['Price'].iloc[0] if not player_df[curr_mask].empty else None
        
        # Future rounds - use predicted prices only if player was available in current round
        for i, rnd in enumerate(player_rounds[:-1]):
            next_rnd = player_rounds[i + 1]
            
            future_data = player_df[curr_mask & (player_df['Round'] == next_rnd)].copy()
            if not future_data.empty:
                # Check if player was available in the current round (rnd)
                curr_rnd_data = player_df[curr_mask & (player_df['Round'] == rnd)]
                if not curr_rnd_data.empty and curr_rnd_data['Available'].values[0] == 1:
                    # Player was available - apply predicted price change
                    pred_price = player_lags[player_lags['Round'] == rnd]['Price_Pred'].values[0]
                    future_data['Price'] = pred_price
                    last_known_price = pred_price
                else:
                    # Player was not available - keep last known price (no change)
                    future_data['Price'] = last_known_price
                player_df_new_list.append(future_data)
    
    if player_df_new_list:
        player_df = pd.concat(player_df_new_list, ignore_index=True)
    else:
        player_df = pd.DataFrame()

    # c. Prepare Player DataFrames for optimisation
    # Create additional rows for players who do not play in every round
    all_rounds = list(range(int(player_df['Round'].min()), int(player_df['Round'].max()) + 1))
    all_players = player_df['Name'].unique()
    full_index = pd.MultiIndex.from_product([all_players, all_rounds], names=['Name', 'Round'])
    player_df = player_df.set_index(['Name', 'Round']).reindex(full_index).reset_index()
    player_df['Price'] = player_df['Price'].ffill()
    player_df['Team'] = player_df['Team'].ffill()
    player_df['Wk_f'] = player_df['Wk_f'].ffill()
    player_df['Bat_f'] = player_df['Bat_f'].ffill()
    player_df['Bowl_f'] = player_df['Bowl_f'].ffill()
    player_df['Role'] = player_df['Role'].ffill()
    player_df['weight'] = player_df['weight'].ffill()
    player_df['exp_rnd_points'] = player_df['exp_rnd_points'].fillna(-100)
    player_df['games_in_round'] = player_df['games_in_round'].fillna(0)
    player_df['Available'] = np.where(player_df['exp_rnd_points'] == -100, 0, player_df['Available'])
    player_df['In_Team'] = player_df['In_Team'].ffill()

    # Split player df by round
    player_dfs = {i: player_df[player_df['Round'] == i].reset_index(drop=True) for i in range(1, 10)}
    player_df_r1, player_df_r2, player_df_r3 = player_dfs[1], player_dfs[2], player_dfs[3]
    player_df_r4, player_df_r5, player_df_r6 = player_dfs[4], player_dfs[5], player_dfs[6]
    player_df_r7, player_df_r8, player_df_r9 = player_dfs[7], player_dfs[8], player_dfs[9]

    # Drop intermediate variables to free memory
    del player_df_init_2, player_df, pts_per_game, bbl15_game_pts_table_pre, bbl15_game_pts_table, player_team, game_rnd_team_df, player_df_lags, player_df_new_list

    # Return player dfs for each round
    return player_df_r1, player_df_r2, player_df_r3, player_df_r4, player_df_r5, player_df_r6, player_df_r7, player_df_r8, player_df_r9
    

# EFP Optimisation Function
def optimise_fn_efp(
        # Round 1
        points_r1, price_r1, weight_r1, in_team_r1, available_r1, wk_weight_r1, bat_weight_r1, bowl_weight_r1,
        play_cnt_r1, total_player_r1, wk_cnt_r1, total_wk_r1, bat_cnt_r1, total_bat_r1, bowl_cnt_r1, total_bowl_r1,
        budget_r1, total_budget_r1, player_df_r1, cnt_r1, max_player_r1,  
        # Round 2
        points_r2, price_r2, weight_r2, in_team_r2, available_r2, wk_weight_r2, bat_weight_r2, bowl_weight_r2,
        play_cnt_r2, total_player_r2, wk_cnt_r2, total_wk_r2, bat_cnt_r2, total_bat_r2, bowl_cnt_r2, total_bowl_r2,
        budget_r2, total_budget_r2, team_play_cnt_r2, total_team_player_r2, player_df_r2, cnt_r2, max_player_r2,
        # Round 3
        points_r3, price_r3, weight_r3, in_team_r3, available_r3, wk_weight_r3, bat_weight_r3, bowl_weight_r3,
        play_cnt_r3, total_player_r3, wk_cnt_r3, total_wk_r3, bat_cnt_r3, total_bat_r3, bowl_cnt_r3, total_bowl_r3,
        budget_r3, total_budget_r3, team_play_cnt_r3, total_team_player_r3, player_df_r3, cnt_r3, max_player_r3,
        # Round 4
        points_r4, price_r4, weight_r4, in_team_r4, available_r4, wk_weight_r4, bat_weight_r4, bowl_weight_r4,
        play_cnt_r4, total_player_r4, wk_cnt_r4, total_wk_r4, bat_cnt_r4, total_bat_r4, bowl_cnt_r4, total_bowl_r4,
        budget_r4, total_budget_r4, team_play_cnt_r4, total_team_player_r4, player_df_r4, cnt_r4, max_player_r4,
        # Round 5
        points_r5, price_r5, weight_r5, in_team_r5, available_r5, wk_weight_r5, bat_weight_r5, bowl_weight_r5,
        play_cnt_r5, total_player_r5, wk_cnt_r5, total_wk_r5, bat_cnt_r5, total_bat_r5, bowl_cnt_r5, total_bowl_r5,
        budget_r5, total_budget_r5, team_play_cnt_r5, total_team_player_r5, player_df_r5, cnt_r5, max_player_r5,
        # Round 6
        points_r6, price_r6, weight_r6, in_team_r6, available_r6, wk_weight_r6, bat_weight_r6, bowl_weight_r6,
        play_cnt_r6, total_player_r6, wk_cnt_r6, total_wk_r6, bat_cnt_r6, total_bat_r6, bowl_cnt_r6, total_bowl_r6,
        budget_r6, total_budget_r6, team_play_cnt_r6, total_team_player_r6, player_df_r6, cnt_r6, max_player_r6,
        # Round 7
        points_r7, price_r7, weight_r7, in_team_r7, available_r7, wk_weight_r7, bat_weight_r7, bowl_weight_r7,
        play_cnt_r7, total_player_r7, wk_cnt_r7, total_wk_r7, bat_cnt_r7, total_bat_r7, bowl_cnt_r7, total_bowl_r7,
        budget_r7, total_budget_r7, team_play_cnt_r7, total_team_player_r7, player_df_r7, cnt_r7, max_player_r7,
        # Round 8
        points_r8, price_r8, weight_r8, in_team_r8, available_r8, wk_weight_r8, bat_weight_r8, bowl_weight_r8,
        play_cnt_r8, total_player_r8, wk_cnt_r8, total_wk_r8, bat_cnt_r8, total_bat_r8, bowl_cnt_r8, total_bowl_r8,
        budget_r8, total_budget_r8, team_play_cnt_r8, total_team_player_r8, player_df_r8, cnt_r8, max_player_r8,
        # Round 9
        points_r9, price_r9, weight_r9, in_team_r9, available_r9, wk_weight_r9, bat_weight_r9, bowl_weight_r9,
        play_cnt_r9, total_player_r9, wk_cnt_r9, total_wk_r9, bat_cnt_r9, total_bat_r9, bowl_cnt_r9, total_bowl_r9,
        budget_r9, total_budget_r9, team_play_cnt_r9, total_team_player_r9, player_df_r9, cnt_r9, max_player_r9):
    
    # Imports
    from mip import Model, xsum, maximize, BINARY

    # a. initialize optimisation parameters
    m = Model("knapsack")  # Using HiGHS solver for better performance
    # x[i] = 1 if player i is selected in roster (13 total)
    # p[i] = 1 if player i is playing (12 total, subset of x)
    # y[i] = 1 if player i is captain (1 total, subset of p)
    # Bench player = x[i]=1 but p[i]=0 (automatically the lowest value among selected)
    
    # Round 1
    x_r1 = [m.add_var(var_type=BINARY) for i in total_player_r1]  # Selected (13)
    p_r1 = [m.add_var(var_type=BINARY) for i in total_player_r1]  # Playing (12)
    y_r1 = [m.add_var(var_type=BINARY) for i in total_player_r1]  # Captain (1)
    # Round 2
    x_r2 = [m.add_var(var_type=BINARY) for i in total_player_r2]
    p_r2 = [m.add_var(var_type=BINARY) for i in total_player_r2]
    y_r2 = [m.add_var(var_type=BINARY) for i in total_player_r2]
    # Round 3
    x_r3 = [m.add_var(var_type=BINARY) for i in total_player_r3]
    p_r3 = [m.add_var(var_type=BINARY) for i in total_player_r3]
    y_r3 = [m.add_var(var_type=BINARY) for i in total_player_r3]
    # Round 4
    x_r4 = [m.add_var(var_type=BINARY) for i in total_player_r4]
    p_r4 = [m.add_var(var_type=BINARY) for i in total_player_r4]
    y_r4 = [m.add_var(var_type=BINARY) for i in total_player_r4]
    # Round 5
    x_r5 = [m.add_var(var_type=BINARY) for i in total_player_r5]
    p_r5 = [m.add_var(var_type=BINARY) for i in total_player_r5]
    y_r5 = [m.add_var(var_type=BINARY) for i in total_player_r5]
    # Round 6
    x_r6 = [m.add_var(var_type=BINARY) for i in total_player_r6]
    p_r6 = [m.add_var(var_type=BINARY) for i in total_player_r6]
    y_r6 = [m.add_var(var_type=BINARY) for i in total_player_r6]
    # Round 7
    x_r7 = [m.add_var(var_type=BINARY) for i in total_player_r7]
    p_r7 = [m.add_var(var_type=BINARY) for i in total_player_r7]
    y_r7 = [m.add_var(var_type=BINARY) for i in total_player_r7]
    # Round 8
    x_r8 = [m.add_var(var_type=BINARY) for i in total_player_r8]
    p_r8 = [m.add_var(var_type=BINARY) for i in total_player_r8]
    y_r8 = [m.add_var(var_type=BINARY) for i in total_player_r8]
    # Round 9
    x_r9 = [m.add_var(var_type=BINARY) for i in total_player_r9]
    p_r9 = [m.add_var(var_type=BINARY) for i in total_player_r9]
    y_r9 = [m.add_var(var_type=BINARY) for i in total_player_r9]

    # linking constraints between consecutive rounds
    # Initialise z_shared for round 1 to round 2 linking
    name_to_idx_r1 = {name: idx for idx, name in enumerate(player_df_r1['Name'].astype(str))}
    z_shared = {}
    for j, name in enumerate(player_df_r2['Name'].astype(str)):
        if name in name_to_idx_r1:
            i = name_to_idx_r1[name]
            z_shared[(i, j)] = m.add_var(var_type=BINARY)
    
    # Initialise Trade Boost Variable
    extra_trade_vars = []
            
    rounds_data = [
        (player_df_r1, player_df_r2, x_r1, x_r2, z_shared, team_play_cnt_r2, "r1_r2"),
        (player_df_r2, player_df_r3, x_r2, x_r3, {}, team_play_cnt_r3, "r2_r3"),
        (player_df_r3, player_df_r4, x_r3, x_r4, {}, team_play_cnt_r4, "r3_r4"),
        (player_df_r4, player_df_r5, x_r4, x_r5, {}, team_play_cnt_r5, "r4_r5"),
        (player_df_r5, player_df_r6, x_r5, x_r6, {}, team_play_cnt_r6, "r5_r6"),
        (player_df_r6, player_df_r7, x_r6, x_r7, {}, team_play_cnt_r7, "r6_r7"),
        (player_df_r7, player_df_r8, x_r7, x_r8, {}, team_play_cnt_r8, "r7_r8"),
        (player_df_r8, player_df_r9, x_r8, x_r9, {}, team_play_cnt_r9, "r8_r9"),
    ]
    
    for curr_df, next_df, curr_x, next_x, z_dict, team_min, label in rounds_data:
        name_to_idx_curr = {name: idx for idx, name in enumerate(curr_df['Name'].astype(str))}
        shared_pairs_rnd = []
        for j, name in enumerate(next_df['Name'].astype(str)):
            if name in name_to_idx_curr:
                i = name_to_idx_curr[name]
                shared_pairs_rnd.append((i, j))
        
        if shared_pairs_rnd:
            z_rnd = {}
            for (i, j) in shared_pairs_rnd:
                z_rnd[(i, j)] = m.add_var(var_type=BINARY)
                m += z_rnd[(i, j)] <= curr_x[i]
                m += z_rnd[(i, j)] <= next_x[j]
                m += z_rnd[(i, j)] >= curr_x[i] + next_x[j] - 1

            # Trade Boost Constraint (4 Trades Allowed for 2 Rounds (Only 9 In Team Players Required))
            extra_trade = m.add_var(var_type=BINARY)
            extra_trade_vars.append(extra_trade)

            # If Trade Boost is used, allow 9 players from previous round, else allow 10
            m += xsum(z_rnd[(i, j)] for (i, j) in z_rnd.keys()) >= team_min - extra_trade

    # Only Two Trade Boosts Allowed in season
    m += xsum(extra_trade_vars for extra_trade_vars in extra_trade_vars) <= 2 

    # b. define objective function
    # Only playing players (p) contribute points; bench player (x=1, p=0) contributes 0 automatically
    obj_r1 = xsum(points_r1[i]*p_r1[i] + points_r1[i]*y_r1[i] for i in total_player_r1)
    obj_r2 = xsum(points_r2[i]*p_r2[i] + points_r2[i]*y_r2[i] for i in total_player_r2)
    obj_r3 = xsum(points_r3[i]*p_r3[i] + points_r3[i]*y_r3[i] for i in total_player_r3)
    obj_r4 = xsum(points_r4[i]*p_r4[i] + points_r4[i]*y_r4[i] for i in total_player_r4)
    obj_r5 = xsum(points_r5[i]*p_r5[i] + points_r5[i]*y_r5[i] for i in total_player_r5)
    obj_r6 = xsum(points_r6[i]*p_r6[i] + points_r6[i]*y_r6[i] for i in total_player_r6)
    obj_r7 = xsum(points_r7[i]*p_r7[i] + points_r7[i]*y_r7[i] for i in total_player_r7)
    obj_r8 = xsum(points_r8[i]*p_r8[i] + points_r8[i]*y_r8[i] for i in total_player_r8)
    obj_r9 = xsum(points_r9[i]*p_r9[i] + points_r9[i]*y_r9[i] for i in total_player_r9)

    m.objective = maximize(obj_r1 + obj_r2 + obj_r3 + obj_r4 + obj_r5 + obj_r6 + obj_r7 + obj_r8 + obj_r9)
    
    # c. define constraints
        # Player selection constraints
    # x = selected (13 total), p = playing (12 total), y = captain (1 total)
    # Bench player is automatically x=1, p=0 (the 13th selected player not playing)
    
    # Round 1
    m += xsum(x_r1[i] for i in total_player_r1) == cnt_r1  # total squad players
    m += xsum(weight_r1[i] * p_r1[i] for i in total_player_r1) == play_cnt_r1  # 12 playing players
    m += xsum(y_r1[i] for i in total_player_r1) == 1  # Only one captain
    # Round 2
    m += xsum(x_r2[i] for i in total_player_r2) == cnt_r2
    m += xsum(weight_r2[i] * p_r2[i] for i in total_player_r2) == play_cnt_r2
    m += xsum(y_r2[i] for i in total_player_r2) == 1
    # Round 3
    m += xsum(x_r3[i] for i in total_player_r3) == cnt_r3
    m += xsum(weight_r3[i] * p_r3[i] for i in total_player_r3) == play_cnt_r3
    m += xsum(y_r3[i] for i in total_player_r3) == 1
    # Round 4
    m += xsum(x_r4[i] for i in total_player_r4) == cnt_r4
    m += xsum(weight_r4[i] * p_r4[i] for i in total_player_r4) == play_cnt_r4
    m += xsum(y_r4[i] for i in total_player_r4) == 1
    # Round 5
    m += xsum(x_r5[i] for i in total_player_r5) == cnt_r5
    m += xsum(weight_r5[i] * p_r5[i] for i in total_player_r5) == play_cnt_r5
    m += xsum(y_r5[i] for i in total_player_r5) == 1
    # Round 6
    m += xsum(x_r6[i] for i in total_player_r6) == cnt_r6
    m += xsum(weight_r6[i] * p_r6[i] for i in total_player_r6) == play_cnt_r6
    m += xsum(y_r6[i] for i in total_player_r6) == 1
    # Round 7
    m += xsum(x_r7[i] for i in total_player_r7) == cnt_r7
    m += xsum(weight_r7[i] * p_r7[i] for i in total_player_r7) == play_cnt_r7
    m += xsum(y_r7[i] for i in total_player_r7) == 1
    # Round 8
    m += xsum(x_r8[i] for i in total_player_r8) == cnt_r8
    m += xsum(weight_r8[i] * p_r8[i] for i in total_player_r8) == play_cnt_r8
    m += xsum(y_r8[i] for i in total_player_r8) == 1
    # Round 9
    m += xsum(x_r9[i] for i in total_player_r9) == cnt_r9
    m += xsum(weight_r9[i] * p_r9[i] for i in total_player_r9) == play_cnt_r9
    m += xsum(y_r9[i] for i in total_player_r9) == 1

    # Hierarchy constraints: p <= x (can only play if selected), y <= p (captain must be playing)
    for i in total_player_r1:
        m += p_r1[i] <= x_r1[i]
        m += y_r1[i] <= p_r1[i]
    for i in total_player_r2:
        m += p_r2[i] <= x_r2[i]
        m += y_r2[i] <= p_r2[i]
    for i in total_player_r3:
        m += p_r3[i] <= x_r3[i]
        m += y_r3[i] <= p_r3[i]
    for i in total_player_r4:
        m += p_r4[i] <= x_r4[i]
        m += y_r4[i] <= p_r4[i]
    for i in total_player_r5:
        m += p_r5[i] <= x_r5[i]
        m += y_r5[i] <= p_r5[i]
    for i in total_player_r6:
        m += p_r6[i] <= x_r6[i]
        m += y_r6[i] <= p_r6[i]
    for i in total_player_r7:
        m += p_r7[i] <= x_r7[i]
        m += y_r7[i] <= p_r7[i]
    for i in total_player_r8:
        m += p_r8[i] <= x_r8[i]
        m += y_r8[i] <= p_r8[i]
    for i in total_player_r9:
        m += p_r9[i] <= x_r9[i]
        m += y_r9[i] <= p_r9[i]

        # Team composition constraints (apply to playing players p, not all selected x)
    # Round 1  
    m += xsum(wk_weight_r1[i] * p_r1[i] for i in total_wk_r1) >= wk_cnt_r1
    m += xsum(bat_weight_r1[i] * p_r1[i] for i in total_bat_r1) >= bat_cnt_r1
    m += xsum(bowl_weight_r1[i] * p_r1[i] for i in total_bowl_r1) >= bowl_cnt_r1
    m += xsum(available_r1[i] * p_r1[i] for i in total_player_r1) == play_cnt_r1
    m += xsum(price_r1[i] * x_r1[i] for i in total_budget_r1) - 39500*(cnt_r1 - play_cnt_r1) <= budget_r1  # 13 selected players with 39500 discount
    # Round 2  
    m += xsum(wk_weight_r2[i] * p_r2[i] for i in total_wk_r2) >= wk_cnt_r2
    m += xsum(bat_weight_r2[i] * p_r2[i] for i in total_bat_r2) >= bat_cnt_r2
    m += xsum(bowl_weight_r2[i] * p_r2[i] for i in total_bowl_r2) >= bowl_cnt_r2
    m += xsum(available_r2[i] * p_r2[i] for i in total_player_r2) == play_cnt_r2
    m += xsum(price_r2[i] * x_r2[i] for i in total_budget_r2) - 39500*(cnt_r2 - play_cnt_r2) <= (
        budget_r1 + xsum((price_r2[i] - price_r1[i]) * x_r1[i] for i in total_player_r1)
    )
     # Round 3
    m += xsum(wk_weight_r3[i] * p_r3[i] for i in total_wk_r3) >= wk_cnt_r3
    m += xsum(bat_weight_r3[i] * p_r3[i] for i in total_bat_r3) >= bat_cnt_r3
    m += xsum(bowl_weight_r3[i] * p_r3[i] for i in total_bowl_r3) >= bowl_cnt_r3
    m += xsum(available_r3[i] * p_r3[i] for i in total_player_r3) == play_cnt_r3
    m += xsum(price_r3[i] * x_r3[i] for i in total_budget_r3) - 39500*(cnt_r3 - play_cnt_r3) <= (
        budget_r1
        + xsum((price_r2[i] - price_r1[i]) * x_r1[i] for i in total_player_r1)
        + xsum((price_r3[i] - price_r2[i]) * x_r2[i] for i in total_player_r2)
    )
    # Round 4
    m += xsum(wk_weight_r4[i] * p_r4[i] for i in total_wk_r4) >= wk_cnt_r4
    m += xsum(bat_weight_r4[i] * p_r4[i] for i in total_bat_r4) >= bat_cnt_r4
    m += xsum(bowl_weight_r4[i] * p_r4[i] for i in total_bowl_r4) >= bowl_cnt_r4
    m += xsum(available_r4[i] * p_r4[i] for i in total_player_r4) == play_cnt_r4
    m += xsum(price_r4[i] * x_r4[i] for i in total_budget_r4) - 39500*(cnt_r4 - play_cnt_r4) <= (
        budget_r1
        + xsum((price_r2[i] - price_r1[i]) * x_r1[i] for i in total_player_r1)
        + xsum((price_r3[i] - price_r2[i]) * x_r2[i] for i in total_player_r2)
        + xsum((price_r4[i] - price_r3[i]) * x_r3[i] for i in total_player_r3)
    )
    # Round 5
    m += xsum(wk_weight_r5[i] * p_r5[i] for i in total_wk_r5) >= wk_cnt_r5
    m += xsum(bat_weight_r5[i] * p_r5[i] for i in total_bat_r5) >= bat_cnt_r5
    m += xsum(bowl_weight_r5[i] * p_r5[i] for i in total_bowl_r5) >= bowl_cnt_r5
    m += xsum(available_r5[i] * p_r5[i] for i in total_player_r5) == play_cnt_r5
    m += xsum(price_r5[i] * x_r5[i] for i in total_budget_r5) - 39500*(cnt_r5 - play_cnt_r5) <= (
        budget_r1
        + xsum((price_r2[i] - price_r1[i]) * x_r1[i] for i in total_player_r1)
        + xsum((price_r3[i] - price_r2[i]) * x_r2[i] for i in total_player_r2)
        + xsum((price_r4[i] - price_r3[i]) * x_r3[i] for i in total_player_r3)
        + xsum((price_r5[i] - price_r4[i]) * x_r4[i] for i in total_player_r4)
    )
    # Round 6
    m += xsum(wk_weight_r6[i] * p_r6[i] for i in total_wk_r6) >= wk_cnt_r6
    m += xsum(bat_weight_r6[i] * p_r6[i] for i in total_bat_r6) >= bat_cnt_r6
    m += xsum(bowl_weight_r6[i] * p_r6[i] for i in total_bowl_r6) >= bowl_cnt_r6
    m += xsum(available_r6[i] * p_r6[i] for i in total_player_r6) == play_cnt_r6
    m += xsum(price_r6[i] * x_r6[i] for i in total_budget_r6) - 39500*(cnt_r6 - play_cnt_r6) <= (
        budget_r1
        + xsum((price_r2[i] - price_r1[i]) * x_r1[i] for i in total_player_r1)
        + xsum((price_r3[i] - price_r2[i]) * x_r2[i] for i in total_player_r2)
        + xsum((price_r4[i] - price_r3[i]) * x_r3[i] for i in total_player_r3)
        + xsum((price_r5[i] - price_r4[i]) * x_r4[i] for i in total_player_r4)
        + xsum((price_r6[i] - price_r5[i]) * x_r5[i] for i in total_player_r5)
    )
    # Round 7
    m += xsum(wk_weight_r7[i] * p_r7[i] for i in total_wk_r7) >= wk_cnt_r7
    m += xsum(bat_weight_r7[i] * p_r7[i] for i in total_bat_r7) >= bat_cnt_r7
    m += xsum(bowl_weight_r7[i] * p_r7[i] for i in total_bowl_r7) >= bowl_cnt_r7
    m += xsum(available_r7[i] * p_r7[i] for i in total_player_r7) == play_cnt_r7
    m += xsum(price_r7[i] * x_r7[i] for i in total_budget_r7) - 39500*(cnt_r7 - play_cnt_r7) <= (
        budget_r1
        + xsum((price_r2[i] - price_r1[i]) * x_r1[i] for i in total_player_r1)
        + xsum((price_r3[i] - price_r2[i]) * x_r2[i] for i in total_player_r2)
        + xsum((price_r4[i] - price_r3[i]) * x_r3[i] for i in total_player_r3)
        + xsum((price_r5[i] - price_r4[i]) * x_r4[i] for i in total_player_r4)
        + xsum((price_r6[i] - price_r5[i]) * x_r5[i] for i in total_player_r5)
        + xsum((price_r7[i] - price_r6[i]) * x_r6[i] for i in total_player_r6)
    )
    # Round 8
    m += xsum(wk_weight_r8[i] * p_r8[i] for i in total_wk_r8) >= wk_cnt_r8
    m += xsum(bat_weight_r8[i] * p_r8[i] for i in total_bat_r8) >= bat_cnt_r8
    m += xsum(bowl_weight_r8[i] * p_r8[i] for i in total_bowl_r8) >= bowl_cnt_r8
    m += xsum(available_r8[i] * p_r8[i] for i in total_player_r8) == play_cnt_r8
    m += xsum(price_r8[i] * x_r8[i] for i in total_budget_r8) - 39500*(cnt_r8 - play_cnt_r8) <= (
        budget_r1
        + xsum((price_r2[i] - price_r1[i]) * x_r1[i] for i in total_player_r1)
        + xsum((price_r3[i] - price_r2[i]) * x_r2[i] for i in total_player_r2)
        + xsum((price_r4[i] - price_r3[i]) * x_r3[i] for i in total_player_r3)
        + xsum((price_r5[i] - price_r4[i]) * x_r4[i] for i in total_player_r4)
        + xsum((price_r6[i] - price_r5[i]) * x_r5[i] for i in total_player_r5)
        + xsum((price_r7[i] - price_r6[i]) * x_r6[i] for i in total_player_r6)
        + xsum((price_r8[i] - price_r7[i]) * x_r7[i] for i in total_player_r7)
    )
    # Round 9
    m += xsum(wk_weight_r9[i] * p_r9[i] for i in total_wk_r9) >= wk_cnt_r9
    m += xsum(bat_weight_r9[i] * p_r9[i] for i in total_bat_r9) >= bat_cnt_r9
    m += xsum(bowl_weight_r9[i] * p_r9[i] for i in total_bowl_r9) >= bowl_cnt_r9
    m += xsum(available_r9[i] * p_r9[i] for i in total_player_r9) == play_cnt_r9
    m += xsum(price_r9[i] * x_r9[i] for i in total_budget_r9) - 39500*(cnt_r9 - play_cnt_r9) <= (
        budget_r1
        + xsum((price_r2[i] - price_r1[i]) * x_r1[i] for i in total_player_r1)
        + xsum((price_r3[i] - price_r2[i]) * x_r2[i] for i in total_player_r2)
        + xsum((price_r4[i] - price_r3[i]) * x_r3[i] for i in total_player_r3)
        + xsum((price_r5[i] - price_r4[i]) * x_r4[i] for i in total_player_r4)
        + xsum((price_r6[i] - price_r5[i]) * x_r5[i] for i in total_player_r5)
        + xsum((price_r7[i] - price_r6[i]) * x_r6[i] for i in total_player_r6)
        + xsum((price_r8[i] - price_r7[i]) * x_r7[i] for i in total_player_r7)
        + xsum((price_r9[i] - price_r8[i]) * x_r8[i] for i in total_player_r8)
    )

    # d. solve optimisation
    # Performance settings
    # m.threads = -1  # Use all available CPU cores
    # m.max_gap = 0.01  # Stop when within 1% of optimal (faster)
    m.optimize()
    # Round 1 - Extract selected (x=1), playing (p=1), benched (x=1 but p=0)
    all_selected_r1 = [i for i in total_player_r1 if x_r1[i].x >= 0.99]
    playing_r1 = [i for i in total_player_r1 if p_r1[i].x >= 0.99]
    benched_r1 = [i for i in all_selected_r1 if i not in playing_r1]
    captained_r1 = [i for i in total_player_r1 if y_r1[i].x >= 0.99]
    selected_r1 = playing_r1  # For backward compatibility with output

    # Round 2
    all_selected_r2 = [i for i in total_player_r2 if x_r2[i].x >= 0.99]
    playing_r2 = [i for i in total_player_r2 if p_r2[i].x >= 0.99]
    benched_r2 = [i for i in all_selected_r2 if i not in playing_r2]
    captained_r2 = [i for i in total_player_r2 if y_r2[i].x >= 0.99]
    selected_r2 = playing_r2

    # Round 3
    all_selected_r3 = [i for i in total_player_r3 if x_r3[i].x >= 0.99]
    playing_r3 = [i for i in total_player_r3 if p_r3[i].x >= 0.99]
    benched_r3 = [i for i in all_selected_r3 if i not in playing_r3]
    captained_r3 = [i for i in total_player_r3 if y_r3[i].x >= 0.99]
    selected_r3 = playing_r3

    # Round 4
    all_selected_r4 = [i for i in total_player_r4 if x_r4[i].x >= 0.99]
    playing_r4 = [i for i in total_player_r4 if p_r4[i].x >= 0.99]
    benched_r4 = [i for i in all_selected_r4 if i not in playing_r4]
    captained_r4 = [i for i in total_player_r4 if y_r4[i].x >= 0.99]
    selected_r4 = playing_r4

    # Round 5
    all_selected_r5 = [i for i in total_player_r5 if x_r5[i].x >= 0.99]
    playing_r5 = [i for i in total_player_r5 if p_r5[i].x >= 0.99]
    benched_r5 = [i for i in all_selected_r5 if i not in playing_r5]
    captained_r5 = [i for i in total_player_r5 if y_r5[i].x >= 0.99]
    selected_r5 = playing_r5

    # Round 6
    all_selected_r6 = [i for i in total_player_r6 if x_r6[i].x >= 0.99]
    playing_r6 = [i for i in total_player_r6 if p_r6[i].x >= 0.99]
    benched_r6 = [i for i in all_selected_r6 if i not in playing_r6]
    captained_r6 = [i for i in total_player_r6 if y_r6[i].x >= 0.99]
    selected_r6 = playing_r6
  
    # Round 7
    all_selected_r7 = [i for i in total_player_r7 if x_r7[i].x >= 0.99]
    playing_r7 = [i for i in total_player_r7 if p_r7[i].x >= 0.99]
    benched_r7 = [i for i in all_selected_r7 if i not in playing_r7]
    captained_r7 = [i for i in total_player_r7 if y_r7[i].x >= 0.99]
    selected_r7 = playing_r7

    # Round 8
    all_selected_r8 = [i for i in total_player_r8 if x_r8[i].x >= 0.99]
    playing_r8 = [i for i in total_player_r8 if p_r8[i].x >= 0.99]
    benched_r8 = [i for i in all_selected_r8 if i not in playing_r8]
    captained_r8 = [i for i in total_player_r8 if y_r8[i].x >= 0.99]
    selected_r8 = playing_r8

    # Round 9
    all_selected_r9 = [i for i in total_player_r9 if x_r9[i].x >= 0.99]
    playing_r9 = [i for i in total_player_r9 if p_r9[i].x >= 0.99]
    benched_r9 = [i for i in all_selected_r9 if i not in playing_r9]
    captained_r9 = [i for i in total_player_r9 if y_r9[i].x >= 0.99]
    selected_r9 = playing_r9
    # e. Optimisation Results
    # Optimal Team Output
    print("----- Optimal Team Selection Summary -----")
    
    # Round 1
    sel_player_df_r1 = player_df_r1.iloc[selected_r1].copy()
    sel_bench_df_r1 = player_df_r1.iloc[benched_r1].copy() if benched_r1 else pd.DataFrame()
    sel_captain_df_r1 = player_df_r1.iloc[captained_r1]
    # Add bench flag
    sel_player_df_r1['Is_Bench'] = 0
    if not sel_bench_df_r1.empty:
        sel_bench_df_r1['Is_Bench'] = 1
        sel_player_df_r1 = pd.concat([sel_player_df_r1, sel_bench_df_r1], ignore_index=True)
    total_cost_r1 = sum(sel_player_df_r1[sel_player_df_r1['Is_Bench'] == 0]["Price"]) + sum(sel_player_df_r1[sel_player_df_r1['Is_Bench'] == 1]["Price"])
    print("----- Round 1 -----")
    print("Total Expected Points (rnd 1):", sum(sel_player_df_r1[sel_player_df_r1['Is_Bench'] == 0]["exp_rnd_points"]) + sum(sel_captain_df_r1["exp_rnd_points"]))
    print("Total Team Cost (rnd 1):", total_cost_r1)
    print("Captain (rnd 1):", sel_captain_df_r1["Name"].values[0])
    if not sel_bench_df_r1.empty:
        bench_player = sel_player_df_r1[sel_player_df_r1['Is_Bench'] == 1]
        print("Bench Player (rnd 1):", bench_player["Name"].values[0], f"(${bench_player['Price'].values[0]:,})")
    print("Current Players Remaining (rnd 1):", sum(sel_player_df_r1["In_Team"]))

    # Round 2
    sel_player_df_r2 = player_df_r2.iloc[selected_r2].copy()
    sel_bench_df_r2 = player_df_r2.iloc[benched_r2].copy() if benched_r2 else pd.DataFrame()
    sel_captain_df_r2 = player_df_r2.iloc[captained_r2]
    # In Team Flag
    sel_names_r1 = player_df_r1.iloc[all_selected_r1]['Name'].astype(str).tolist() if all_selected_r1 else []
    sel_player_df_r2['In_Team'] = sel_player_df_r2['Name'].astype(str).isin(sel_names_r1).astype(int)
    # Add bench flag
    sel_player_df_r2['Is_Bench'] = 0
    if not sel_bench_df_r2.empty:
        sel_bench_df_r2['In_Team'] = sel_bench_df_r2['Name'].astype(str).isin(sel_names_r1).astype(int)
        sel_bench_df_r2['Is_Bench'] = 1
        sel_player_df_r2 = pd.concat([sel_player_df_r2, sel_bench_df_r2], ignore_index=True)
    total_cost_r2 = sum(sel_player_df_r2[sel_player_df_r2['Is_Bench'] == 0]["Price"]) + sum(sel_player_df_r2[sel_player_df_r2['Is_Bench'] == 1]["Price"])
    print("----- Round 2 -----")
    print("Total Expected Points (rnd 2):", sum(sel_player_df_r2[sel_player_df_r2['Is_Bench'] == 0]["exp_rnd_points"]) + sum(sel_captain_df_r2["exp_rnd_points"]))
    print("Total Team Cost (rnd 2):", total_cost_r2)
    print("Captain (rnd 2):", sel_captain_df_r2["Name"].values[0])
    if not sel_bench_df_r2.empty:
        bench_player = sel_player_df_r2[sel_player_df_r2['Is_Bench'] == 1]
        print("Bench Player (rnd 2):", bench_player["Name"].values[0], f"(${bench_player['Price'].values[0]:,})")
    print("Current Players Remaining (rnd 2):", sum(sel_player_df_r2["In_Team"]))

    # Round 3
    sel_player_df_r3 = player_df_r3.iloc[selected_r3].copy()
    sel_bench_df_r3 = player_df_r3.iloc[benched_r3].copy() if benched_r3 else pd.DataFrame()
    sel_captain_df_r3 = player_df_r3.iloc[captained_r3]
    # In Team Flag
    sel_names_r2 = player_df_r2.iloc[all_selected_r2]['Name'].astype(str).tolist() if all_selected_r2 else []
    sel_player_df_r3['In_Team'] = sel_player_df_r3['Name'].astype(str).isin(sel_names_r2).astype(int)
    # Add bench flag
    sel_player_df_r3['Is_Bench'] = 0
    if not sel_bench_df_r3.empty:
        sel_bench_df_r3['In_Team'] = sel_bench_df_r3['Name'].astype(str).isin(sel_names_r2).astype(int)
        sel_bench_df_r3['Is_Bench'] = 1
        sel_player_df_r3 = pd.concat([sel_player_df_r3, sel_bench_df_r3], ignore_index=True)
    total_cost_r3 = sum(sel_player_df_r3[sel_player_df_r3['Is_Bench'] == 0]["Price"]) + sum(sel_player_df_r3[sel_player_df_r3['Is_Bench'] == 1]["Price"])
    print("----- Round 3 -----")
    print("Total Expected Points (rnd 3):", sum(sel_player_df_r3[sel_player_df_r3['Is_Bench'] == 0]["exp_rnd_points"]) + sum(sel_captain_df_r3["exp_rnd_points"]))
    print("Total Team Cost (rnd 3):", total_cost_r3)
    print("Captain (rnd 3):", sel_captain_df_r3["Name"].values[0])
    if not sel_bench_df_r3.empty:
        bench_player = sel_player_df_r3[sel_player_df_r3['Is_Bench'] == 1]
        print("Bench Player (rnd 3):", bench_player["Name"].values[0], f"(${bench_player['Price'].values[0]:,})")
    print("Current Players Remaining (rnd 3):", sum(sel_player_df_r3["In_Team"]))
    
    # Round 4
    sel_player_df_r4 = player_df_r4.iloc[selected_r4].copy()
    sel_bench_df_r4 = player_df_r4.iloc[benched_r4].copy() if benched_r4 else pd.DataFrame()
    sel_captain_df_r4 = player_df_r4.iloc[captained_r4]
    # In Team Flag
    sel_names_r3 = player_df_r3.iloc[all_selected_r3]['Name'].astype(str).tolist() if all_selected_r3 else []
    sel_player_df_r4['In_Team'] = sel_player_df_r4['Name'].astype(str).isin(sel_names_r3).astype(int)
    # Add bench flag
    sel_player_df_r4['Is_Bench'] = 0
    if not sel_bench_df_r4.empty:
        sel_bench_df_r4['In_Team'] = sel_bench_df_r4['Name'].astype(str).isin(sel_names_r3).astype(int)
        sel_bench_df_r4['Is_Bench'] = 1
        sel_player_df_r4 = pd.concat([sel_player_df_r4, sel_bench_df_r4], ignore_index=True)
    total_cost_r4 = sum(sel_player_df_r4[sel_player_df_r4['Is_Bench'] == 0]["Price"]) + sum(sel_player_df_r4[sel_player_df_r4['Is_Bench'] == 1]["Price"])
    print("----- Round 4 -----")
    print("Total Expected Points (rnd 4):", sum(sel_player_df_r4[sel_player_df_r4['Is_Bench'] == 0]["exp_rnd_points"]) + sum(sel_captain_df_r4["exp_rnd_points"]))
    print("Total Team Cost (rnd 4):", total_cost_r4)
    print("Captain (rnd 4):", sel_captain_df_r4["Name"].values[0])
    if not sel_bench_df_r4.empty:
        bench_player = sel_player_df_r4[sel_player_df_r4['Is_Bench'] == 1]
        print("Bench Player (rnd 4):", bench_player["Name"].values[0], f"(${bench_player['Price'].values[0]:,})")
    print("Current Players Remaining (rnd 4):", sum(sel_player_df_r4["In_Team"]))
    
    # Round 5
    sel_player_df_r5 = player_df_r5.iloc[selected_r5].copy()
    sel_bench_df_r5 = player_df_r5.iloc[benched_r5].copy() if benched_r5 else pd.DataFrame()
    sel_captain_df_r5 = player_df_r5.iloc[captained_r5]
    # In Team Flag
    sel_names_r4 = player_df_r4.iloc[all_selected_r4]['Name'].astype(str).tolist() if all_selected_r4 else []
    sel_player_df_r5['In_Team'] = sel_player_df_r5['Name'].astype(str).isin(sel_names_r4).astype(int)
    # Add bench flag
    sel_player_df_r5['Is_Bench'] = 0
    if not sel_bench_df_r5.empty:
        sel_bench_df_r5['In_Team'] = sel_bench_df_r5['Name'].astype(str).isin(sel_names_r4).astype(int)
        sel_bench_df_r5['Is_Bench'] = 1
        sel_player_df_r5 = pd.concat([sel_player_df_r5, sel_bench_df_r5], ignore_index=True)
    total_cost_r5 = sum(sel_player_df_r5[sel_player_df_r5['Is_Bench'] == 0]["Price"]) + sum(sel_player_df_r5[sel_player_df_r5['Is_Bench'] == 1]["Price"])
    print("----- Round 5 -----")
    print("Total Expected Points (rnd 5):", sum(sel_player_df_r5[sel_player_df_r5['Is_Bench'] == 0]["exp_rnd_points"]) + sum(sel_captain_df_r5["exp_rnd_points"]))
    print("Total Team Cost (rnd 5):", total_cost_r5)
    print("Captain (rnd 5):", sel_captain_df_r5["Name"].values[0])
    if not sel_bench_df_r5.empty:
        bench_player = sel_player_df_r5[sel_player_df_r5['Is_Bench'] == 1]
        print("Bench Player (rnd 5):", bench_player["Name"].values[0], f"(${bench_player['Price'].values[0]:,})")
    print("Current Players Remaining (rnd 5):", sum(sel_player_df_r5["In_Team"]))
    
    # Round 6
    sel_player_df_r6 = player_df_r6.iloc[selected_r6].copy()
    sel_bench_df_r6 = player_df_r6.iloc[benched_r6].copy() if benched_r6 else pd.DataFrame()
    sel_captain_df_r6 = player_df_r6.iloc[captained_r6]
    # In Team Flag
    sel_names_r5 = player_df_r5.iloc[all_selected_r5]['Name'].astype(str).tolist() if all_selected_r5 else []
    sel_player_df_r6['In_Team'] = sel_player_df_r6['Name'].astype(str).isin(sel_names_r5).astype(int)
    # Add bench flag
    sel_player_df_r6['Is_Bench'] = 0
    if not sel_bench_df_r6.empty:
        sel_bench_df_r6['In_Team'] = sel_bench_df_r6['Name'].astype(str).isin(sel_names_r5).astype(int)
        sel_bench_df_r6['Is_Bench'] = 1
        sel_player_df_r6 = pd.concat([sel_player_df_r6, sel_bench_df_r6], ignore_index=True)
    total_cost_r6 = sum(sel_player_df_r6[sel_player_df_r6['Is_Bench'] == 0]["Price"]) + sum(sel_player_df_r6[sel_player_df_r6['Is_Bench'] == 1]["Price"])
    print("----- Round 6 -----")
    print("Total Expected Points (rnd 6):", sum(sel_player_df_r6[sel_player_df_r6['Is_Bench'] == 0]["exp_rnd_points"]) + sum(sel_captain_df_r6["exp_rnd_points"]))
    print("Total Team Cost (rnd 6):", total_cost_r6)
    print("Captain (rnd 6):", sel_captain_df_r6["Name"].values[0])
    if not sel_bench_df_r6.empty:
        bench_player = sel_player_df_r6[sel_player_df_r6['Is_Bench'] == 1]
        print("Bench Player (rnd 6):", bench_player["Name"].values[0], f"(${bench_player['Price'].values[0]:,})")
    print("Current Players Remaining (rnd 6):", sum(sel_player_df_r6["In_Team"]))
    
    # Round 7
    sel_player_df_r7 = player_df_r7.iloc[selected_r7].copy()
    sel_bench_df_r7 = player_df_r7.iloc[benched_r7].copy() if benched_r7 else pd.DataFrame()
    sel_captain_df_r7 = player_df_r7.iloc[captained_r7]
    # In Team Flag
    sel_names_r6 = player_df_r6.iloc[all_selected_r6]['Name'].astype(str).tolist() if all_selected_r6 else []
    sel_player_df_r7['In_Team'] = sel_player_df_r7['Name'].astype(str).isin(sel_names_r6).astype(int)
    # Add bench flag
    sel_player_df_r7['Is_Bench'] = 0
    if not sel_bench_df_r7.empty:
        sel_bench_df_r7['In_Team'] = sel_bench_df_r7['Name'].astype(str).isin(sel_names_r6).astype(int)
        sel_bench_df_r7['Is_Bench'] = 1
        sel_player_df_r7 = pd.concat([sel_player_df_r7, sel_bench_df_r7], ignore_index=True)
    total_cost_r7 = sum(sel_player_df_r7[sel_player_df_r7['Is_Bench'] == 0]["Price"]) + sum(sel_player_df_r7[sel_player_df_r7['Is_Bench'] == 1]["Price"])
    print("----- Round 7 -----")
    print("Total Expected Points (rnd 7):", sum(sel_player_df_r7[sel_player_df_r7['Is_Bench'] == 0]["exp_rnd_points"]) + sum(sel_captain_df_r7["exp_rnd_points"]))
    print("Total Team Cost (rnd 7):", total_cost_r7)
    print("Captain (rnd 7):", sel_captain_df_r7["Name"].values[0])
    if not sel_bench_df_r7.empty:
        bench_player = sel_player_df_r7[sel_player_df_r7['Is_Bench'] == 1]
        print("Bench Player (rnd 7):", bench_player["Name"].values[0], f"(${bench_player['Price'].values[0]:,})")
    print("Current Players Remaining (rnd 7):", sum(sel_player_df_r7["In_Team"]))
    
    # Round 8
    sel_player_df_r8 = player_df_r8.iloc[selected_r8].copy()
    sel_bench_df_r8 = player_df_r8.iloc[benched_r8].copy() if benched_r8 else pd.DataFrame()
    sel_captain_df_r8 = player_df_r8.iloc[captained_r8]
    # In Team Flag
    sel_names_r7 = player_df_r7.iloc[all_selected_r7]['Name'].astype(str).tolist() if all_selected_r7 else []
    sel_player_df_r8['In_Team'] = sel_player_df_r8['Name'].astype(str).isin(sel_names_r7).astype(int)
    # Add bench flag
    sel_player_df_r8['Is_Bench'] = 0
    if not sel_bench_df_r8.empty:
        sel_bench_df_r8['In_Team'] = sel_bench_df_r8['Name'].astype(str).isin(sel_names_r7).astype(int)
        sel_bench_df_r8['Is_Bench'] = 1
        sel_player_df_r8 = pd.concat([sel_player_df_r8, sel_bench_df_r8], ignore_index=True)
    total_cost_r8 = sum(sel_player_df_r8[sel_player_df_r8['Is_Bench'] == 0]["Price"]) + sum(sel_player_df_r8[sel_player_df_r8['Is_Bench'] == 1]["Price"])
    print("----- Round 8 -----")
    print("Total Expected Points (rnd 8):", sum(sel_player_df_r8[sel_player_df_r8['Is_Bench'] == 0]["exp_rnd_points"]) + sum(sel_captain_df_r8["exp_rnd_points"]))
    print("Total Team Cost (rnd 8):", total_cost_r8)
    print("Captain (rnd 8):", sel_captain_df_r8["Name"].values[0])
    if not sel_bench_df_r8.empty:
        bench_player = sel_player_df_r8[sel_player_df_r8['Is_Bench'] == 1]
        print("Bench Player (rnd 8):", bench_player["Name"].values[0], f"(${bench_player['Price'].values[0]:,})")
    print("Current Players Remaining (rnd 8):", sum(sel_player_df_r8["In_Team"]))

    # Round 9  
    sel_player_df_r9 = player_df_r9.iloc[selected_r9].copy()
    sel_bench_df_r9 = player_df_r9.iloc[benched_r9].copy() if benched_r9 else pd.DataFrame()
    sel_captain_df_r9 = player_df_r9.iloc[captained_r9]
    # In Team Flag
    sel_names_r8 = player_df_r8.iloc[all_selected_r8]['Name'].astype(str).tolist() if all_selected_r8 else []
    sel_player_df_r9['In_Team'] = sel_player_df_r9['Name'].astype(str).isin(sel_names_r8).astype(int)
    # Add bench flag
    sel_player_df_r9['Is_Bench'] = 0
    if not sel_bench_df_r9.empty:
        sel_bench_df_r9['In_Team'] = sel_bench_df_r9['Name'].astype(str).isin(sel_names_r8).astype(int)
        sel_bench_df_r9['Is_Bench'] = 1
        sel_player_df_r9 = pd.concat([sel_player_df_r9, sel_bench_df_r9], ignore_index=True)
    total_cost_r9 = sum(sel_player_df_r9[sel_player_df_r9['Is_Bench'] == 0]["Price"]) + sum(sel_player_df_r9[sel_player_df_r9['Is_Bench'] == 1]["Price"])
    print("----- Round 9 -----")
    print("Total Expected Points (rnd 9):", sum(sel_player_df_r9[sel_player_df_r9['Is_Bench'] == 0]["exp_rnd_points"]) + sum(sel_captain_df_r9["exp_rnd_points"]))
    print("Total Team Cost (rnd 9):", total_cost_r9)
    print("Captain (rnd 9):", sel_captain_df_r9["Name"].values[0])
    if not sel_bench_df_r9.empty:
        bench_player = sel_player_df_r9[sel_player_df_r9['Is_Bench'] == 1]
        print("Bench Player (rnd 9):", bench_player["Name"].values[0], f"(${bench_player['Price'].values[0]:,})")
    print("Current Players Remaining (rnd 9):", sum(sel_player_df_r9["In_Team"]))

    # Combine Selected Player DataFrames
    sel_player_df = pd.concat([sel_player_df_r1, sel_player_df_r2, sel_player_df_r3, sel_player_df_r4,
                               sel_player_df_r5, sel_player_df_r6, sel_player_df_r7,sel_player_df_r8, sel_player_df_r9], ignore_index=True)

    print("Total Expected Points:", sum(sel_player_df["exp_rnd_points"]) + 
                                    sum(sel_captain_df_r1["exp_rnd_points"]) + 
                                    sum(sel_captain_df_r2["exp_rnd_points"]) +
                                    sum(sel_captain_df_r3["exp_rnd_points"]) +
                                    sum(sel_captain_df_r4["exp_rnd_points"]) +
                                    sum(sel_captain_df_r5["exp_rnd_points"]) +
                                    sum(sel_captain_df_r6["exp_rnd_points"]) +
                                    sum(sel_captain_df_r7["exp_rnd_points"]) +
                                    sum(sel_captain_df_r8["exp_rnd_points"]) +
                                    sum(sel_captain_df_r9["exp_rnd_points"]))
    
    return sel_player_df, sel_player_df_r1, sel_player_df_r2, sel_player_df_r3, sel_player_df_r4, sel_player_df_r5, sel_player_df_r6, sel_player_df_r7,sel_player_df_r8, sel_player_df_r9

def _run_single_sfp_sim(sim_id, conf_int, lower_z_thresh, upper_z_thresh, current_rnd, player_df_raw, price_df, price_model_obj_1, price_model_obj_2, price_model_obj_3,squad_players, optimise_fn_efp):
    """
    Helper function to run a single simulation.
    Returns: (sim_id, selected_players_df)
    """
    import pandas as pd
    import numpy as np

    # a. Calculate Player Z Score
    player_df_init = player_df_raw.copy()
    player_df_init["z_score"] = np.random.uniform(lower_z_thresh, upper_z_thresh, size=len(player_df_raw))
    player_df_init["sim_points"] = player_df_init["mean"] + (player_df_init["z_score"] * player_df_init["std_dev"])
    player_df_init["sim_points"] = player_df_init["sim_points"].clip(lower=0).round(0)
    player_df_init = player_df_init.rename(columns={"sim_points":"exp_points"})
    player_df_init = player_df_init.drop(columns=["mean", "std_dev", "z_score"])

    player_df_r1, player_df_r2, player_df_r3, player_df_r4, player_df_r5, player_df_r6, player_df_r7, player_df_r8, player_df_r9 = roll_rnd_price_fn(player_df_init, price_df, current_rnd, price_model_obj_1, price_model_obj_2, price_model_obj_3)

    # 2. Run Optimisation
    # a. EFP Optimisation Variables Setup
    # Round 1
    points_r1 = player_df_r1["exp_rnd_points"]
    price_r1 = player_df_r1["Price"]
    weight_r1 = player_df_r1["weight"]
    in_team_r1 = player_df_r1["In_Team"]
    available_r1 = player_df_r1["Available"]
    wk_weight_r1 = player_df_r1["Wk_f"]
    bat_weight_r1 = player_df_r1["Bat_f"]
    bowl_weight_r1 = player_df_r1["Bowl_f"]
    cnt_r1, max_player_r1 = squad_players, range(len(price_r1))
    play_cnt_r1, total_player_r1 = 12, range(len(price_r1))
    wk_cnt_r1, total_wk_r1 = 1, range(len(price_r1))
    bat_cnt_r1, total_bat_r1 = 6, range(len(price_r1))
    bowl_cnt_r1, total_bowl_r1 = 5, range(len(price_r1))
    budget_r1, total_budget_r1 = 1802500, range(len(price_r1))

    # Round 2
    points_r2 = player_df_r2["exp_rnd_points"]
    price_r2 = player_df_r2["Price"]
    weight_r2 = player_df_r2["weight"]
    in_team_r2 = player_df_r2["In_Team"]
    available_r2 = player_df_r2["Available"] 
    wk_weight_r2 = player_df_r2["Wk_f"]
    bat_weight_r2 = player_df_r2["Bat_f"]
    bowl_weight_r2 = player_df_r2["Bowl_f"]
    cnt_r2, max_player_r2 = squad_players, range(len(price_r2))
    play_cnt_r2, total_player_r2 = 12, range(len(price_r2))
    wk_cnt_r2, total_wk_r2 = 1, range(len(price_r2))
    bat_cnt_r2, total_bat_r2 = 6, range(len(price_r2))
    bowl_cnt_r2, total_bowl_r2 = 5, range(len(price_r2))
    budget_r2, total_budget_r2 = 1802500, range(len(price_r2))
    team_play_cnt_r2, total_team_player_r2 = (squad_players - 3), range(len(price_r2)) # At least 10 players from round 1 to be in round 2 team

    # Round 3
    points_r3 = player_df_r3["exp_rnd_points"]
    price_r3 = player_df_r3["Price"]
    weight_r3 = player_df_r3["weight"]
    in_team_r3 = player_df_r3["In_Team"]
    available_r3 = player_df_r3["Available"]
    wk_weight_r3 = player_df_r3["Wk_f"]
    bat_weight_r3 = player_df_r3["Bat_f"]
    bowl_weight_r3 = player_df_r3["Bowl_f"]
    cnt_r3, max_player_r3 = squad_players, range(len(price_r3))
    play_cnt_r3, total_player_r3 = 12, range(len(price_r3))
    wk_cnt_r3, total_wk_r3 = 1, range(len(price_r3))
    bat_cnt_r3, total_bat_r3 = 6, range(len(price_r3))
    bowl_cnt_r3, total_bowl_r3 = 5, range(len(price_r3))
    budget_r3, total_budget_r3 = 1802500, range(len(price_r3))
    team_play_cnt_r3, total_team_player_r3 = (squad_players - 3), range(len(price_r3)) # At least 10 players from round 2 to be in round 3 team

    # Round 4
    points_r4 = player_df_r4["exp_rnd_points"]
    price_r4 = player_df_r4["Price"]
    weight_r4 = player_df_r4["weight"]
    in_team_r4 = player_df_r4["In_Team"]
    available_r4 = player_df_r4["Available"]
    wk_weight_r4 = player_df_r4["Wk_f"]
    bat_weight_r4 = player_df_r4["Bat_f"]
    bowl_weight_r4 = player_df_r4["Bowl_f"]
    cnt_r4, max_player_r4 = squad_players, range(len(price_r4))
    play_cnt_r4, total_player_r4 = 12, range(len(price_r4))
    wk_cnt_r4, total_wk_r4 = 1, range(len(price_r4))
    bat_cnt_r4, total_bat_r4 = 6, range(len(price_r4))
    bowl_cnt_r4, total_bowl_r4 = 5, range(len(price_r4))
    budget_r4, total_budget_r4 = 1802500, range(len(price_r4))
    team_play_cnt_r4, total_team_player_r4 = (squad_players - 3), range(len(price_r4)) # At least 10 players from round 3 to be in round 4 team

    # Round 5
    points_r5 = player_df_r5["exp_rnd_points"]
    price_r5 = player_df_r5["Price"]
    weight_r5 = player_df_r5["weight"]
    in_team_r5 = player_df_r5["In_Team"]
    available_r5 = player_df_r5["Available"]
    wk_weight_r5 = player_df_r5["Wk_f"]
    bat_weight_r5 = player_df_r5["Bat_f"]
    bowl_weight_r5 = player_df_r5["Bowl_f"]
    cnt_r5, max_player_r5 = squad_players, range(len(price_r5))
    play_cnt_r5, total_player_r5 = 12, range(len(price_r5))
    wk_cnt_r5, total_wk_r5 = 1, range(len(price_r5))
    bat_cnt_r5, total_bat_r5 = 6, range(len(price_r5))
    bowl_cnt_r5, total_bowl_r5 = 5, range(len(price_r5))
    budget_r5, total_budget_r5 = 1802500, range(len(price_r5))
    team_play_cnt_r5, total_team_player_r5 = (squad_players - 3), range(len(price_r5)) # At least 10 players from round 4 to be in round 5 team

    # Round 6
    points_r6 = player_df_r6["exp_rnd_points"]
    price_r6 = player_df_r6["Price"]
    weight_r6 = player_df_r6["weight"]
    in_team_r6 = player_df_r6["In_Team"]
    available_r6 = player_df_r6["Available"]
    wk_weight_r6 = player_df_r6["Wk_f"]
    bat_weight_r6 = player_df_r6["Bat_f"]
    bowl_weight_r6 = player_df_r6["Bowl_f"]
    cnt_r6, max_player_r6 = squad_players, range(len(price_r6))
    play_cnt_r6, total_player_r6 = 12, range(len(price_r6))
    wk_cnt_r6, total_wk_r6 = 1, range(len(price_r6))
    bat_cnt_r6, total_bat_r6 = 6, range(len(price_r6))
    bowl_cnt_r6, total_bowl_r6 = 5, range(len(price_r6))
    budget_r6, total_budget_r6 = 1783500, range(len(price_r6))
    team_play_cnt_r6, total_team_player_r6 = (squad_players - 3), range(len(price_r6)) # At least 10 players from round 5 to be in round 6 team

    # Round 7
    points_r7 = player_df_r7["exp_rnd_points"]
    price_r7 = player_df_r7["Price"]
    weight_r7 = player_df_r7["weight"]
    in_team_r7 = player_df_r7["In_Team"]
    available_r7 = player_df_r7["Available"]
    wk_weight_r7 = player_df_r7["Wk_f"]
    bat_weight_r7 = player_df_r7["Bat_f"]
    bowl_weight_r7 = player_df_r7["Bowl_f"]
    cnt_r7, max_player_r7 = squad_players, range(len(price_r7))
    play_cnt_r7, total_player_r7 = 12, range(len(price_r7))
    wk_cnt_r7, total_wk_r7 = 1, range(len(price_r7))
    bat_cnt_r7, total_bat_r7 = 6, range(len(price_r7))
    bowl_cnt_r7, total_bowl_r7 = 5, range(len(price_r7))
    budget_r7, total_budget_r7 = 1802500, range(len(price_r7))
    team_play_cnt_r7, total_team_player_r7 = (squad_players - 3), range(len(price_r7)) # At least 10 players from round 6 to be in round 7 team

    # Round 8
    points_r8 = player_df_r8["exp_rnd_points"]
    price_r8 = player_df_r8["Price"]
    weight_r8 = player_df_r8["weight"]
    in_team_r8 = player_df_r8["In_Team"]
    available_r8 = player_df_r8["Available"]
    wk_weight_r8 = player_df_r8["Wk_f"]
    bat_weight_r8 = player_df_r8["Bat_f"]
    bowl_weight_r8 = player_df_r8["Bowl_f"]
    cnt_r8, max_player_r8 = squad_players, range(len(price_r8))
    play_cnt_r8, total_player_r8 = 12, range(len(price_r8))
    wk_cnt_r8, total_wk_r8 = 1, range(len(price_r8))
    bat_cnt_r8, total_bat_r8 = 6, range(len(price_r8))
    bowl_cnt_r8, total_bowl_r8 = 5, range(len(price_r8))
    budget_r8, total_budget_r8 = 1802500, range(len(price_r8))
    team_play_cnt_r8, total_team_player_r8 = (squad_players - 3), range(len(price_r8)) # At least 10 players from round 7 to be in round 8 team

    # Round 9
    points_r9 = player_df_r9["exp_rnd_points"]
    price_r9 = player_df_r9["Price"]
    weight_r9 = player_df_r9["weight"]
    in_team_r9 = player_df_r9["In_Team"]
    available_r9 = player_df_r9["Available"]
    wk_weight_r9 = player_df_r9["Wk_f"]
    bat_weight_r9 = player_df_r9["Bat_f"]
    bowl_weight_r9 = player_df_r9["Bowl_f"]
    cnt_r9, max_player_r9 = squad_players, range(len(price_r9))
    play_cnt_r9, total_player_r9 = 12, range(len(price_r9))
    wk_cnt_r9, total_wk_r9 = 1, range(len(price_r9))
    bat_cnt_r9, total_bat_r9 = 6, range(len(price_r9))
    bowl_cnt_r9, total_bowl_r9 = 5, range(len(price_r9))
    budget_r9, total_budget_r9 = 1802500, range(len(price_r9))
    team_play_cnt_r9, total_team_player_r9 = (squad_players - 3), range(len(price_r9)) # At least 10 players from round 8 to be in round 9 team

    # b. Run optimization
    sel_player_df, sel_player_df_r1, sel_player_df_r2, sel_player_df_r3, sel_player_df_r4, sel_player_df_r5, sel_player_df_r6, sel_player_df_r7,sel_player_df_r8, sel_player_df_r9 = optimise_fn_efp(
        # Round 1
        points_r1, price_r1, weight_r1, in_team_r1, available_r1, wk_weight_r1, bat_weight_r1, bowl_weight_r1,
        play_cnt_r1, total_player_r1, wk_cnt_r1, total_wk_r1, bat_cnt_r1, total_bat_r1, bowl_cnt_r1, total_bowl_r1,
        budget_r1, total_budget_r1, player_df_r1, cnt_r1, max_player_r1,
        # Round 2
        points_r2, price_r2, weight_r2, in_team_r2, available_r2, wk_weight_r2, bat_weight_r2, bowl_weight_r2,
        play_cnt_r2, total_player_r2, wk_cnt_r2, total_wk_r2, bat_cnt_r2, total_bat_r2, bowl_cnt_r2, total_bowl_r2,
        budget_r2, total_budget_r2, team_play_cnt_r2, total_team_player_r2, player_df_r2, cnt_r2, max_player_r2,
        # Round 3
        points_r3, price_r3, weight_r3, in_team_r3, available_r3, wk_weight_r3, bat_weight_r3, bowl_weight_r3,
        play_cnt_r3, total_player_r3, wk_cnt_r3, total_wk_r3, bat_cnt_r3, total_bat_r3, bowl_cnt_r3, total_bowl_r3,
        budget_r3, total_budget_r3, team_play_cnt_r3, total_team_player_r3, player_df_r3, cnt_r3, max_player_r3,
        # Round 4
        points_r4, price_r4, weight_r4, in_team_r4, available_r4, wk_weight_r4, bat_weight_r4, bowl_weight_r4,
        play_cnt_r4, total_player_r4, wk_cnt_r4, total_wk_r4, bat_cnt_r4, total_bat_r4, bowl_cnt_r4, total_bowl_r4,
        budget_r4, total_budget_r4, team_play_cnt_r4, total_team_player_r4, player_df_r4, cnt_r4, max_player_r4,
        # Round 5
        points_r5, price_r5, weight_r5, in_team_r5, available_r5, wk_weight_r5, bat_weight_r5, bowl_weight_r5,
        play_cnt_r5, total_player_r5, wk_cnt_r5, total_wk_r5, bat_cnt_r5, total_bat_r5, bowl_cnt_r5, total_bowl_r5,
        budget_r5, total_budget_r5, team_play_cnt_r5, total_team_player_r5, player_df_r5, cnt_r5, max_player_r5,
        # Round 6
        points_r6, price_r6, weight_r6, in_team_r6, available_r6, wk_weight_r6, bat_weight_r6, bowl_weight_r6,
        play_cnt_r6, total_player_r6, wk_cnt_r6, total_wk_r6, bat_cnt_r6, total_bat_r6, bowl_cnt_r6, total_bowl_r6,
        budget_r6, total_budget_r6, team_play_cnt_r6, total_team_player_r6, player_df_r6, cnt_r6, max_player_r6,
        # Round 7
        points_r7, price_r7, weight_r7, in_team_r7, available_r7, wk_weight_r7, bat_weight_r7, bowl_weight_r7,
        play_cnt_r7, total_player_r7, wk_cnt_r7, total_wk_r7, bat_cnt_r7, total_bat_r7, bowl_cnt_r7, total_bowl_r7,
        budget_r7, total_budget_r7, team_play_cnt_r7, total_team_player_r7, player_df_r7, cnt_r7, max_player_r7,
        # Round 8
        points_r8, price_r8, weight_r8, in_team_r8, available_r8, wk_weight_r8, bat_weight_r8, bowl_weight_r8,
        play_cnt_r8, total_player_r8, wk_cnt_r8, total_wk_r8, bat_cnt_r8, total_bat_r8, bowl_cnt_r8, total_bowl_r8,
        budget_r8, total_budget_r8, team_play_cnt_r8, total_team_player_r8, player_df_r8, cnt_r8, max_player_r8,
        # Round 9
        points_r9, price_r9, weight_r9, in_team_r9, available_r9, wk_weight_r9, bat_weight_r9, bowl_weight_r9,
        play_cnt_r9, total_player_r9, wk_cnt_r9, total_wk_r9, bat_cnt_r9, total_bat_r9, bowl_cnt_r9, total_bowl_r9,
        budget_r9, total_budget_r9, team_play_cnt_r9, total_team_player_r9, player_df_r9, cnt_r9, max_player_r9
    )

    # 3. Store Simulation Results
    # Only store traded in players for storage efficiency
    sim_sel_players = sel_player_df_r1[['Name']][sel_player_df_r1['In_Team'] == 0]
    sim_sel_players['Simulation'] = sim_id + 1

    # Drop dataframes to free up memory
    del player_df_init, sel_player_df_r1, sel_player_df_r2, sel_player_df_r3, sel_player_df_r4, sel_player_df_r5
    del sel_player_df_r6, sel_player_df_r7, sel_player_df_r8, sel_player_df_r9, sel_player_df
    del price_r1, price_r2, price_r3, price_r4, price_r5, price_r6, price_r7, price_r8, price_r9
    del weight_r1, weight_r2, weight_r3, weight_r4, weight_r5, weight_r6, weight_r7, weight_r8, weight_r9
    del in_team_r1, in_team_r2, in_team_r3, in_team_r4, in_team_r5, in_team_r6, in_team_r7, in_team_r8, in_team_r9
    del available_r1, available_r2, available_r3, available_r4, available_r5, available_r6, available_r7, available_r8, available_r9
    del wk_weight_r1, wk_weight_r2, wk_weight_r3, wk_weight_r4, wk_weight_r5, wk_weight_r6, wk_weight_r7, wk_weight_r8, wk_weight_r9
    del bat_weight_r1, bat_weight_r2, bat_weight_r3, bat_weight_r4, bat_weight_r5, bat_weight_r6, bat_weight_r7, bat_weight_r8, bat_weight_r9
    del bowl_weight_r1, bowl_weight_r2, bowl_weight_r3, bowl_weight_r4, bowl_weight_r5, bowl_weight_r6, bowl_weight_r7, bowl_weight_r8, bowl_weight_r9
    del play_cnt_r1, play_cnt_r2, play_cnt_r3, play_cnt_r4, play_cnt_r5, play_cnt_r6, play_cnt_r7, play_cnt_r8, play_cnt_r9
    del budget_r1, budget_r2, budget_r3, budget_r4, budget_r5, budget_r6, budget_r7, budget_r8, budget_r9
    del total_player_r1, total_player_r2, total_player_r3, total_player_r4, total_player_r5, total_player_r6, total_player_r7, total_player_r8, total_player_r9
    del total_budget_r1, total_budget_r2, total_budget_r3, total_budget_r4, total_budget_r5, total_budget_r6, total_budget_r7, total_budget_r8, total_budget_r9
    del team_play_cnt_r2, team_play_cnt_r3, team_play_cnt_r4, team_play_cnt_r5, team_play_cnt_r6, team_play_cnt_r7, team_play_cnt_r8, team_play_cnt_r9
    del total_team_player_r2, total_team_player_r3, total_team_player_r4, total_team_player_r5, total_team_player_r6, total_team_player_r7, total_team_player_r8, total_team_player_r9
    del points_r1, points_r2, points_r3, points_r4, points_r5, points_r6, points_r7, points_r8, points_r9
    del player_df_r1, player_df_r2, player_df_r3, player_df_r4, player_df_r5, player_df_r6, player_df_r7, player_df_r8, player_df_r9

    return sim_sel_players


def optimise_fn_sim_fp(conf_int, sim_num, current_rnd, player_df_raw, price_df, price_model_obj_1, price_model_obj_2, price_model_obj_3, squad_players, use_parallel=True):
    # Run Optimisation Process for Specified Number of Simulations
    # Import Packages
    from scipy.stats import norm
    import pandas as pd
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor

    # Simulation Setup
    def z_score_bounds(confidence_level):
        """
        Returns the lower and upper z-scores for a given confidence level.
        For example, confidence_level = 0.90 for 90%.
        """
        alpha = 1 - confidence_level
        lower = norm.ppf(alpha / 2)
        upper = norm.ppf(1 - alpha / 2)
        return lower, upper

    lower_z_thresh, upper_z_thresh = z_score_bounds(conf_int)
    print(f"For {conf_int*100:.0f}% simulated points confidence interval the lower z score is {lower_z_thresh:.3f} and upper z score is {upper_z_thresh:.3f}")
    
    # Prepare DataFrame for Simulation Outputs
    all_sim_sel_players = pd.DataFrame()
    all_sim_sel_players["Name"] = []
    all_sim_sel_players["Simulation"] = []

    if use_parallel:
        # Run simulations in parallel
        max_workers = min(10, sim_num)  # Reduced from 15 to avoid memory issues
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _run_single_sfp_sim,
                    i, conf_int, lower_z_thresh, upper_z_thresh, current_rnd, 
                    player_df_raw, price_df, price_model_obj_1, price_model_obj_2, 
                    price_model_obj_3,squad_players, optimise_fn_efp
                )
                for i in range(sim_num)
            ]
            
            # Collect results as they complete with error handling
            for i, future in enumerate(futures):
                try:
                    sim_result = future.result()
                    all_sim_sel_players = pd.concat([all_sim_sel_players, sim_result], ignore_index=True)
                    # Print progress every 10 simulations
                    if (i + 1) % 10 == 0:
                        print(f"Completed {i + 1}/{sim_num} simulations")
                except Exception as e:
                    print(f"Simulation {i+1} failed with error: {e}")
                    import traceback
                    traceback.print_exc()
    
    else:
        # Run simulations sequentially (original method)
        for i in range(sim_num):
            sim_result = _run_single_sfp_sim(
                i, conf_int, lower_z_thresh, upper_z_thresh, current_rnd,
                player_df_raw, price_df, price_model_obj_1, price_model_obj_2,
                price_model_obj_3,squad_players, optimise_fn_efp
            )
            all_sim_sel_players = pd.concat([all_sim_sel_players, sim_result], ignore_index=True)
            # Print progress every 10 simulations
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{sim_num} simulations")
    
    # Return all simulation selected players
    return all_sim_sel_players