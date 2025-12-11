# Optimisation Functions
# Imports
import pandas as pd
import numpy as np
import os
import random

# EFP Optimisation Function
def optimise_fn_efp(
        # Round 1
        points_r1, price_r1, weight_r1, in_team_r1, available_r1, wk_weight_r1, bat_weight_r1, bowl_weight_r1,
        play_cnt_r1, total_player_r1, wk_cnt_r1, total_wk_r1, bat_cnt_r1, total_bat_r1, bowl_cnt_r1, total_bowl_r1,
        budget_r1, total_budget_r1, player_df_r1,
        # Round 2
        points_r2, price_r2, weight_r2, in_team_r2, available_r2, wk_weight_r2, bat_weight_r2, bowl_weight_r2,
        play_cnt_r2, total_player_r2, wk_cnt_r2, total_wk_r2, bat_cnt_r2, total_bat_r2, bowl_cnt_r2, total_bowl_r2,
        budget_r2, total_budget_r2, team_play_cnt_r2, total_team_player_r2, player_df_r2,
        # Round 3
        points_r3, price_r3, weight_r3, in_team_r3, available_r3, wk_weight_r3, bat_weight_r3, bowl_weight_r3,
        play_cnt_r3, total_player_r3, wk_cnt_r3, total_wk_r3, bat_cnt_r3, total_bat_r3, bowl_cnt_r3, total_bowl_r3,
        budget_r3, total_budget_r3, team_play_cnt_r3, total_team_player_r3, player_df_r3,
        # Round 4
        points_r4, price_r4, weight_r4, in_team_r4, available_r4, wk_weight_r4, bat_weight_r4, bowl_weight_r4,
        play_cnt_r4, total_player_r4, wk_cnt_r4, total_wk_r4, bat_cnt_r4, total_bat_r4, bowl_cnt_r4, total_bowl_r4,
        budget_r4, total_budget_r4, team_play_cnt_r4, total_team_player_r4, player_df_r4,
        # Round 5
        points_r5, price_r5, weight_r5, in_team_r5, available_r5, wk_weight_r5, bat_weight_r5, bowl_weight_r5,
        play_cnt_r5, total_player_r5, wk_cnt_r5, total_wk_r5, bat_cnt_r5, total_bat_r5, bowl_cnt_r5, total_bowl_r5,
        budget_r5, total_budget_r5, team_play_cnt_r5, total_team_player_r5, player_df_r5,
        # Round 6
        points_r6, price_r6, weight_r6, in_team_r6, available_r6, wk_weight_r6, bat_weight_r6, bowl_weight_r6,
        play_cnt_r6, total_player_r6, wk_cnt_r6, total_wk_r6, bat_cnt_r6, total_bat_r6, bowl_cnt_r6, total_bowl_r6,
        budget_r6, total_budget_r6, team_play_cnt_r6, total_team_player_r6, player_df_r6,
        # Round 7
        points_r7, price_r7, weight_r7, in_team_r7, available_r7, wk_weight_r7, bat_weight_r7, bowl_weight_r7,
        play_cnt_r7, total_player_r7, wk_cnt_r7, total_wk_r7, bat_cnt_r7, total_bat_r7, bowl_cnt_r7, total_bowl_r7,
        budget_r7, total_budget_r7, team_play_cnt_r7, total_team_player_r7, player_df_r7,
        # Round 8
        points_r8, price_r8, weight_r8, in_team_r8, available_r8, wk_weight_r8, bat_weight_r8, bowl_weight_r8,
        play_cnt_r8, total_player_r8, wk_cnt_r8, total_wk_r8, bat_cnt_r8, total_bat_r8, bowl_cnt_r8, total_bowl_r8,
        budget_r8, total_budget_r8, team_play_cnt_r8, total_team_player_r8, player_df_r8,
        # Round 9
        points_r9, price_r9, weight_r9, in_team_r9, available_r9, wk_weight_r9, bat_weight_r9, bowl_weight_r9,
        play_cnt_r9, total_player_r9, wk_cnt_r9, total_wk_r9, bat_cnt_r9, total_bat_r9, bowl_cnt_r9, total_bowl_r9,
        budget_r9, total_budget_r9, team_play_cnt_r9, total_team_player_r9, player_df_r9):
    
    # Imports
    from mip import Model, xsum, maximize, BINARY 

    # a. initialize optimisation parameters
    m = Model("knapsack")
    # x is player selected variable, y is captain selected variable (2x points)
    # Round 1
    x_r1 = [m.add_var(var_type=BINARY) for i in total_player_r1]
    y_r1 = [m.add_var(var_type=BINARY) for i in total_player_r1]
    # Round 2
    x_r2 = [m.add_var(var_type=BINARY) for i in total_player_r2]
    y_r2 = [m.add_var(var_type=BINARY) for i in total_player_r2]
    # Round 3
    x_r3 = [m.add_var(var_type=BINARY) for i in total_player_r3]
    y_r3 = [m.add_var(var_type=BINARY) for i in total_player_r3]
    # Round 4
    x_r4 = [m.add_var(var_type=BINARY) for i in total_player_r4]
    y_r4 = [m.add_var(var_type=BINARY) for i in total_player_r4]
    # Round 5
    x_r5 = [m.add_var(var_type=BINARY) for i in total_player_r5]
    y_r5 = [m.add_var(var_type=BINARY) for i in total_player_r5]
    # Round 6
    x_r6 = [m.add_var(var_type=BINARY) for i in total_player_r6]
    y_r6 = [m.add_var(var_type=BINARY) for i in total_player_r6]
    # Round 7
    x_r7 = [m.add_var(var_type=BINARY) for i in total_player_r7]
    y_r7 = [m.add_var(var_type=BINARY) for i in total_player_r7]
    # Round 8
    x_r8 = [m.add_var(var_type=BINARY) for i in total_player_r8]
    y_r8 = [m.add_var(var_type=BINARY) for i in total_player_r8]
    # Round 9
    x_r9 = [m.add_var(var_type=BINARY) for i in total_player_r9]
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

            # Trade Boost Constraint (4 Trades Allowed for 2 Rounds (Only 8 In Team Players Required))
            extra_trade = m.add_var(var_type=BINARY)
            extra_trade_vars.append(extra_trade)

            # If Trade Boost is used, allow 8 players from previous round, else allow 9
            m += xsum(z_rnd[(i, j)] for (i, j) in z_rnd.keys()) >= team_min - extra_trade

    # Only Two Trade Boosts Allowed in season
    m += xsum(extra_trade_vars for extra_trade_vars in extra_trade_vars) <= 2 

    # b. define objective function
    obj_r1 = xsum(points_r1[i]*x_r1[i] + points_r1[i]*y_r1[i] for i in total_player_r1)
    obj_r2 = xsum(points_r2[i]*x_r2[i] + points_r2[i]*y_r2[i] for i in total_player_r2)
    obj_r3 = xsum(points_r3[i]*x_r3[i] + points_r3[i]*y_r3[i] for i in total_player_r3)
    obj_r4 = xsum(points_r4[i]*x_r4[i] + points_r4[i]*y_r4[i] for i in total_player_r4)
    obj_r5 = xsum(points_r5[i]*x_r5[i] + points_r5[i]*y_r5[i] for i in total_player_r5)
    obj_r6 = xsum(points_r6[i]*x_r6[i] + points_r6[i]*y_r6[i] for i in total_player_r6)
    obj_r7 = xsum(points_r7[i]*x_r7[i] + points_r7[i]*y_r7[i] for i in total_player_r7)
    obj_r8 = xsum(points_r8[i]*x_r8[i] + points_r8[i]*y_r8[i] for i in total_player_r8)
    obj_r9 = xsum(points_r9[i]*x_r9[i] + points_r9[i]*y_r9[i] for i in total_player_r9)

    m.objective = maximize(obj_r1 + obj_r2 + obj_r3 + obj_r4 + obj_r5 + obj_r6 + obj_r7 + obj_r8 + obj_r9)
    
    # c. define constraints
        # Player selection constraints
    # Round 1
    m += xsum(weight_r1[i] * x_r1[i] for i in total_player_r1) == play_cnt_r1
    m += xsum(y_r1[i] for i in total_player_r1) == 1  # Only one captain
    # Round 2
    m += xsum(weight_r2[i] * x_r2[i] for i in total_player_r2) == play_cnt_r2
    m += xsum(y_r2[i] for i in total_player_r2) == 1  # Only one captain
    # Round 3
    m += xsum(weight_r3[i] * x_r3[i] for i in total_player_r3) == play_cnt_r3
    m += xsum(y_r3[i] for i in total_player_r3) == 1  # Only one captain
    # Round 4
    m += xsum(weight_r4[i] * x_r4[i] for i in total_player_r4) == play_cnt_r4
    m += xsum(y_r4[i] for i in total_player_r4) == 1  # Only one captain
    # Round 5
    m += xsum(weight_r5[i] * x_r5[i] for i in total_player_r5) == play_cnt_r5
    m += xsum(y_r5[i] for i in total_player_r5) == 1  # Only one captain
    # Round 6
    m += xsum(weight_r6[i] * x_r6[i] for i in total_player_r6) == play_cnt_r6
    m += xsum(y_r6[i] for i in total_player_r6) == 1  # Only one captain
    # Round 7
    m += xsum(weight_r7[i] * x_r7[i] for i in total_player_r7) == play_cnt_r7
    m += xsum(y_r7[i] for i in total_player_r7) == 1  # Only one captain
    # Round 8
    m += xsum(weight_r8[i] * x_r8[i] for i in total_player_r8) == play_cnt_r8
    m += xsum(y_r8[i] for i in total_player_r8) == 1  # Only one captain
    # Round 9
    m += xsum(weight_r9[i] * x_r9[i] for i in total_player_r9) == play_cnt_r9
    m += xsum(y_r9[i] for i in total_player_r9) == 1  # Only one captain

    # captain can only be chosen among selected players
    for i in total_player_r1:
        m += y_r1[i] <= x_r1[i]
    for i in total_player_r2:
        m += y_r2[i] <= x_r2[i]
    for i in total_player_r3:
        m += y_r3[i] <= x_r3[i]
    for i in total_player_r4:
        m += y_r4[i] <= x_r4[i]
    for i in total_player_r5:
        m += y_r5[i] <= x_r5[i]
    for i in total_player_r6:
        m += y_r6[i] <= x_r6[i]
    for i in total_player_r7:
        m += y_r7[i] <= x_r7[i]
    for i in total_player_r8:
        m += y_r8[i] <= x_r8[i]
    for i in total_player_r9:
        m += y_r9[i] <= x_r9[i]

        # Team composition constraints
    # Round 1  
    m += xsum(wk_weight_r1[i] * x_r1[i] for i in total_wk_r1) >= wk_cnt_r1
    m += xsum(bat_weight_r1[i] * x_r1[i] for i in total_bat_r1) >= bat_cnt_r1
    m += xsum(bowl_weight_r1[i] * x_r1[i] for i in total_bowl_r1) >= bowl_cnt_r1
    m += xsum(available_r1[i] * x_r1[i] for i in total_player_r1) == play_cnt_r1
    m += xsum(price_r1[i] * x_r1[i] for i in total_budget_r1) <= budget_r1
    # Round 2  
    m += xsum(wk_weight_r2[i] * x_r2[i] for i in total_wk_r2) >= wk_cnt_r2
    m += xsum(bat_weight_r2[i] * x_r2[i] for i in total_bat_r2) >= bat_cnt_r2
    m += xsum(bowl_weight_r2[i] * x_r2[i] for i in total_bowl_r2) >= bowl_cnt_r2
    m += xsum(available_r2[i] * x_r2[i] for i in total_player_r2) == play_cnt_r2
    m += xsum(price_r2[i] * x_r2[i] for i in total_budget_r2) <= (
        budget_r1 + xsum((price_r2[i] - price_r1[i]) * x_r1[i] for i in total_player_r1)
    )
     # Round 3
    m += xsum(wk_weight_r3[i] * x_r3[i] for i in total_wk_r3) >= wk_cnt_r3
    m += xsum(bat_weight_r3[i] * x_r3[i] for i in total_bat_r3) >= bat_cnt_r3
    m += xsum(bowl_weight_r3[i] * x_r3[i] for i in total_bowl_r3) >= bowl_cnt_r3
    m += xsum(available_r3[i] * x_r3[i] for i in total_player_r3) == play_cnt_r3
    m += xsum(price_r3[i] * x_r3[i] for i in total_budget_r3) <= (
        budget_r1
        + xsum((price_r2[i] - price_r1[i]) * x_r1[i] for i in total_player_r1)
        + xsum((price_r3[i] - price_r2[i]) * x_r2[i] for i in total_player_r2)
    )
    # Round 4
    m += xsum(wk_weight_r4[i] * x_r4[i] for i in total_wk_r4) >= wk_cnt_r4
    m += xsum(bat_weight_r4[i] * x_r4[i] for i in total_bat_r4) >= bat_cnt_r4
    m += xsum(bowl_weight_r4[i] * x_r4[i] for i in total_bowl_r4) >= bowl_cnt_r4
    m += xsum(available_r4[i] * x_r4[i] for i in total_player_r4) == play_cnt_r4
    m += xsum(price_r4[i] * x_r4[i] for i in total_budget_r4) <= (
        budget_r1
        + xsum((price_r2[i] - price_r1[i]) * x_r1[i] for i in total_player_r1)
        + xsum((price_r3[i] - price_r2[i]) * x_r2[i] for i in total_player_r2)
        + xsum((price_r4[i] - price_r3[i]) * x_r3[i] for i in total_player_r3)
    )
    # Round 5
    m += xsum(wk_weight_r5[i] * x_r5[i] for i in total_wk_r5) >= wk_cnt_r5
    m += xsum(bat_weight_r5[i] * x_r5[i] for i in total_bat_r5) >= bat_cnt_r5
    m += xsum(bowl_weight_r5[i] * x_r5[i] for i in total_bowl_r5) >= bowl_cnt_r5
    m += xsum(available_r5[i] * x_r5[i] for i in total_player_r5) == play_cnt_r5
    m += xsum(price_r5[i] * x_r5[i] for i in total_budget_r5) <= (
        budget_r1
        + xsum((price_r2[i] - price_r1[i]) * x_r1[i] for i in total_player_r1)
        + xsum((price_r3[i] - price_r2[i]) * x_r2[i] for i in total_player_r2)
        + xsum((price_r4[i] - price_r3[i]) * x_r3[i] for i in total_player_r3)
        + xsum((price_r5[i] - price_r4[i]) * x_r4[i] for i in total_player_r4)
    )
    # Round 6
    m += xsum(wk_weight_r6[i] * x_r6[i] for i in total_wk_r6) >= wk_cnt_r6
    m += xsum(bat_weight_r6[i] * x_r6[i] for i in total_bat_r6) >= bat_cnt_r6
    m += xsum(bowl_weight_r6[i] * x_r6[i] for i in total_bowl_r6) >= bowl_cnt_r6
    m += xsum(available_r6[i] * x_r6[i] for i in total_player_r6) == play_cnt_r6
    m += xsum(price_r6[i] * x_r6[i] for i in total_budget_r6) <= (
        budget_r1
        + xsum((price_r2[i] - price_r1[i]) * x_r1[i] for i in total_player_r1)
        + xsum((price_r3[i] - price_r2[i]) * x_r2[i] for i in total_player_r2)
        + xsum((price_r4[i] - price_r3[i]) * x_r3[i] for i in total_player_r3)
        + xsum((price_r5[i] - price_r4[i]) * x_r4[i] for i in total_player_r4)
        + xsum((price_r6[i] - price_r5[i]) * x_r5[i] for i in total_player_r5)
    )
    # Round 7
    m += xsum(wk_weight_r7[i] * x_r7[i] for i in total_wk_r7) >= wk_cnt_r7
    m += xsum(bat_weight_r7[i] * x_r7[i] for i in total_bat_r7) >= bat_cnt_r7
    m += xsum(bowl_weight_r7[i] * x_r7[i] for i in total_bowl_r7) >= bowl_cnt_r7
    m += xsum(available_r7[i] * x_r7[i] for i in total_player_r7) == play_cnt_r7
    m += xsum(price_r7[i] * x_r7[i] for i in total_budget_r7) <= (
        budget_r1
        + xsum((price_r2[i] - price_r1[i]) * x_r1[i] for i in total_player_r1)
        + xsum((price_r3[i] - price_r2[i]) * x_r2[i] for i in total_player_r2)
        + xsum((price_r4[i] - price_r3[i]) * x_r3[i] for i in total_player_r3)
        + xsum((price_r5[i] - price_r4[i]) * x_r4[i] for i in total_player_r4)
        + xsum((price_r6[i] - price_r5[i]) * x_r5[i] for i in total_player_r5)
        + xsum((price_r7[i] - price_r6[i]) * x_r6[i] for i in total_player_r6)
    )
    # Round 8
    m += xsum(wk_weight_r8[i] * x_r8[i] for i in total_wk_r8) >= wk_cnt_r8
    m += xsum(bat_weight_r8[i] * x_r8[i] for i in total_bat_r8) >= bat_cnt_r8
    m += xsum(bowl_weight_r8[i] * x_r8[i] for i in total_bowl_r8) >= bowl_cnt_r8
    m += xsum(available_r8[i] * x_r8[i] for i in total_player_r8) == play_cnt_r8
    m += xsum(price_r8[i] * x_r8[i] for i in total_budget_r8) <= (
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
    m += xsum(wk_weight_r9[i] * x_r9[i] for i in total_wk_r9) >= wk_cnt_r9
    m += xsum(bat_weight_r9[i] * x_r9[i] for i in total_bat_r9) >= bat_cnt_r9
    m += xsum(bowl_weight_r9[i] * x_r9[i] for i in total_bowl_r9) >= bowl_cnt_r9
    m += xsum(available_r9[i] * x_r9[i] for i in total_player_r9) == play_cnt_r9
    m += xsum(price_r9[i] * x_r9[i] for i in total_budget_r9) <= (
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
    m.optimize()
    # Round 1 
    selected_r1 = [i for i in total_player_r1 if x_r1[i].x >= 0.99]
    captained_r1 = [i for i in total_player_r1 if y_r1[i].x >= 0.99]

    # Round 2
    selected_r2 = [i for i in total_player_r2 if x_r2[i].x >= 0.99]
    captained_r2 = [i for i in total_player_r2 if y_r2[i].x >= 0.99]

    # Round 3
    selected_r3 = [i for i in total_player_r3 if x_r3[i].x >= 0.99]
    captained_r3 = [i for i in total_player_r3 if y_r3[i].x >= 0.99]

    # Round 4
    selected_r4 = [i for i in total_player_r4 if x_r4[i].x >= 0.99]
    captained_r4 = [i for i in total_player_r4 if y_r4[i].x >= 0.99]

    # Round 5
    selected_r5 = [i for i in total_player_r5 if x_r5[i].x >= 0.99]
    captained_r5 = [i for i in total_player_r5 if y_r5[i].x >= 0.99]

    # Round 6
    selected_r6 = [i for i in total_player_r6 if x_r6[i].x >= 0.99]
    captained_r6 = [i for i in total_player_r6 if y_r6[i].x >= 0.99]
  
    # Round 7
    selected_r7 = [i for i in total_player_r7 if x_r7[i].x >= 0.99]
    captained_r7 = [i for i in total_player_r7 if y_r7[i].x >= 0.99]

    # Round 8
    selected_r8 = [i for i in total_player_r8 if x_r8[i].x >= 0.99]
    captained_r8 = [i for i in total_player_r8 if y_r8[i].x >= 0.99]
 
    # Round 9
    selected_r9 = [i for i in total_player_r9 if x_r9[i].x >= 0.99]
    captained_r9 = [i for i in total_player_r9 if y_r9[i].x >= 0.99]

    # e. Optimisation Results
    # print("Selected items (rnd 1): {}".format(selected_r1))
    # print("Captain selected (rnd 1): {}".format(captained_r1))

    # print("Selected items (rnd 2): {}".format(selected_r2))
    # print("Captain selected (rnd 2): {}".format(captained_r2))

    # Optimal Team Output
    print("----- Optimal Team Selection Summary -----")
    
    # Round 1
    sel_player_df_r1 = player_df_r1.iloc[selected_r1]
    sel_captain_df_r1 = player_df_r1.iloc[captained_r1]
    print("----- Round 1 -----")
    print("Total Expected Points (rnd 1):", sum(sel_player_df_r1["exp_rnd_points"]) + sum(sel_captain_df_r1["exp_rnd_points"]))
    print("Total Team Cost (rnd 1):", sum(sel_player_df_r1["Price"]))
    print("Captain (rnd 1):", sel_captain_df_r1["Name"].values[0])
    print("Current Players Remaining (rnd 1):", sum(sel_player_df_r1["In_Team"]))

    # Round 2
    sel_player_df_r2 = player_df_r2.iloc[selected_r2]
    sel_captain_df_r2 = player_df_r2.iloc[captained_r2]
    # In Team Flag
    sel_names_r1 = player_df_r1.iloc[selected_r1]['Name'].astype(str).tolist() if selected_r1 else []
    sel_player_df_r2['In_Team'] = sel_player_df_r2['Name'].astype(str).isin(sel_names_r1).astype(int)
    print("----- Round 2 -----")
    print("Total Expected Points (rnd 2):", sum(sel_player_df_r2["exp_rnd_points"]) + sum(sel_captain_df_r2["exp_rnd_points"]))
    print("Total Team Cost (rnd 2):", sum(sel_player_df_r2["Price"]))
    print("Captain (rnd 2):", sel_captain_df_r2["Name"].values[0])
    print("Current Players Remaining (rnd 2):", sum(sel_player_df_r2["In_Team"]))

    # Round 3
    sel_player_df_r3 = player_df_r3.iloc[selected_r3]
    sel_captain_df_r3 = player_df_r3.iloc[captained_r3]
    # In Team Flag
    sel_names_r2 = player_df_r2.iloc[selected_r2]['Name'].astype(str).tolist() if selected_r2 else []
    sel_player_df_r3['In_Team'] = sel_player_df_r3['Name'].astype(str).isin(sel_names_r2).astype(int)
    print("----- Round 3 -----")
    print("Total Expected Points (rnd 3):", sum(sel_player_df_r3["exp_rnd_points"]) + sum(sel_captain_df_r3["exp_rnd_points"]))
    print("Total Team Cost (rnd 3):", sum(sel_player_df_r3["Price"]))
    print("Captain (rnd 3):", sel_captain_df_r3["Name"].values[0])
    print("Current Players Remaining (rnd 3):", sum(sel_player_df_r3["In_Team"]))
    
    # Round 4
    sel_player_df_r4 = player_df_r4.iloc[selected_r4]
    sel_captain_df_r4 = player_df_r4.iloc[captained_r4]
    # In Team Flag
    sel_names_r3 = player_df_r3.iloc[selected_r3]['Name'].astype(str).tolist() if selected_r3 else []
    sel_player_df_r4['In_Team'] = sel_player_df_r4['Name'].astype(str).isin(sel_names_r3).astype(int)
    print("----- Round 4 -----")
    print("Total Expected Points (rnd 4):", sum(sel_player_df_r4["exp_rnd_points"]) + sum(sel_captain_df_r4["exp_rnd_points"]))
    print("Total Team Cost (rnd 4):", sum(sel_player_df_r4["Price"]))
    print("Captain (rnd 4):", sel_captain_df_r4["Name"].values[0])
    print("Current Players Remaining (rnd 4):", sum(sel_player_df_r4["In_Team"]))
    
    # Round 5
    sel_player_df_r5 = player_df_r5.iloc[selected_r5]
    sel_captain_df_r5 = player_df_r5.iloc[captained_r5]
    # In Team Flag  
    sel_names_r4 = player_df_r4.iloc[selected_r4]['Name'].astype(str).tolist() if selected_r4 else []
    sel_player_df_r5['In_Team'] = sel_player_df_r5['Name'].astype(str).isin(sel_names_r4).astype(int)
    print("----- Round 5 -----")
    print("Total Expected Points (rnd 5):", sum(sel_player_df_r5["exp_rnd_points"]) + sum(sel_captain_df_r5["exp_rnd_points"]))
    print("Total Team Cost (rnd 5):", sum(sel_player_df_r5["Price"]))
    print("Captain (rnd 5):", sel_captain_df_r5["Name"].values[0])
    print("Current Players Remaining (rnd 5):", sum(sel_player_df_r5["In_Team"]))
    
    # Round 6
    sel_player_df_r6 = player_df_r6.iloc[selected_r6]
    sel_captain_df_r6 = player_df_r6.iloc[captained_r6]
    # In Team Flag
    sel_names_r5 = player_df_r5.iloc[selected_r5]['Name'].astype(str).tolist() if selected_r5 else []
    sel_player_df_r6['In_Team'] = sel_player_df_r6['Name'].astype(str).isin(sel_names_r5).astype(int)
    print("----- Round 6 -----")
    print("Total Expected Points (rnd 6):", sum(sel_player_df_r6["exp_rnd_points"]) + sum(sel_captain_df_r6["exp_rnd_points"]))
    print("Total Team Cost (rnd 6):", sum(sel_player_df_r6["Price"]))
    print("Captain (rnd 6):", sel_captain_df_r6["Name"].values[0])
    print("Current Players Remaining (rnd 6):", sum(sel_player_df_r6["In_Team"]))
    
    # Round 7
    sel_player_df_r7 = player_df_r7.iloc[selected_r7]
    sel_captain_df_r7 = player_df_r7.iloc[captained_r7]
    # In Team Flag
    sel_names_r6 = player_df_r6.iloc[selected_r6]['Name'].astype(str).tolist() if selected_r6 else []
    sel_player_df_r7['In_Team'] = sel_player_df_r7['Name'].astype(str).isin(sel_names_r6).astype(int)
    print("----- Round 7 -----")
    print("Total Expected Points (rnd 7):", sum(sel_player_df_r7["exp_rnd_points"]) + sum(sel_captain_df_r7["exp_rnd_points"]))
    print("Total Team Cost (rnd 7):", sum(sel_player_df_r7["Price"]))
    print("Captain (rnd 7):", sel_captain_df_r7["Name"].values[0])
    print("Current Players Remaining (rnd 7):", sum(sel_player_df_r7["In_Team"]))
    
    # Round 8
    sel_player_df_r8 = player_df_r8.iloc[selected_r8]
    sel_captain_df_r8 = player_df_r8.iloc[captained_r8]
    # In Team Flag
    sel_names_r7 = player_df_r7.iloc[selected_r7]['Name'].astype(str).tolist() if selected_r7 else []
    sel_player_df_r8['In_Team'] = sel_player_df_r8['Name'].astype(str).isin(sel_names_r7).astype(int)
    print("----- Round 8 -----")
    print("Total Expected Points (rnd 8):", sum(sel_player_df_r8["exp_rnd_points"]) + sum(sel_captain_df_r8["exp_rnd_points"]))
    print("Total Team Cost (rnd 8):", sum(sel_player_df_r8["Price"]))
    print("Captain (rnd 8):", sel_captain_df_r8["Name"].values[0])
    print("Current Players Remaining (rnd 8):", sum(sel_player_df_r8["In_Team"]))

    # Round 9  
    sel_player_df_r9 = player_df_r9.iloc[selected_r9]
    sel_captain_df_r9 = player_df_r9.iloc[captained_r9]
    # In Team Flag
    sel_names_r8 = player_df_r8.iloc[selected_r8]['Name'].astype(str).tolist() if selected_r8 else []
    sel_player_df_r9['In_Team'] = sel_player_df_r9['Name'].astype(str).isin(sel_names_r8).astype(int)
    print("----- Round 9 -----")
    print("Total Expected Points (rnd 8):", sum(sel_player_df_r9["exp_rnd_points"]) + sum(sel_captain_df_r9["exp_rnd_points"]))
    print("Total Team Cost (rnd 8):", sum(sel_player_df_r9["Price"]))
    print("Captain (rnd 8):", sel_captain_df_r9["Name"].values[0])
    print("Current Players Remaining (rnd 8):", sum(sel_player_df_r9["In_Team"]))

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

def optimise_fn_sim_fp(conf_int, sim_num, player_df_raw):
    # Run Optimisation Process for Specified Number of Simulations
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

        # Set confidence interval and calculate z score bounds
    lower_z_thresh, upper_z_thresh = z_score_bounds(conf_int)
    print(f"For {conf_int*100:.0f}% simulated points confidence interval the lower z score is {lower_z_thresh:.3f} and upper z score is {upper_z_thresh:.3f}")
    
        # Prepare DataFrame for Simulation Outputs
    all_sim_sel_players = pd.DataFrame()
    all_sim_sel_players["Name"] = []
    all_sim_sel_players["sim"] = []

    # Simulations Loop
    for i in range(sim_num):
        print(f"----- Simulation Run: {i+1} -----")
        # a. Calculate Player Z Score
        # Assign random z score for each df row within bounds and calculate simulated points
        player_df_raw["z_score"] = np.random.uniform(lower_z_thresh, upper_z_thresh, size=len(player_df_raw))
        player_df_raw["sim_points"] = player_df_raw["mean"] + (player_df_raw["z_score"] * player_df_raw["std_dev"])
        player_df_raw["sim_points"] = player_df_raw["sim_points"].clip(lower=0).round(0)  # Ensure no negative points
            
        # Aggregate by player name per round
        player_df = player_df_raw.groupby(['Name', 'Price', "Team", "Round", "Wk_f", "Bat_f", "Bowl_f", "Role", "weight","Available", "In_Team"], as_index=False).agg(
        exp_rnd_points=('sim_points',"sum"),
        games_in_round=('Round',"count"))
        
        # b. Calculate Player Price
        # Derive Features required for Pricing Models
        # Loop through all BBL15 players
        pts_per_game = player_df_raw[['Name', 'game_num', 'exp_points']]
        # Create empty dataframe to hold fantasy point outputs from the for loop
        bbl15_game_pts_table_pre = pd.DataFrame()
        bbl15_game_pts_table_pre["Name"] = []
        bbl15_game_pts_table_pre["game_num"] = []
        bbl15_game_pts_table_pre["curr_game_pts"] = []
        bbl15_game_pts_table_pre["prev_game_pts"] = []
        bbl15_game_pts_table_pre["two_prev_game_pts"] = []
        bbl15_game_pts_table_pre["last_2_games_ma_pts"] = []
        bbl15_game_pts_table_pre["last_3_games_ma_pts"] = []
        bbl15_game_pts_table_pre["seas_avg_games_pts"] = []

        # List of all the unique players in the bbl15 season    
        bbl15_play_list = pts_per_game['Name'].unique()

        # For loop for all BBL players in bbl15 season
        with ProcessPoolExecutor() as executor:
            for j in bbl15_play_list:
                full_play_pts_df = pts_per_game[pts_per_game['Name'] == j]
                full_play_pts_df = full_play_pts_df.sort_values(by = ['game_num'])
                full_play_pts_df['games_cnt'] = np.arange(len(full_play_pts_df)) + 1  

                # List of all the rounds played by individual    
                game_num_list = full_play_pts_df['games_cnt'].unique()

                # For loop for all individual rounds by player
                with ProcessPoolExecutor() as executor:
                    for k in game_num_list:
                        # Current Game
                        curr_game_play_pts_feat_df = full_play_pts_df[full_play_pts_df['games_cnt'] == k].rename(columns={"exp_points":"curr_game_pts"}).drop(columns=["games_cnt"], axis = 1)

                        # prior game points
                        prior_game_play_pts_df = full_play_pts_df[full_play_pts_df['games_cnt'] == k - 1].rename(columns={"exp_points":"prev_game_pts"})
                        prior_game_play_pts_df = prior_game_play_pts_df.drop(columns=["game_num", "games_cnt"], axis = 1)
                        curr_game_play_pts_feat_df = pd.merge(curr_game_play_pts_feat_df, prior_game_play_pts_df, left_on = ["Name"], right_on = ["Name"], how = "left")

                        # two prior round points
                        two_prior_game_play_pts_df = full_play_pts_df[full_play_pts_df['games_cnt'] == k - 2].rename(columns={"exp_points":"two_prev_game_pts"})
                        two_prior_game_play_pts_df = two_prior_game_play_pts_df.drop(columns=["game_num", "games_cnt"], axis = 1)
                        curr_game_play_pts_feat_df = pd.merge(curr_game_play_pts_feat_df, two_prior_game_play_pts_df, left_on = ["Name"], right_on = ["Name"], how = "left")

                        # Last two rounds moving average
                        if k >= 2:
                            ma_2_games_play_pts_df = full_play_pts_df[(full_play_pts_df['games_cnt'] <= k) & (full_play_pts_df['games_cnt'] >= k - 1)]
                            ma_2_games_play_pts_df = ma_2_games_play_pts_df.drop(columns=["game_num", "games_cnt"], axis = 1)
                            ma_2_games_play_pts_df_agg = ma_2_games_play_pts_df.groupby(["Name"], as_index=False).agg(
                            last_2_games_ma_pts = ('exp_points', "mean"),
                            )

                        else:
                            ma_2_games_play_pts_df_agg = full_play_pts_df[(full_play_pts_df['games_cnt'] == k)]
                            ma_2_games_play_pts_df_agg = ma_2_games_play_pts_df_agg.drop(columns=["game_num", "games_cnt", "exp_points"], axis = 1)
                            ma_2_games_play_pts_df_agg['last_2_games_ma_pts'] = np.nan
                                                                        
                        curr_game_play_pts_feat_df = pd.merge(curr_game_play_pts_feat_df, ma_2_games_play_pts_df_agg, left_on = ["Name"], right_on = ["Name"], how = "left")

                        # Prior 3 Games in the season aggregate attributes
                        if k >= 3:
                            ma_3_games_play_pts_df = full_play_pts_df[(full_play_pts_df['games_cnt'] <= k) & (full_play_pts_df['games_cnt'] >= k - 2)]
                            ma_3_games_play_pts_df = ma_3_games_play_pts_df.drop(columns=["game_num", "games_cnt"], axis = 1)
                            ma_3_games_play_pts_df_agg = ma_3_games_play_pts_df.groupby(["Name"], as_index=False).agg(
                            last_3_games_ma_pts = ('exp_points', "mean"),
                            )

                        else:
                            ma_3_games_play_pts_df_agg = full_play_pts_df[(full_play_pts_df['games_cnt'] == k)]
                            ma_3_games_play_pts_df_agg = ma_3_games_play_pts_df_agg.drop(columns=["game_num", "games_cnt","exp_points"], axis = 1)
                            ma_3_games_play_pts_df_agg['last_3_games_ma_pts'] = np.nan

                        curr_game_play_pts_feat_df = pd.merge(curr_game_play_pts_feat_df, ma_3_games_play_pts_df_agg, left_on = ["Name"], right_on = ["Name"], how = "left")
                        
                        # All prior games in the season aggregate attributes
                        play_seas_df = full_play_pts_df[full_play_pts_df['games_cnt'] < k+1]
                        play_seas_df = play_seas_df.drop(columns=["game_num", "games_cnt"], axis = 1)
                        play_seas_df_agg = play_seas_df.groupby(["Name"], as_index=False).agg(
                        seas_avg_games_pts = ('exp_points', "mean"),
                        )

                        curr_game_play_pts_feat_df = pd.merge(curr_game_play_pts_feat_df, play_seas_df_agg, left_on = ["Name"], right_on = ["Name"], how = "left")

                        # Add all during season attributes to empty table
                        bbl15_game_pts_table_pre = pd.concat([bbl15_game_pts_table_pre, curr_game_play_pts_feat_df])

        # Join round points features to price delta table
        # Add rnd column via player team
        player_team = player_df_raw.groupby(['Name', 'Team'])['exp_points'].count().reset_index().drop(['exp_points'], axis=1)

        bbl15_game_pts_table = pd.merge(bbl15_game_pts_table_pre, player_team, on = 'Name')

        # Get unique team round game counts
        game_rnd_team_df = player_df_raw.groupby(['Team','Round','game_num'])['exp_points'].count().reset_index().drop(['exp_points'], axis=1)
        bbl15_game_pts_table = pd.merge(bbl15_game_pts_table, game_rnd_team_df, on = ['Team', 'game_num']).drop(['Team'], axis=1)

        # For player double gameweek rounds, only return the second game row
        # Identify the max gameweek for each player in each round
        double_GW_index = bbl15_game_pts_table[['Name','Round','game_num']].sort_values(by=['Name', 'Round', 'game_num'], ascending=[True, True, False]).groupby(['Name', 'Round']).nth(0)

        # Select rows which only exist in double GW index
        bbl15_game_pts_table = pd.merge(bbl15_game_pts_table, double_GW_index, on = ['Name', 'Round', 'game_num'], how='inner')

        bbl15_game_pts_table['Round'] = bbl15_game_pts_table['Round'].astype("Int64")
        bbl15_game_pts_table = bbl15_game_pts_table.sort_values(['Name','Round'])

        # Add player price 
        bbl15_game_pts_table = pd.merge(bbl15_game_pts_table, price_df[['Name', 'Price']], on = 'Name', how = 'left').rename(columns={"Price":"price_pre"})

        # Extract Pricing Models
        price_model_obj_1 = joblib.load(os.path.join(directory, 'python_script/pricing-models/models/bbl15_price_model_1_game'))
        price_model_obj_2 = joblib.load(os.path.join(directory, 'python_script/pricing-models/models/bbl15_price_model_2_game'))
        price_model_obj_3 = joblib.load(os.path.join(directory, 'python_script/pricing-models/models/bbl15_price_model_3_game'))

        player_df_lags = bbl15_game_pts_table[['Name', 'Round', 'game_num', 'price_pre', 'seas_avg_games_pts', 'last_2_games_ma_pts', 'last_3_games_ma_pts']]

        # For Loop: For each player create rolling price prediction using previous round prediction as new pre price
        # Create empty player df dataframe to hold new player price dataframe
        player_df_new = pd.DataFrame()
        player_df_new["Name"] = []
        player_df_new["Price"] = []
        player_df_new["Team"] = []
        player_df_new["Round"] = []
        player_df_new["Wk_f"] = []
        player_df_new["Bat_f"] = []
        player_df_new["Bowl_f"] = []
        player_df_new["Role"] = []
        player_df_new["weight"] = []
        player_df_new["exp_rnd_points"] = []
        player_df_new["games_in_round"] = []

        for player in player_df['Name'].unique():
            for rnd in player_df_lags[player_df_lags['Name'] == player]['Round'].unique():                  
                # Filter player df lags for current round & player
                curr_rnd_player_df_lags = player_df_lags[(player_df_lags['Round'] == rnd) & (player_df_lags['Name'] == player)]

                # If current round append current row with price prediction
                if rnd == current_round:
                    player_df_new_row = player_df[(player_df['Round'] == rnd) & (player_df['Name'] == player)]
                    player_df_new = pd.concat([player_df_new, player_df_new_row])
                    
                # Check if player plays in the round
                if curr_rnd_player_df_lags.empty:
                    print(f"Player {player} does not play in round {rnd}")
                    continue

                # Next Available Round
                # unique rounds from player_df_lags
                player_rounds = player_df_lags[player_df_lags['Name'] == player]['Round'].unique()
                # Find next available round
                next_avail_rnd = min([r for r in player_rounds if r > rnd], default=None)
                print(f"Next available round for player {player} after round {rnd} is {next_avail_rnd}")
                if next_avail_rnd is None:
                    print(f"No more rounds for player {player} after round {rnd}")
                    continue

                # Split into game count dataframes
                curr_rnd_player_df_price_1 = curr_rnd_player_df_lags[curr_rnd_player_df_lags['game_num'] == 1][['Name', 'price_pre','Round','game_num', 'seas_avg_games_pts']]
                curr_rnd_player_df_price_2 = curr_rnd_player_df_lags[curr_rnd_player_df_lags['game_num'] == 2][['Name', 'price_pre','Round','game_num', 'last_2_games_ma_pts']]
                curr_rnd_player_df_price_3 = curr_rnd_player_df_lags[curr_rnd_player_df_lags['game_num'] >= 3][['Name', 'price_pre','Round','game_num', 'last_3_games_ma_pts']]

                # Predict Prices
                if not curr_rnd_player_df_price_1.empty:
                    curr_rnd_player_df_price_1['Price_Pred'] = price_model_obj_1.predict(curr_rnd_player_df_price_1[['price_pre','seas_avg_games_pts']])
                    curr_rnd_player_df_price_1 = curr_rnd_player_df_price_1[['Name', 'Round', 'game_num', 'Price_Pred']]
                    curr_rnd_player_df_price = curr_rnd_player_df_price_1.copy()
                if not curr_rnd_player_df_price_2.empty:
                    curr_rnd_player_df_price_2['Price_Pred'] = price_model_obj_2.predict(curr_rnd_player_df_price_2[['price_pre','last_2_games_ma_pts']])
                    curr_rnd_player_df_price_2 = curr_rnd_player_df_price_2[['Name', 'Round', 'game_num', 'Price_Pred']]
                    curr_rnd_player_df_price = curr_rnd_player_df_price_2.copy()
                if not curr_rnd_player_df_price_3.empty:
                    curr_rnd_player_df_price_3['Price_Pred'] = price_model_obj_3.predict(curr_rnd_player_df_price_3[['price_pre','last_3_games_ma_pts']])
                    curr_rnd_player_df_price_3 = curr_rnd_player_df_price_3[['Name', 'Round', 'game_num', 'Price_Pred']]
                    curr_rnd_player_df_price = curr_rnd_player_df_price_3.copy()
                if curr_rnd_player_df_price_1.empty and curr_rnd_player_df_price_2.empty and curr_rnd_player_df_price_3.empty:
                    print("No players in current round")
                    continue

                # Update player df lags price pre for next round
                player_df_lags_ind = pd.merge(player_df_lags, curr_rnd_player_df_price, left_on = ["Name", "Round", "game_num"], right_on = ["Name", "Round", "game_num"], how = "inner", suffixes=('', '_new'))
                player_df_lags_ind['price_pre'] = np.where(player_df_lags_ind['Price_Pred'].notnull(), player_df_lags_ind['Price_Pred'], player_df_lags_ind['price_pre'])
                player_df_lags_ind = player_df_lags_ind.drop(columns=['Price_Pred'])
                
                player_df_price = player_df_lags_ind.rename(columns={"price_pre": "Price_Pred"})

                # Increase round by 1 for price prediction to align with next round
                player_df_price['Round'] = next_avail_rnd
                player_df_pred_price = player_df_price[['Name', 'Round', 'Price_Pred']]

                # Merge price predictions back to main player df & append to new df
                player_df_new_row = pd.merge(player_df, player_df_pred_price, left_on = ["Name", "Round"], right_on = ["Name", "Round"], how = "inner")
                
                # Player price is actual price for current round, predicted price for subsequent rounds
                player_df_new_row['Price'] = np.where(player_df_new_row['Round'] == current_rnd, player_df_new_row['Price'], player_df_new_row['Price_Pred'])
                player_df_new_row = player_df_new_row.drop(columns=['Price_Pred'])

                # Append to new dataframe
                player_df_new = pd.concat([player_df_new, player_df_new_row])

        player_df = player_df_new.reset_index(drop=True)

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
        player_df_r1 = player_df[player_df['Round'] == 1].reset_index(drop=True)
        player_df_r2 = player_df[player_df['Round'] == 2].reset_index(drop=True)
        player_df_r3 = player_df[player_df['Round'] == 3].reset_index(drop=True)
        player_df_r4 = player_df[player_df['Round'] == 4].reset_index(drop=True)
        player_df_r5 = player_df[player_df['Round'] == 5].reset_index(drop=True)
        player_df_r6 = player_df[player_df['Round'] == 6].reset_index(drop=True)
        player_df_r7 = player_df[player_df['Round'] == 7].reset_index(drop=True)
        player_df_r8 = player_df[player_df['Round'] == 8].reset_index(drop=True)
        player_df_r9 = player_df[player_df['Round'] == 9].reset_index(drop=True)

        # 2. Run Optimisation
        # a. Sim Optimisation Variables Setup
        # Round 1
        points_r1 = player_df_r1["exp_rnd_points"]
        price_r1 = player_df_r1["Price"]
        weight_r1 = player_df_r1["weight"]
        in_team_r1 = player_df_r1["In_Team"]
        available_r1 = player_df_r1["Available"]
        wk_weight_r1 = player_df_r1["Wk_f"]
        bat_weight_r1 = player_df_r1["Bat_f"]
        bowl_weight_r1 = player_df_r1["Bowl_f"]
        play_cnt_r1, total_player_r1 = 12, range(len(price_r1))
        wk_cnt_r1, total_wk_r1 = 1, range(len(price_r1))
        bat_cnt_r1, total_bat_r1 = 6, range(len(price_r1))
        bowl_cnt_r1, total_bowl_r1 = 5, range(len(price_r1))
        budget_r1, total_budget_r1 = 1783500, range(len(price_r1))

        # Round 2
        points_r2 = player_df_r2["exp_rnd_points"]
        price_r2 = player_df_r2["Price"]
        weight_r2 = player_df_r2["weight"]
        in_team_r2 = player_df_r2["In_Team"]
        available_r2 = player_df_r2["Available"]
        wk_weight_r2 = player_df_r2["Wk_f"]
        bat_weight_r2 = player_df_r2["Bat_f"]
        bowl_weight_r2 = player_df_r2["Bowl_f"]
        play_cnt_r2, total_player_r2 = 12, range(len(price_r2))
        wk_cnt_r2, total_wk_r2 = 1, range(len(price_r2))
        bat_cnt_r2, total_bat_r2 = 6, range(len(price_r2))
        bowl_cnt_r2, total_bowl_r2 = 5, range(len(price_r2))
        budget_r2, total_budget_r2 = 1783500, range(len(price_r2))
        team_play_cnt_r2, total_team_player_r2 = 9, range(len(price_r2)) # At least 9 players from round 1 to be in round 2 team

        # Round 3
        points_r3 = player_df_r3["exp_rnd_points"]
        price_r3 = player_df_r3["Price"]
        weight_r3 = player_df_r3["weight"]
        in_team_r3 = player_df_r3["In_Team"]
        available_r3 = player_df_r3["Available"]
        wk_weight_r3 = player_df_r3["Wk_f"]
        bat_weight_r3 = player_df_r3["Bat_f"]
        bowl_weight_r3 = player_df_r3["Bowl_f"]
        play_cnt_r3, total_player_r3 = 12, range(len(price_r3))
        wk_cnt_r3, total_wk_r3 = 1, range(len(price_r3))
        bat_cnt_r3, total_bat_r3 = 6, range(len(price_r3))
        bowl_cnt_r3, total_bowl_r3 = 5, range(len(price_r3))
        budget_r3, total_budget_r3 = 1783500, range(len(price_r3))
        team_play_cnt_r3, total_team_player_r3 = 9, range(len(price_r3)) # At least 9 players from round 2 to be in round 3 team

        # Round 4
        points_r4 = player_df_r4["exp_rnd_points"]
        price_r4 = player_df_r4["Price"]
        weight_r4 = player_df_r4["weight"]
        in_team_r4 = player_df_r4["In_Team"]
        available_r4 = player_df_r4["Available"]
        wk_weight_r4 = player_df_r4["Wk_f"]
        bat_weight_r4 = player_df_r4["Bat_f"]
        bowl_weight_r4 = player_df_r4["Bowl_f"]
        play_cnt_r4, total_player_r4 = 12, range(len(price_r4))
        wk_cnt_r4, total_wk_r4 = 1, range(len(price_r4))
        bat_cnt_r4, total_bat_r4 = 6, range(len(price_r4))
        bowl_cnt_r4, total_bowl_r4 = 5, range(len(price_r4))
        budget_r4, total_budget_r4 = 1783500, range(len(price_r4))
        team_play_cnt_r4, total_team_player_r4 = 9, range(len(price_r4)) # At least 9 players from round 3 to be in round 4 team

        # Round 5
        points_r5 = player_df_r5["exp_rnd_points"]
        price_r5 = player_df_r5["Price"]
        weight_r5 = player_df_r5["weight"]
        in_team_r5 = player_df_r5["In_Team"]
        available_r5 = player_df_r5["Available"]
        wk_weight_r5 = player_df_r5["Wk_f"]
        bat_weight_r5 = player_df_r5["Bat_f"]
        bowl_weight_r5 = player_df_r5["Bowl_f"]
        play_cnt_r5, total_player_r5 = 12, range(len(price_r5))
        wk_cnt_r5, total_wk_r5 = 1, range(len(price_r5))
        bat_cnt_r5, total_bat_r5 = 6, range(len(price_r5))
        bowl_cnt_r5, total_bowl_r5 = 5, range(len(price_r5))
        budget_r5, total_budget_r5 = 1783500, range(len(price_r5))
        team_play_cnt_r5, total_team_player_r5 = 9, range(len(price_r5)) # At least 9 players from round 4 to be in round 5 team

        # Round 6
        points_r6 = player_df_r6["exp_rnd_points"]
        price_r6 = player_df_r6["Price"]
        weight_r6 = player_df_r6["weight"]
        in_team_r6 = player_df_r6["In_Team"]
        available_r6 = player_df_r6["Available"]
        wk_weight_r6 = player_df_r6["Wk_f"]
        bat_weight_r6 = player_df_r6["Bat_f"]
        bowl_weight_r6 = player_df_r6["Bowl_f"]
        play_cnt_r6, total_player_r6 = 12, range(len(price_r6))
        wk_cnt_r6, total_wk_r6 = 1, range(len(price_r6))
        bat_cnt_r6, total_bat_r6 = 6, range(len(price_r6))
        bowl_cnt_r6, total_bowl_r6 = 5, range(len(price_r6))
        budget_r6, total_budget_r6 = 1783500, range(len(price_r6))
        team_play_cnt_r6, total_team_player_r6 = 9, range(len(price_r6)) # At least 9 players from round 5 to be in round 6 team

        # Round 7
        points_r7 = player_df_r7["exp_rnd_points"]
        price_r7 = player_df_r7["Price"]
        weight_r7 = player_df_r7["weight"]
        in_team_r7 = player_df_r7["In_Team"]
        available_r7 = player_df_r7["Available"]
        wk_weight_r7 = player_df_r7["Wk_f"]
        bat_weight_r7 = player_df_r7["Bat_f"]
        bowl_weight_r7 = player_df_r7["Bowl_f"]
        play_cnt_r7, total_player_r7 = 12, range(len(price_r7))
        wk_cnt_r7, total_wk_r7 = 1, range(len(price_r7))
        bat_cnt_r7, total_bat_r7 = 6, range(len(price_r7))
        bowl_cnt_r7, total_bowl_r7 = 5, range(len(price_r7))
        budget_r7, total_budget_r7 = 1783500, range(len(price_r7))
        team_play_cnt_r7, total_team_player_r7 = 9, range(len(price_r7)) # At least 9 players from round 6 to be in round 7 team

        # Round 8
        points_r8 = player_df_r8["exp_rnd_points"]
        price_r8 = player_df_r8["Price"]
        weight_r8 = player_df_r8["weight"]
        in_team_r8 = player_df_r8["In_Team"]
        available_r8 = player_df_r8["Available"]
        wk_weight_r8 = player_df_r8["Wk_f"]
        bat_weight_r8 = player_df_r8["Bat_f"]
        bowl_weight_r8 = player_df_r8["Bowl_f"]
        play_cnt_r8, total_player_r8 = 12, range(len(price_r8))
        wk_cnt_r8, total_wk_r8 = 1, range(len(price_r8))
        bat_cnt_r8, total_bat_r8 = 6, range(len(price_r8))
        bowl_cnt_r8, total_bowl_r8 = 5, range(len(price_r8))
        budget_r8, total_budget_r8 = 1783500, range(len(price_r8))
        team_play_cnt_r8, total_team_player_r8 = 9, range(len(price_r8)) # At least 9 players from round 7 to be in round 8 team

        # Round 9
        points_r9 = player_df_r9["exp_rnd_points"]
        price_r9 = player_df_r9["Price"]
        weight_r9 = player_df_r9["weight"]
        in_team_r9 = player_df_r9["In_Team"]
        available_r9 = player_df_r9["Available"]
        wk_weight_r9 = player_df_r9["Wk_f"]
        bat_weight_r9 = player_df_r9["Bat_f"]
        bowl_weight_r9 = player_df_r9["Bowl_f"]
        play_cnt_r9, total_player_r9 = 12, range(len(price_r9))
        wk_cnt_r9, total_wk_r9 = 1, range(len(price_r9))
        bat_cnt_r9, total_bat_r9 = 6, range(len(price_r9))
        bowl_cnt_r9, total_bowl_r9 = 5, range(len(price_r9))
        budget_r9, total_budget_r9 = 1783500, range(len(price_r9))
        team_play_cnt_r9, total_team_player_r9 = 9, range(len(price_r9)) # At least 9 players from round 8 to be in round 9 team

        # b. Sim Optimisation Function Call
        sel_player_df, sel_player_df_r1, sel_player_df_r2, sel_player_df_r3, sel_player_df_r4, sel_player_df_r5, sel_player_df_r6, sel_player_df_r7,sel_player_df_r8, sel_player_df_r9 = optimise_fn_efp(
        # Round 1
        points_r1, price_r1, weight_r1, in_team_r1, available_r1, wk_weight_r1, bat_weight_r1, bowl_weight_r1,
        play_cnt_r1, total_player_r1, wk_cnt_r1, total_wk_r1, bat_cnt_r1, total_bat_r1, bowl_cnt_r1, total_bowl_r1,
        budget_r1, total_budget_r1, player_df_r1,
        # Round 2
        points_r2, price_r2, weight_r2, in_team_r2, available_r2, wk_weight_r2, bat_weight_r2, bowl_weight_r2,
        play_cnt_r2, total_player_r2, wk_cnt_r2, total_wk_r2, bat_cnt_r2, total_bat_r2, bowl_cnt_r2, total_bowl_r2,
        budget_r2, total_budget_r2, team_play_cnt_r2, total_team_player_r2, player_df_r2,
        # Round 3
        points_r3, price_r3, weight_r3, in_team_r3, available_r3, wk_weight_r3, bat_weight_r3, bowl_weight_r3,
        play_cnt_r3, total_player_r3, wk_cnt_r3, total_wk_r3, bat_cnt_r3, total_bat_r3, bowl_cnt_r3, total_bowl_r3,
        budget_r3, total_budget_r3, team_play_cnt_r3, total_team_player_r3, player_df_r3,
        # Round 4
        points_r4, price_r4, weight_r4, in_team_r4, available_r4, wk_weight_r4, bat_weight_r4, bowl_weight_r4,
        play_cnt_r4, total_player_r4, wk_cnt_r4, total_wk_r4, bat_cnt_r4, total_bat_r4, bowl_cnt_r4, total_bowl_r4,
        budget_r4, total_budget_r4, team_play_cnt_r4, total_team_player_r4, player_df_r4,
        # Round 5
        points_r5, price_r5, weight_r5, in_team_r5, available_r5, wk_weight_r5, bat_weight_r5, bowl_weight_r5,
        play_cnt_r5, total_player_r5, wk_cnt_r5, total_wk_r5, bat_cnt_r5, total_bat_r5, bowl_cnt_r5, total_bowl_r5,
        budget_r5, total_budget_r5, team_play_cnt_r5, total_team_player_r5, player_df_r5,
        # Round 6
        points_r6, price_r6, weight_r6, in_team_r6, available_r6, wk_weight_r6, bat_weight_r6, bowl_weight_r6,
        play_cnt_r6, total_player_r6, wk_cnt_r6, total_wk_r6, bat_cnt_r6, total_bat_r6, bowl_cnt_r6, total_bowl_r6,
        budget_r6, total_budget_r6, team_play_cnt_r6, total_team_player_r6, player_df_r6,
        # Round 7
        points_r7, price_r7, weight_r7, in_team_r7, available_r7, wk_weight_r7, bat_weight_r7, bowl_weight_r7,
        play_cnt_r7, total_player_r7, wk_cnt_r7, total_wk_r7, bat_cnt_r7, total_bat_r7, bowl_cnt_r7, total_bowl_r7,
        budget_r7, total_budget_r7, team_play_cnt_r7, total_team_player_r7, player_df_r7,
        # Round 8
        points_r8, price_r8, weight_r8, in_team_r8, available_r8, wk_weight_r8, bat_weight_r8, bowl_weight_r8,
        play_cnt_r8, total_player_r8, wk_cnt_r8, total_wk_r8, bat_cnt_r8, total_bat_r8, bowl_cnt_r8, total_bowl_r8,
        budget_r8, total_budget_r8, team_play_cnt_r8, total_team_player_r8, player_df_r8,
        # Round 9
        points_r9, price_r9, weight_r9, in_team_r9, available_r9, wk_weight_r9, bat_weight_r9, bowl_weight_r9,
        play_cnt_r9, total_player_r9, wk_cnt_r9, total_wk_r9, bat_cnt_r9, total_bat_r9, bowl_cnt_r9, total_bowl_r9,
        budget_r9, total_budget_r9, team_play_cnt_r9, total_team_player_r9, player_df_r9,
    )

        # 3. Store Simulation Results
        # Only Store Traded in players for storage efficiency
        sim_sel_players = sel_player_df_r1[['Name']][sel_player_df_r1['In_Team'] == 0]
        sim_sel_players['sim'] = i + 1
        all_sim_sel_players = pd.concat([all_sim_sel_players, sim_sel_players], ignore_index=True)

    # Return all simulation selected players
    return all_sim_sel_players