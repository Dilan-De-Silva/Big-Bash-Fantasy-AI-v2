# Optimisation Functions
# Imports
import pandas as pd
import numpy as np
import os
import random

# EFP Optimisation Function
def optimise_fn_efp(# Round 1
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
    # 1. MIP Optimsation Setup
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
    # Initialize z_shared for round 1 to round 2 linking
    name_to_idx_r1 = {name: idx for idx, name in enumerate(player_df_r1['Name'].astype(str))}
    z_shared = {}
    for j, name in enumerate(player_df_r2['Name'].astype(str)):
        if name in name_to_idx_r1:
            i = name_to_idx_r1[name]
            z_shared[(i, j)] = m.add_var(var_type=BINARY)
            
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
            m += xsum(z_rnd[(i, j)] for (i, j) in z_rnd.keys()) >= team_min

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
    m += xsum(price_r1[i] * x_r1[i] for i in total_budget_r1) <= budget_r1
    m += xsum(available_r1[i] * x_r1[i] for i in total_player_r1) == play_cnt_r1
    # Round 2  
    m += xsum(wk_weight_r2[i] * x_r2[i] for i in total_wk_r2) >= wk_cnt_r2
    m += xsum(bat_weight_r2[i] * x_r2[i] for i in total_bat_r2) >= bat_cnt_r2
    m += xsum(bowl_weight_r2[i] * x_r2[i] for i in total_bowl_r2) >= bowl_cnt_r2
    m += xsum(price_r2[i] * x_r2[i] for i in total_budget_r2) <= budget_r2
    m += xsum(available_r2[i] * x_r2[i] for i in total_player_r2) == play_cnt_r2
    # Round 3
    m += xsum(wk_weight_r3[i] * x_r3[i] for i in total_wk_r3) >= wk_cnt_r3
    m += xsum(bat_weight_r3[i] * x_r3[i] for i in total_bat_r3) >= bat_cnt_r3
    m += xsum(bowl_weight_r3[i] * x_r3[i] for i in total_bowl_r3) >= bowl_cnt_r3
    m += xsum(price_r3[i] * x_r3[i] for i in total_budget_r3) <= budget_r3
    m += xsum(available_r3[i] * x_r3[i] for i in total_player_r3) == play_cnt_r3
    # Round 4
    m += xsum(wk_weight_r4[i] * x_r4[i] for i in total_wk_r4) >= wk_cnt_r4
    m += xsum(bat_weight_r4[i] * x_r4[i] for i in total_bat_r4) >= bat_cnt_r4
    m += xsum(bowl_weight_r4[i] * x_r4[i] for i in total_bowl_r4) >= bowl_cnt_r4
    m += xsum(price_r4[i] * x_r4[i] for i in total_budget_r4) <= budget_r4
    m += xsum(available_r4[i] * x_r4[i] for i in total_player_r4) == play_cnt_r4
    # Round 5
    m += xsum(wk_weight_r5[i] * x_r5[i] for i in total_wk_r5) >= wk_cnt_r5
    m += xsum(bat_weight_r5[i] * x_r5[i] for i in total_bat_r5) >= bat_cnt_r5
    m += xsum(bowl_weight_r5[i] * x_r5[i] for i in total_bowl_r5) >= bowl_cnt_r5
    m += xsum(price_r5[i] * x_r5[i] for i in total_budget_r5) <= budget_r5
    m += xsum(available_r5[i] * x_r5[i] for i in total_player_r5) == play_cnt_r5
    # Round 6
    m += xsum(wk_weight_r6[i] * x_r6[i] for i in total_wk_r6) >= wk_cnt_r6
    m += xsum(bat_weight_r6[i] * x_r6[i] for i in total_bat_r6) >= bat_cnt_r6
    m += xsum(bowl_weight_r6[i] * x_r6[i] for i in total_bowl_r6) >= bowl_cnt_r6
    m += xsum(price_r6[i] * x_r6[i] for i in total_budget_r6) <= budget_r6
    m += xsum(available_r6[i] * x_r6[i] for i in total_player_r6) == play_cnt_r6
    # Round 7
    m += xsum(wk_weight_r7[i] * x_r7[i] for i in total_wk_r7) >= wk_cnt_r7
    m += xsum(bat_weight_r7[i] * x_r7[i] for i in total_bat_r7) >= bat_cnt_r7
    m += xsum(bowl_weight_r7[i] * x_r7[i] for i in total_bowl_r7) >= bowl_cnt_r7
    m += xsum(price_r7[i] * x_r7[i] for i in total_budget_r7) <= budget_r7
    m += xsum(available_r7[i] * x_r7[i] for i in total_player_r7) == play_cnt_r7
    # Round 8
    m += xsum(wk_weight_r8[i] * x_r8[i] for i in total_wk_r8) >= wk_cnt_r8
    m += xsum(bat_weight_r8[i] * x_r8[i] for i in total_bat_r8) >= bat_cnt_r8
    m += xsum(bowl_weight_r8[i] * x_r8[i] for i in total_bowl_r8) >= bowl_cnt_r8
    m += xsum(price_r8[i] * x_r8[i] for i in total_budget_r8) <= budget_r8
    m += xsum(available_r8[i] * x_r8[i] for i in total_player_r8) == play_cnt_r8
    # Round 9
    m += xsum(wk_weight_r9[i] * x_r9[i] for i in total_wk_r9) >= wk_cnt_r9
    m += xsum(bat_weight_r9[i] * x_r9[i] for i in total_bat_r9) >= bat_cnt_r9
    m += xsum(bowl_weight_r9[i] * x_r9[i] for i in total_bowl_r9) >= bowl_cnt_r9
    m += xsum(price_r9[i] * x_r9[i] for i in total_budget_r9) <= budget_r9
    m += xsum(available_r9[i] * x_r9[i] for i in total_player_r9) == play_cnt_r9

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
    print("Selected items (rnd 1): {}".format(selected_r1))
    print("Captain selected (rnd 1): {}".format(captained_r1))

    print("Selected items (rnd 2): {}".format(selected_r2))
    print("Captain selected (rnd 2): {}".format(captained_r2))

    # Optimal Team Output
    # Round 1
    sel_player_df_r1 = player_df_r1.iloc[selected_r1]
    sel_captain_df_r1 = player_df_r1.iloc[captained_r1]
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
    print("Total Expected Points (rnd 8):", sum(sel_player_df_r8["exp_rnd_points"]) + sum(sel_captain_df_r8["exp_rnd_points"]))
    print("Total Team Cost (rnd 8):", sum(sel_player_df_r8["Price"]))
    print("Captain (rnd 8):", sel_captain_df_r8["Name"].values[0])
    print("Current Players Remaining (rnd 8):", sum(sel_player_df_r8["In_Team"]))
    # Round 9  
    sel_player_df_r9 = player_df_r9.iloc[selected_r9]
    sel_captain_df_r9 = player_df_r9.iloc[captained_r9]
    # In Team Flag
    sel_names_r8 = player_df_r7.iloc[selected_r8]['Name'].astype(str).tolist() if selected_r8 else []
    sel_player_df_r9['In_Team'] = sel_player_df_r9['Name'].astype(str).isin(sel_names_r8).astype(int)
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
    
    return sel_player_df, sel_captain_df_r1, sel_captain_df_r2, sel_captain_df_r3, sel_captain_df_r4, sel_captain_df_r5, sel_captain_df_r6, sel_captain_df_r7, sel_captain_df_r8, sel_captain_df_r9