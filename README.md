![](strategy/Results/readme/project-banner.png)

## Welcome
This is my project repository for my second iteration of my Big Bash League (BBL) AI Fantasy Team. By leveraging the ball by ball data of each match from the previous ten BBL seasons, I created a sophisticated decision making AIML system which leverages a bespoke optimisation process and several ML forecasting models to select my initial 15 player squad prior to the start of the tournament and to identify the optimal player trades for each of the 9 rounds during the season.

## AI Team Performance
### Individual Performance
![](strategy/Results/Round-9/ChrisLynnTheorem/Team%20Summary.png)

![](strategy/Results/Overall/ChrisLynnTheorem/history_table.png)

### League Performance
![](strategy/Results/Overall/ChrisLynnTheorem/league_table.png)

## AI Team Build


### Data Collection & Manipulation
- **Raw Data:** Extracted ball by ball BBL data from Cricsheet (https://cricsheet.org/downloads/) for each game in BBL01 to BBL13 which was stored in a single CSV file. Special thank you and shout out to Stephen Rushe of Cricsheet for creating this amazing dataset, which I have leveraged primarily for this project!

- **Venue Name Clean Up:** As the raw data includes over 13 seasons of BBL matches, the names of many venues have changed over the years. To build accurate venue trends, the collection of names for each venue was grouped and relabelled to a standard name.

- **Null & Missing Data:** Many rows in the data had some of the columns missing values or nulls. This was mainly tackled by overriding these fields as 0, which is appropriate as most variables lower limit is 0. 

### Response Variable and Explanatory Feature Creation
- **Response Variable:** Number of Fantasy Points (Bowling + Batting) the player will get in the game. As the raw data did not include individual fielding statistics, these points were added separately in the optimisation.
- **Explanatory Features Considered:** 
1. Player's previous season/s fantasy points summary statistics (up to 3 seasons prior)
2. Venue of game
3. Opposition team
4. Venue x Opposition interaction
5. Home Ground flag
6. Player's current season fantasy points summary statistics   
Team Rank
7. Opposition team rank
8. Player's overs bowled and batting position

### Model Build
- **Modelling Data Split:** 80% of the data randomly allocated for training set and remaining 20% for testing set. 
- **Model Pipeline:** Allows the user to select which variables to consider in the model, EDA between response variable and explanatory features, model builder loop and model performance metrics.
- **Model objects considered:** Linear Regression, Decision Trees, Random Forest, Gradient Boosting Machine & Explainable Boosting Machine.
- **Hyperparameter tuning:** For the machine learning model objects, unique hp grids were used to optimise the models, leveraging a 5 fold cross validation process on the training data to identify the best parameters. 
- **Model Performance Metrics:** MAE, MAPE, RMSE, R2 & Actual vs Expected plots to assess the overall model predictive power. Variable feature importance was used to assess most impactful features.

### Model Scoring Process
- **Overall process:** The scoring process first extracts the latest round's actual data for every player to rebuild the dataset used to create the model. Then the latest performance data is fed into the model to predict the expected fantasy points for every active player's remaining games. This process creates a final dataframe which has the 10 expected fantasy points predictions per player for each game in the season. 

### Optimisation
- **Optimisation set up:** Extract the final player scoring dataset and join additional fantasy features required for optimisation constraints e.g. player price. The dataset is then sliced and aggregated up to a player level, only considering the upcoming few rounds based on how many future rounds are selected. 
- **Objective Function:** Maximise the amount of fantasy points
- **Optimisation Constraints:** 
1. Number of players in the team = 12
2. Number of players from previous round team >= 9
3. Number of available players = 12
4. Number of wicketkeepers >= 1
5. Number of batters >= 6
6. Number of bowlers >= 5
7. Team Budget < current round budget

## Tournament Overview

### <ins>Round 1</ins>
![](strategy/Results/Round-1/ChrisLynnTheorem/Team%20Summary.png)
- AI Captain: Beau Webster (132 points - 17.93%)

**Selected Team**

![](strategy/Results/Round-1/ChrisLynnTheorem/Team%20Performance.png)

**Champion vs Challenger vs Semi-Pro Result**


### <ins>Round 2</ins>
**AI Team Round Trades**

| Traded In  | Fantasy Points | Traded Out |
| :---       |     :---:      | :---       |
| L.Pope     | 156            | J.Clarke   |
| H.Thornton | 106            | B.Couch    |
| P.Walter   | 44             | C.Lynn     |

![](strategy/Results/Round-2/ChrisLynnTheorem/Team%20Summary.png)
- AI Captain: Matt Short (230 points - 20.05%)

**Selected Team**

![](strategy/Results/Round-2/ChrisLynnTheorem/Team%20Performance.png)

**Champion vs Challenger vs Semi-Pro Result**


### <ins>Round 3</ins>
**AI Team Round Trades**

| Traded In  | Fantasy Points | Traded Out    |
| :---       |     :---:      | :---          |
| J.Edwards  | 24             | J.Behrendorff |
| H.Kerr     | 104            | H.Cartwright  |
| M.Stoinis  | 51             | B.Webster     |

![](strategy/Results/Round-3/ChrisLynnTheorem/Team%20Summary.png)
- AI Captain: Jack Edwards (24 points - 2.93%)

**Selected Team**

![](strategy/Results/Round-3/ChrisLynnTheorem/Team%20Performance.png)

**Champion vs Challenger vs Semi-Pro Result**

### <ins>Round 4</ins>
**AI Team Round Trades**

| Traded In    | Fantasy Points | Traded Out |
| :---         |     :---:      | :---       |
| J.Brown      | 128            | T.Curran   |
| F.O'Niell    | 16             | M.Short    |
| W.Sutherland | 66             | P.Walter   |

![](strategy/Results/Round-4/ChrisLynnTheorem/Team%20Summary.png)
- AI Captain: Jamie Overton (142 points - 18.56%)

**Selected Team**

![](strategy/Results/Round-4/ChrisLynnTheorem/Team%20Performance.png)

**Champion vs Challenger vs Semi-Pro Result**

### <ins>Round 5</ins>
**AI Team Round Trades**

| Traded In  | Fantasy Points | Traded Out  |
| :---       |     :---:      | :---        |
| W.Agar     | 4              | B.Dhawshuis |
| S.Billings | 13             | H.Kerr      |
| L.Ferguson | 8              | S.Konstas   |
| *D.Sams*   | *15*           | *F.O'Niell* |

![](strategy/Results/Round-5/ChrisLynnTheorem/Team%20Summary.png)
- AI Captain: Henry Thornton (54 points - 14.83%)

**Selected Team**

![](strategy/Results/Round-5/ChrisLynnTheorem/Team%20Performance.png)

**Champion vs Challenger vs Semi-Pro Result**

### <ins>Round 6</ins>
**AI Team Round Trades**

| Traded In  | Fantasy Points | Traded Out   |
| :---       |     :---:      | :---         |
| C.Green    | 10             | L.Pope       |
| S.Johnson  | 81             | J.Overton    |
| G.Sandhu   | 19             | D.Sams       |
| *D.Warner* | *70*           | *H.Thornton* |

![](strategy/Results/Round-6/ChrisLynnTheorem/Team%20Summary.png)
- AI Captain: Wes Agar (48 points - 7.21%)

**Selected Team**

![](strategy/Results/Round-6/ChrisLynnTheorem/Team%20Performance.png)

**Champion vs Challenger vs Semi-Pro Result**

### <ins>Round 7</ins>
**AI Team Round Trades**

| Traded In   | Fantasy Points | Traded Out   |
| :---        |     :---:      | :---         |
| P.Hatzoglu  | 14             | L.Ferguson   |
| C.Jordan    | 48             | G.Sandhu     |
| B.McDermott | 0              | M.Stoinis    |

![](strategy/Results/Round-7/ChrisLynnTheorem/Team%20Summary.png)
- AI Captain: Will Sunderland (90 points - 16.89%)

**Selected Team**

![](strategy/Results/Round-7/ChrisLynnTheorem/Team%20Performance.png)

**Champion vs Challenger vs Semi-Pro Result**

### <ins>Round 8</ins>
**AI Team Round Trades**

| Traded In  | Fantasy Points | Traded Out   |
| :---       |     :---:      | :---         |
| N.Ellis    | 143            | W.Agar       |
| M.Owen     | 219            | G.McDermott  |
| B.Stanlake | 13             | P.Siddle     |

![](strategy/Results/Round-8/ChrisLynnTheorem/Team%20Summary.png)
- AI Captain: Peter Hatzoglou (22 points - 2.74%)

**Selected Team**

![](strategy/Results/Round-8/ChrisLynnTheorem/Team%20Performance.png)

**Champion vs Challenger vs Semi-Pro Result**

### <ins>Round 9</ins>

**AI Team Round Trades**

| Traded In  | Fantasy Points | Traded Out   |
| :---       |     :---:      | :---         |
| J.Clarke   | 0              | C.Green      |
| M.Neser    | 42             | B.Stanlake   |
| T.Rogers   | 30             | D.Warner     |

![](strategy/Results/Round-9/ChrisLynnTheorem/Team%20Summary.png)
- AI Captain: Nathan Ellis (178 points - 27.22%)

**Selected Team**

![](strategy/Results/Round-9/ChrisLynnTheorem/Team%20Performance.png)

**Champion vs Challenger vs Semi-Pro Result**

## AI Team Next Season Improvements

### Feature Creation
- **Power Surge:** Currently used basic power surge features, but next season complete history over the last 5 years should be used to build more accurate features.
- **Fielding Points:** Improved methodology for the fielding points proxy or even individual past season fielding features leveraging webscraping of cricinfo scorecards.
- **Player Labelling:** Develop approach to assign player labels for all modelling data (e.g. domestic vs international players, rookie player vs veteran player etc.)
- **Pre Tournament other domestic tournament records:** Currently new players to the competition, either young players or international signings can not be differentiated due to no past season records. Other domestic tournaments can be used as a proxy to help differentiate players. 
- **Improve Scoring Process:** Reduce daily manual scoring process during BBL season by leveraging webscraping to automate the data capturing process from Cricinfo and BBL Supercoach App. Should look into automating scoring and optimisation process leveraging agents and github actions.

### Modelling
- **Alternative Modelling Techniques:** Currently all models are XGBoost models, but should look into 
- **Explainable & Casual Modelling Techniques:** As can be seen in many of the models, they consist of several feature all providing similar amounts of information. Leveraging advanced explainable & casual ML techniques could help dive deeper to identify the true key drivers and features of player performance.
- **Rebuilding current models:** Due to the numerous models built and features created, their are several different options and experiments which can be considered to construct the best solution. Though I consider several approaches, due to time limitations and the face pace of the tournament I definetely did not exhaust all possible modelling ideas and I would like to focus deeper into the raw modelling.

### Strategy (via Optimisation)
- This is the section which I developed the most enhancements and I was very happy with its performance. 
- **Addition of new rules:** Build additional constraints to capture new fantasy rules
