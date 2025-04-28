# TM2
TCDformer-based Momentum Transfer Model for Long-term Sports Prediction Open source code

## Dataset
We conducted experiments on multiple sports event datasets, including ETTh1, NBA, and Beach Volleyball. A brief description is as follows:

(1) ETTh1 contains the 2023 Wimbledon men's tournament, with detailed match data for each player. The dataset originally contained 7,285 rows and 49 columns, totaling 356,965 data points.

(2) The NBA Dataset comprises multiple key files:

	1. teams.csv: Contains team information including names, cities, historical data, and home arenas. This enables analysis of franchise evolution, cultural context, and performance trends across seasons.
	
	2. games_details.csv: Records granular game statistics (date, teams, venue, score, player minutes, points, rebounds, assists). Facilitates tactical analysis, player performance evaluation, and predictive modeling for future outcomes.
	
	3. games.csv: Provides macro-level game metadata (season, game type – regular/playoffs, results). Essential for building win probability models and studying team performance patterns against specific opponents.
	
	4. ranking.csv: Documents seasonal standings with win-loss records and win percentages. Reveals competitive landscapes, identifies consistently dominant teams, and tracks short-term performance fluctuations.
	
	5. players.csv: Centralizes player profiles and seasonal metrics (PPG, RPG, APG, blocks, steals). Critical for assessing individual skills, role dynamics within teams, and predicting career trajectories.
	
(3) The Beach Volleyball Dataset contains game results, player details, and match statistics for selected games. All games are 2-on-2, so there are 2 columns for winners (1 team) and 2 columns for losers (1 team). Although there are some duplicate columns and the data is wide due to 2 players per team, the data is relatively ready for analysis and cleaned.

ETTh1, NBA dataset, and Beach Volleyball dataset all have more change points. ETTh1 is a trend change of a single person, while NBA dataset and beach volleyball dataset are both team trend change analysis. According to the standard protocol, we divide the dataset into training set, validation set, and test set in chronological order. The ratio of ETTh1, NBA dataset, and beach volleyball dataset is 8:1:1.

## Benchmarks models
To validate our model, we conduct extensive experiments comparing TM2 with baseline models (ELO, decision trees, logistic regression , support vector machines, random forests) and advanced neural architectures (Mamba4Cast , Seq2Event) on multiple evaluation metrics. 
