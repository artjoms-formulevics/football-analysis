# Football-Analysis/Predictor
 
Repo for football match outcome predictor

## Disclaimer

...

## Foreword

...

## Goal

...

## Structure

...

## Detailed Description

..

### Step 1: Data Gathering & Wrangling

The data is gathered from [Understat](https://understat.com/]) - a great sourse for in-depth football statistics (like xG and more). The data is scraped from Understat using python library [understat](https://github.com/amosbastian/understat) - an asynchronous Python package that helps to collect all the data available there.

#### Data collected

Data for six European football leagues is present on Understat:
* English Premier League 
* Spanish La Liga
* German Bundesliga
* Italian Serie A
* French Ligue 1
* Russian Premier League

The historical data is available starting from season 2014/15 up to last in-progress saeson 2020/21, totalling 7 seasons (6 full and 1 ongoing).

#### Collectable tables

Data on Understand is presented in various sections, hence it is required to use multiple functions to gather all the relevant data. Main information on season-level data is presented in three tables:
* Fixtures - data about key stats from each fixture in a given season in a given league. 
* Teams - some more detailed stats on each team's match.
* Players - season-level statistics of eaxh individual player (currently not used).

In addition, a match-level data for each single fixture in a season is gathered, that contains even more detailed stats and metrics. 

Foe each year/league the data is collected with a siple async loop and stored to json file:

```
async def collect_fixtures(year, league):
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        fixtures = await understat.get_league_results(
            league, year)
        with open(p+'/data/fixtures.json', 'w') as fp:
            json.dump(fixtures, fp)
```
#### Initial processing

As the original data is not structured in a way to be used later, it gets immediatly pre-processed (partially) during the gathering stage: some features are split or renamed to better represent the data, some (e.g. duplicating) are deleted from the data. Final features from the each dataset that are left, are presented below:

**Fixtures table**

After short processing, fixtures table consists of following columns:
| Column       | Type  | Description                                       |
|--------------|-------|---------------------------------------------------|
| datetime     | str   | timestamp of the start of the fixture             |
| goals_a      | int   | goals scored by away team                         |
| goals_h      | int   | goals scored by home team                         |
| h_forecast_d | float | probability of home team to draw (bookmaker odds) |
| h_forecast_l | float | probability of home team to lose (bookmaker odds) |
| h_forecast_w | float | probability of home team to win (bookmaker odds)  |
| id           | str   | id of a fixture                                   |
| id_a         | str   | id of the away team                               |
| id_h         | str   | id of the home team                               |
| isResult     | bool  | is the fixture finished?                          |
| team_a       | str   | away team name                                    |
| team_h       | str   | home team name                                    |
| xG_a         | float | expected goals (xG) made by away team             |
| xG_h         | float | expected goals (xG) made by home team             |

**Team table**

Team table consitsts of the data for each team's matches (team's stats and opponent's stats). This data is heavily transformed so that it would fit the home/away classification of the fixtures table and added to fixtures table (e.g. classify team as home/away and their opponent to the opposite side). The columns from team-level data that are used:

| Column      | Type  | Description                                                |
|-------------|-------|------------------------------------------------------------|
| deep_a      | int   | deep passes made by away team                              |
| deep_h      | int   | deep passes made by home team                              |
| npxG_a      | float | xG without penalties for away team                         |
| npxG_h      | float | xG without penalties for home team                         |
| ppda__att_a | dict  | passes per defensive action for away team (attacking side) |
| ppda_att_h  | dict  | passes per defensive action for home team (attacking side) |
| ppda_def_a  | dict  | passes per defensive action for away team (defensive side) |
| ppda_def_h  | dict  | passes per defensive action for home team (defensive side) |
| pts_a       | int   | points away team got from the match (0, 1 or 3)            |
| pts_h       | int   | points home team got from the match (0, 1 or 3)            |
| xpts_a      | float | expected points for the match for the away team            |
| xpts_h      | float | expected points for the match for the home team            |

**Match table**

Some stats are stored in the match-level tables for each match. Therefore, each fixture id is taken to make a query to the match table and collect & transform additional stats, that are appended to the existing dataset:

| Column        | Type    | Description                             |
|---------------|---------|-----------------------------------------|
| key_passes_a  | float64 | key passes made by away team            |
| key_passes_h  | float64 | key passes made by home team            |
| red_card_a    | float64 | red cards received by away team         |
| red_card_h    | float64 | red cards received by home team         |
| shots_a       | float64 | shots made by away team                 |
| shots_h       | float64 | shots made by home team                 |
| xA_a          | float64 | expected assists (xA) made by away team |
| xA_h          | float64 | expected assists (xA) made by home team |
| xGBuildup_a   | float64 |                                         |
| xGBuildup_h   | float64 |                                         |
| xGChain_a     | float64 |                                         |
| xGChain_h     | float64 |                                         |
| yellow_card_a | float64 | yellow cards received by away team      |
| yellow_card_h | float64 | yellow cards received by home team      |

#### Calculating ELO ratings

One of the major features to be used later is the [ELO rating](https://www.eloratings.net/about) of the team. The main idea & formula are taken from the article and ratings are calculated for each team using custom-made function. THe default value for the first year for each team was 1500 & default K=40. That adds two more columns to the dataset:

| Column        | Type    | Description                             |
|---------------|---------|-----------------------------------------|
| elo_a  | float64 | elo rating before the match of the away team            |
| elo_h  | float64 | elo rating before the match of the home team            |

### Step 2: Data Exploration

Data Exploration step is present in the notebook file "exploration.ipynb".
```
# Get shape of dataset
data.shape
(13853, 44)

# Describing the dataset
data.describe()
```

| goals_h |      goals_a |         xG_h |         xG_a | h_forecast_w | h_forecast_d | h_forecast_l |      shots_h |      shots_a | yellow_card_h |   red_card_h | ... |       xpts_a |       xpxG_h |   ppda_att_h |   ppda_def_h |       deep_h |       xpts_h |        pts_h |        elo_h | elo_a        |
|--------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|--------------:|-------------:|----:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|--------------|
|   count | 13853.000000 | 13853.000000 | 13853.000000 | 13853.000000 | 13853.000000 | 13853.000000 | 13853.000000 | 13853.000000 |  13853.000000 | 13853.000000 | ... | 13853.000000 | 13853.000000 | 13853.000000 | 13853.000000 | 13853.000000 | 13853.000000 | 13853.000000 | 13853.000000 | 13853.000000 |
|    mean |     1.508915 |     1.178950 |     1.476906 |     1.152637 |     0.447480 |     0.241903 |     0.310615 |    13.730672 |     11.223273 |     1.897206 | ... |     1.161337 |     1.338405 |   237.321302 |    25.014798 |     6.438750 |     1.584344 |     1.585577 |  1490.796304 |  1494.658844 |
|     std |     1.306599 |     1.152825 |     0.885260 |     0.767695 |     0.283574 |     0.111346 |     0.260603 |     5.273697 |      4.669162 |     1.322074 | ... |     1.276422 |     0.815276 |    83.262398 |     7.172355 |     4.296018 |     0.813286 |     1.317813 |   131.441556 |   131.447553 |
|     min |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |      0.000000 |     0.000000 | ... |     0.000000 |     0.000000 |    47.000000 |     4.000000 |     0.000000 |     0.000300 |     0.000000 |  1138.167883 |  1147.444700 |
|     25% |     1.000000 |     0.000000 |     0.818975 |     0.575886 |     0.198300 |     0.166500 |     0.088900 |    10.000000 |      8.000000 |     1.000000 | ... |     0.000000 |     0.737979 |   177.000000 |    20.000000 |     3.000000 |     0.914100 |     0.000000 |  1401.379862 |  1405.299680 |
|     50% |     1.000000 |     1.000000 |     1.319560 |     1.003800 |     0.427300 |     0.251200 |     0.239600 |    13.000000 |     11.000000 |     2.000000 | ... |     1.000000 |     1.182800 |   229.000000 |    24.000000 |     6.000000 |     1.612200 |     1.000000 |  1474.781373 |  1478.263238 |
|     75% |     2.000000 |     2.000000 |     1.968160 |     1.564340 |     0.680500 |     0.311300 |     0.487400 |    17.000000 |     14.000000 |     3.000000 | ... |     3.000000 |     1.773270 |   288.000000 |    30.000000 |     9.000000 |     2.264700 |     3.000000 |  1565.940424 |  1570.687567 |
|     max |    10.000000 |     9.000000 |     6.630490 |     6.186960 |     1.000000 |     0.859000 |     0.999700 |    47.000000 |     39.000000 |     8.000000 | ... |     3.000000 |     6.610910 |   633.000000 |    65.000000 |    42.000000 |     3.000000 |     3.000000 |  1957.502660 |  1958.810098 |

```
# Check if any NAs left
data.isna().sum()
```
| id            | 0 |
|---------------|---|
| isResult      | 0 |
| datetime      | 0 |
| team_h        | 0 |
| team_a        | 0 |
| id_h          | 0 |
| id_a          | 0 |
| goals_h       | 0 |
| goals_a       | 0 |
| xG_h          | 0 |
| xG_a          | 0 |
| h_forecast_w  | 0 |
| h_forecast_d  | 0 |
| h_forecast_l  | 0 |
| shots_h       | 0 |
| shots_a       | 0 |
| yellow_card_h | 0 |
| yellow_card_a | 0 |
| red_card_h    | 0 |
| red_card_a    | 0 |
| key_passes_h  | 0 |
| key_passes_a  | 0 |
| xA_h          | 0 |
| xA_a          | 0 |
| xGChain_h     | 0 |
| xGChain_a     | 0 |
| xGBuildup_h   | 0 |
| xGBuildup_a   | 0 |
| npxG_a        | 0 |
| ppda_att_a    | 0 |
| ppda_def_a    | 0 |
| deep_a        | 0 |
| xpts_a        | 0 |
| pts_a         | 0 |
| npxG_h        | 0 |
| ppda_att_h    | 0 |
| ppda_def_h    | 0 |
| deep_h        | 0 |
| xpts_h        | 0 |
| pts_h         | 0 |
| elo_h         | 0 |
| elo_a         | 0 |
| league        | 0 |
| match_id      | 0 |
| dtype: int64  |   |

```
# Data preview
data.head()
```
| id | isResult | datetime |              team_h |               team_a |        id_h | id_a | goals_h | goals_a |     xG_h | ... |   npxG_h | ppda_att_h | ppda_def_h | deep_h | xpts_h | pts_h |  elo_h |  elo_a | league | match_id                     |
|---:|---------:|---------:|--------------------:|---------------------:|------------:|-----:|--------:|--------:|---------:|----:|---------:|-----------:|-----------:|-------:|-------:|------:|-------:|-------:|-------:|------------------------------|
|  0 |     4749 |     True | 2014-08-16 12:45:00 |    Manchester United |     Swansea |   89 |      84 |       1 | 1.166350 | ... | 1.166350 |        253 |         25 |      8 | 2.2359 |     0 | 1500.0 | 1500.0 |    epl |      MSUaaacdeeeehinnnrssttw |
|  1 |     4750 |     True | 2014-08-16 15:00:00 |            Leicester |     Everton |   75 |      72 |       2 | 1.278300 | ... | 1.278300 |        362 |         25 |      1 | 1.9461 |     1 | 1500.0 | 1500.0 |    epl |             ELceeeeinorrsttv |
|  2 |     4751 |     True | 2014-08-16 15:00:00 |  Queens Park Rangers |        Hull |  202 |      91 |       0 | 1.900670 | ... | 1.126890 |        218 |         17 |      1 | 2.0149 |     0 | 1500.0 | 1500.0 |    epl |        HPQRaaeeegkllnnrrssuu |
|  3 |     4752 |     True | 2014-08-16 15:00:00 |                Stoke | Aston Villa |   85 |      71 |       0 | 0.423368 | ... | 0.423368 |        132 |         32 |      3 | 0.8041 |     0 | 1500.0 | 1500.0 |    epl |              ASVaeikllnoostt |
|  4 |     4753 |     True | 2014-08-16 15:00:00 | West Bromwich Albion |  Sunderland |   76 |      77 |       2 | 1.683430 | ... | 0.922260 |        184 |         31 |      6 | 2.0358 |     1 | 1500.0 | 1500.0 |    epl | ABSWabcddeehiillmnnnoorrstuw |

### Step 3: Data Cleaning & Processing

Now it is time clean and process the data necessary for the future model training. At this stage we firstly remove several columns that present way to detailed statistics that might be not fully relevant to predicting match goals. The columns removed are:

* ...

For all the features we are planning to use, we have to do some pre-processing. Namely, we do not know the exact statistics of the match before it was played. So we need to take historical averages for each team. For that we are splitting data on per-team level, sorting it historically and calculating averages from some time perion in the past. The periods that were tried are:

* Moving average through all the historical data
* Average from last 10 games
* Average from last 5 games
* Average from last 3 games

In the end, average of last 5 games was selected, as it was the most descriptive for the data. 

In this stage we also define out target variable, that we are willing to predict. This is going to be goal difference (simply goals by home team minus goals by away team). That would imply a win of home team in case of positive number, lose of home team in case of negative number and draw in case the value is equal to zero.

As after splitting to each team-level data, each game is represented twice (one game is played beween to team, so a game between Liverool and Manchester United will be part of both Liverpool's and United's subsets), the duplciates are identified and removed.

The data then is getting splitted to train and test set with the ratio of 90% / 10% and getting scaled to a normal distribution based on the training set.

### Step 4: Model Training

The model is built using TF 2.0 / Keras. Multiple models were examined for a long time. The final parameters selected were:

* Model = Keras Regressor
* Input layer with 20 neurons - ReLU & L2 regularization
* Three hidden layers (16, 10 & 6 neurons) - all ReLU
* Output layer - linear activation
* Optimizer - Adam (LR = 0.001)
* Loss = MSE (+ MAE for reporting)
* Batch Size = 10
* N of Epochs = 100

The model looks like this:

```
def keras_model():
    
    model = keras.Sequential()
    model.add(layers.Dense(20, input_dim=col_number, activation='relu', kernel_initializer='normal', kernel_regularizer='l2'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(6, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(1, activation='linear'))
    opt = keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=[tf.keras.metrics.MeanAbsoluteError()])
    model.summary()
    
    return model
    
estimator = KerasRegressor(build_fn=keras_model, epochs=100, batch_size=10, verbose=1,
                           validation_data=(X_test, y_test))
```


## Validation & Results

All models created were validated with multiple techniques:

* The regression metrics - MSE, MAE & R2 to determine the best regression model 


* Since goals are discrete, the results were also rounded to the nearest integer and compared to actual goal difference as classification metric. This did not yield consistent results, as there are several matches of an "outlier type" (i.e. win or lose by 5 or more goals) that appear only several times and model is having a hard time capturing outliers like this. Yet the predicrion of goal difference for some regular outcomes (i.e. in range [-3;3]) is yielding good resutls.


* Lastly, the outcomes were also grouped to just a 3-way outcome: win/lose/draw and again viewed as a classification problem - how many outcomes does the model predict correctly?



## Licensing

MIT License Copyright (c) 2021, Artjoms Formulevics
