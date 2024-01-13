import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

from selenium.webdriver import Firefox
from selenium.webdriver.firefox.options import Options
from bs4 import BeautifulSoup as bs

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def getSoup(url, driver=None):
    close = False
    if driver is None:
        close = True
        options = Options()
        options.add_argument('--headless')
        driver = Firefox(options=options)
    driver.get(url)
    soup = bs(driver.page_source, 'lxml')
    if close:
        driver.close()
    return soup

def getDate(date):
    try:
        new_date = datetime.strptime(date, '%a, %b %d, %Y').date()
    except ValueError:
        new_date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').date()
    return new_date

def getTeamsID(soup):
    table = soup.find('table', {'id': 'teams_active'})
    active_teams = table.findAll('th', {'data-stat': 'franch_name', 'scope': 'row'})

    teams_dict, team_idx = [], 1
    for team in active_teams:
        name = team.find('a').text
        link = team.find('a').get('href')
        id_ = link[-4:-1]
        
        if id_ == 'NJN':
            id_ = 'BRK'
        elif id_ == 'CHA':
            id_ = 'CHO'
        elif id_ == 'NOH':
            id_ = 'NOP'
        
        team_data = {'Name': name, 'ID': id_, 'IDX': team_idx}
        team_idx += 1
        teams_dict.append(team_data)
    teams_df = pd.DataFrame(teams_dict)
    return teams_df

def mapTeamID(team_id):
    try:
        df = pd.read_excel('data/teams_links.xlsx')
    except:
        df = getTeamsID(getSoup('https://www.basketball-reference.com/teams'))
        df.to_excel('data/teams_links.xlsx', index=0)
    
    idx = df.loc[df['ID'] == team_id]['IDX'].values[0]
    return idx

def getTeamGame(soup, team_id):
    teams = [x.text for x in soup.find('table', {'id': 'line_score'}).findAll('th', {'data-stat': 'team', 'scope':'row'})]
    try:
        local = teams.index(team_id)
    except:
        local = None
    
    if local is not None:
        visitor = abs(local-1)

        scores = [int(x.text) for x in soup.findAll('td', {'data-stat':'T'})]
        winner = 0 if scores[local] < scores[visitor] else 1
        pace = float(soup.find('td', {'data-stat':'pace'}).text)
        efg = [float(x.text) for x in soup.findAll('td', {'data-stat': 'efg_pct'})[:2]]
        tov = [float(x.text) for x in soup.findAll('td', {'data-stat': 'tov_pct'})[:2]]
        orb = [float(x.text) for x in soup.findAll('td', {'data-stat': 'orb_pct'})[:2]]
        ftr = [float(x.text) for x in soup.findAll('td', {'data-stat': 'ft_rate'})[:2]]
        ort = [float(x.text) for x in soup.findAll('td', {'data-stat': 'off_rtg'})[:2]]

        game_stats = {'Pts': scores[local], 'Pace': pace, 'eFG': efg[local], 'TOV': tov[local], 
                    'ORB': orb[local], 'FTR': ftr[local], 'ORT': ort[local],
                    'OppID': teams[visitor], 'OppPts': scores[visitor], 'OppeFG': efg[visitor], 'OppTOV': tov[visitor], 
                    'OppORB': orb[visitor], 'OppFTR': ftr[visitor], 'OppORT': ort[visitor],
                    'Location': local, 'Target': winner}
    else:
        game_stats = None

    return game_stats

def getTeamSeason(driver, soup, team_id):
    table = soup.find('table', {'id': 'games'}).find('tbody')
    games = table.findAll('tr')

    n_games, season_stats = 0, []
    for game in games:
        date = game.find('td', {'data-stat': 'date_game'})
        if date:
            n_games += 1
            game_link = game.find('td', {'data-stat': 'box_score_text'}).find('a').get('href')
            game_soup = getSoup('https://www.basketball-reference.com'+game_link, driver)
            game_data = getTeamGame(game_soup, team_id)
            streak = int(game.find('td', {'data-stat': 'game_streak'}).text.replace('W ', '+').replace('L ', '-'))

            new_date = getDate(date.text)
            if game_data:
                data = {'Game': n_games, 'Date': new_date, 'Streak': streak} | game_data
                season_stats.append(data)
            else:
                print(f'Error Scrapping {team_id} game on {new_date}. Link: {game_link}')
    
    return pd.DataFrame(season_stats)

def getTeamCurrentSeason(driver, soup, team_id):
    table = soup.find('table', {'id': 'games'}).find('tbody')
    games = table.findAll('tr')

    n_games, season_stats, next_games = 0, [], []
    for game in games:
        boxscore = game.find('td', {'data-stat': 'box_score_text'})
        if (boxscore):
            if (boxscore.text == 'Box Score'):
                n_games += 1
                date = game.find('td', {'data-stat': 'date_game'}).text
                new_date = getDate(date)
                game_link = boxscore.find('a').get('href')
                game_soup = getSoup('https://www.basketball-reference.com'+game_link, driver)
                game_data = getTeamGame(game_soup, team_id)
                streak = int(game.find('td', {'data-stat': 'game_streak'}).text.replace('W ','+').replace('L ', '-'))

                data = {'Game': n_games, 'Date': new_date, 'Streak': streak} | game_data
                season_stats.append(data)

            else:
                n_games += 1
                date = game.find('td', {'data-stat': 'date_game'}).text
                new_date = getDate(date)
                next_opp = game.find('td',{'data-stat': 'opp_name'}).find('a').get('href').split('/')[2]
                location = game.find('td', {'data-stat': 'game_location'}).text
                local = 0 if location == '@' else 1
                next_data = {'Date': new_date, 'OppID': next_opp, 'Location': local}
                next_games.append(next_data)

    return pd.DataFrame(season_stats), pd.DataFrame(next_games)

def getAllTeamsSeason(driver, n_years=1, current=True):
    teams_url = 'https://www.basketball-reference.com/teams/'
    teams_soup = getSoup(teams_url, driver)
    teams_df = getTeamsID(teams_soup)
    teams_id = list(teams_df['ID'])
    n_teams = len(teams_id)

    teams_df.to_excel('data/teams_links.xlsx', index=0)

    last_year = 2023
    initial_url = 'https://basketball-reference.com/teams/TID/YEAR_games.html'
    print(f'[{" "*(n_teams)}] ({(0/n_teams)*100:.2f}%)', end='\r')
    for idx, team_id in enumerate(teams_id):
        team_url = initial_url.replace('TID', team_id)
        
        try:
            excel = pd.ExcelFile(f'data/TeamsData/{team_id}.xlsx')
            dfs = {}
            for year in excel.sheet_names:
                df = pd.read_excel(excel, year)
                dfs[year] = df
        except:
            dfs = {}

        writer = pd.ExcelWriter(f'data/TeamsData/{team_id}.xlsx')
        for year in range(n_years):
            year_to_scrape = str(last_year-year)
            try:
                season_df = dfs[year_to_scrape]
                season_df['Date'] = season_df['Date'].apply(lambda x: getDate(x))

            except:
                season_url = team_url.replace('YEAR', year_to_scrape)
                season_soup = getSoup(season_url, driver)
                season_df = getTeamSeason(driver, season_soup, team_id)
        
            season_df.to_excel(writer, sheet_name=year_to_scrape, index=0)
        writer.close()

        if current:
            current_year = str(last_year+1)
            current_url = team_url.replace('YEAR', current_year)
            current_soup = getSoup(current_url, driver)
            current_df, next_games_df = getTeamCurrentSeason(driver, current_soup, team_id)

            with pd.ExcelWriter(f'data/CurrentSeason/{team_id}.xlsx') as writer:  
                current_df.to_excel(writer, sheet_name='Games', index=0)
                next_games_df.to_excel(writer, sheet_name='Next', index=0)

        print(f'[{"="*(idx+1)}{" "*(n_teams-idx-1)}] ({((idx+1)/n_teams)*100:.2f}%)', end='\r')

def getTeamRollingSeason(team_df):
    df = team_df.copy()
    df['Streak'] = df['Streak'].shift(1)
    df.loc[df.isna().any(axis=1), ['Streak']] = 0

    team_cols = ['Pts', 'Pace', 'eFG', 'TOV', 'ORB', 'FTR', 'ORT']
    final = df.copy()
    for idx in range(1, df.shape[0]):
        copy = df.copy()
        for col in team_cols:
            if col == 'Pace':
                copy[col] = df[col].transform(lambda x: x.shift(1).rolling(window=idx).mean())
            else:
                copy[col] = df[col].transform(lambda x: x.shift(1).rolling(window=idx).mean())
                copy['Opp'+col] = df['Opp'+col].transform(lambda x: x.shift(1).rolling(window=idx).mean())

        final.iloc[idx] = copy.iloc[idx]

    return final

def saveTeamsRollings(team_id):
    excel = pd.ExcelFile(f'data/TeamsData/{team_id}.xlsx')
    with pd.ExcelWriter(f'data/TeamsRolling/{team_id}.xlsx') as writer:
        for sheet in excel.sheet_names[::-1]:
            df = pd.read_excel(excel, sheet)
            rolling_df = getTeamRollingSeason(df)
            rolling_df['Date'] = rolling_df['Date'].apply(lambda x: getDate(str(x)))
            rolling_df.to_excel(writer, sheet_name=sheet, index=0)

def saveRollings():
    teams_url = 'https://www.basketball-reference.com/teams/'
    teams_soup = getSoup(teams_url)
    teams_df = getTeamsID(teams_soup)
    teams_id = list(teams_df['ID'])

    for team_id in teams_id:
        saveTeamsRollings(team_id)

def dataPrep(df, sheet):
    copy = df.copy()
    opponents = list(copy['OppID'])
    dates = list(copy['Date'])
    opp_cols = ['Pts', 'eFG', 'TOV', 'ORB', 'FTR', 'ORT']

    for idx, opp_id in enumerate(opponents):
        date = dates[idx]
        if sheet == 'Games':
            opp_df = pd.read_excel(f'data/CurrentSeason/{opp_id}.xlsx', sheet_name=sheet)
        else:
            opp_df = pd.read_excel(f'data/TeamsRolling/{opp_id}.xlsx', sheet_name=sheet)

        for col in opp_cols:
            copy.loc[copy['Date'] == date, ['Opp'+col]] = opp_df.loc[opp_df['Date'] == date][col].values[0]

    copy['Month'] = copy['Date'].dt.month
    copy['DayOfWeek'] = copy['Date'].dt.dayofweek
    copy['DaysOfRest'] = copy['Date'].diff().dt.days
    
    new_df = copy[copy.Game != 1]

    return new_df

def savePrepData(team_id):
    excel = pd.ExcelFile(f'data/TeamsRolling/{team_id}.xlsx')
    with pd.ExcelWriter(f'data/TeamsPrep/{team_id}.xlsx') as writer:
        for sheet in excel.sheet_names:
            df = pd.read_excel(excel, sheet)
            prep_df = dataPrep(df, sheet)
            prep_df['Date'] = prep_df['Date'].apply(lambda x: getDate(str(x)))
            prep_df.to_excel(writer, sheet_name=sheet, index=0)

def savePreps():
    teams_url = 'https://www.basketball-reference.com/teams/'
    teams_soup = getSoup(teams_url)
    teams_df = getTeamsID(teams_soup)
    teams_id = list(teams_df['ID'])

    for team_id in teams_id:
        savePrepData(team_id)

def train_model(df, model, batch_size, opp_id=True):
    if opp_id:
        features = ['Month', 'DayOfWeek', 'DaysOfRest', 'Streak', 'Pace', 'Location',
                    'Pts',  'eFG', 'TOV', 'ORB', 'FTR', 'ORT', 'OppID',
                    'OppPts', 'OppeFG', 'OppTOV', 'OppORB', 'OppFTR', 'OppORT']
        try:
            df['OppID'] = df['OppID'].apply(lambda x: mapTeamID(x))
        except:
            pass
    else:
        features = ['Month', 'DayOfWeek', 'DaysOfRest', 'Streak', 'Pace', 'Location',
                    'Pts',  'eFG', 'TOV', 'ORB', 'FTR', 'ORT',
                    'OppPts', 'OppeFG', 'OppTOV', 'OppORB', 'OppFTR', 'OppORT']

    batches = []
    for i in range(0, len(df), batch_size):
        batch_df = df[i:i+batch_size]
        if len(batch_df) != batch_size:
            batch_df = pd.concat([batches[-1], batch_df])
        batches.append(batch_df)

    train_racs, test_racs = [], []
    for idx, batch in enumerate(batches):
        X = batch[features]
        y = batch['Target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_rac = accuracy_score(y_train, y_pred_train)
        train_racs.append(train_rac)

        test_rac = accuracy_score(y_test, y_pred_test)
        test_racs.append(test_rac)

    return np.mean(train_racs), np.mean(test_racs), model

def getTrainedModel(df, model, return_best=True, show=True):
    batch_sizes = [2,5,10,15,20,30,50]
    max_acc, results = {'Batch': -1, 'Acc': 0}, {}
    for idx, batch_size in enumerate(batch_sizes):
        results[batch_size] = train_model(df, model, batch_size)
        if results[batch_size][1] >=  max_acc['Acc']:
            max_acc['Batch'] = batch_size
            max_acc['Acc'] = results[batch_size][1]

    if return_best:
        max_batch = max_acc['Batch']
        model = results[max_batch][2]
        train, test = results[max_batch][0], results[max_batch][1]
        if show:
            print(f'BatchSize = {max_batch}\nTrain Mean Accuracy: {train}\nTest Mean Accuracy: {test}')
    else:
        model = results[batch_sizes[-1]][2]
        train, test = results[batch_sizes[-1]][0], results[batch_sizes[-1]][1]
        if show:
            print(f'Train Mean Accuracy: {train}\nTest Mean Accuracy: {test}')

    return model, train, test

def predTeam(team_id, model):
    excel = pd.ExcelFile(f'data/TeamsPrep/{team_id}.xlsx')
    
    team_df = []
    for sheet in excel.sheet_names:
        df = pd.read_excel(excel, sheet)
        team_df.append(df)
    team_df = pd.concat(team_df, ignore_index=True)

    best_model, train_acc, test_acc = getTrainedModel(team_df, model, show=False)

    return best_model, train_acc, test_acc

def getTeamsPred(teams, model):
    teams_df = []
    for team in teams:
        best_model, train_acc, test_acc = predTeam(team, model)
        teams_df.append({'Team':team ,'Model': best_model, 'Train': train_acc, 'Test': test_acc})

    teams_df = pd.DataFrame(teams_df)
    sorted_df = teams_df.sort_values(by=['Test'], ascending=False)
    return sorted_df

def getCurrentPred(team_id, teams_df, opp_id=True):
    team_row = teams_df.loc[teams_df['Team'] == team_id]
    index = team_row.index[0]
    team_model = team_row['Model'].values[0]

    current_df = pd.read_excel(f'data/CurrentSeason/{team_id}.xlsx', sheet_name='Games')
    prep_df = dataPrep(current_df, 'Games')

    if opp_id:
        features = ['Month', 'DayOfWeek', 'DaysOfRest', 'Streak', 'Pace', 'Location',
                    'Pts',  'eFG', 'TOV', 'ORB', 'FTR', 'ORT', 'OppID',
                    'OppPts', 'OppeFG', 'OppTOV', 'OppORB', 'OppFTR', 'OppORT']
        prep_df['OppID'] = prep_df['OppID'].apply(lambda x: mapTeamID(x))

    else:
        features = ['Month', 'DayOfWeek', 'DaysOfRest', 'Streak', 'Pace', 'Location',
                    'Pts',  'eFG', 'TOV', 'ORB', 'FTR', 'ORT',
                    'OppPts', 'OppeFG', 'OppTOV', 'OppORB', 'OppFTR', 'OppORT']

    X = prep_df[features]
    y = prep_df ['Target']

    y_hat = team_model.predict(X)
    season_acc = accuracy_score(y, y_hat)

    return season_acc

def getTeamsCurrentPreds(model):
    teams = [team[:3] for team in os.listdir('data/TeamsPrep')]
    sorted_df = getTeamsPred(teams, model)

    current_preds_df = []
    for team in teams:
        team_acc = getCurrentPred(team, sorted_df)
        current_preds_df.append({'Team': team, 'Accuracy': team_acc})

    current_preds_df = pd.DataFrame(current_preds_df)
    current_sorted_df = current_preds_df.sort_values(by=['Accuracy'], ascending=False)
    return sorted_df, current_sorted_df

def getNextGameCols(games_played_df, next_games_df, next_opp_df, next_opp):
    cols = {'Game': games_played_df['Game'].values[-1], 'Date': next_games_df.iloc[0]['Date'].date(), 'Streak': games_played_df['Streak'].values[-1], 
        'Pace': games_played_df['Pace'].mean(), 'OppID': mapTeamID(next_opp), 'Location': next_games_df.iloc[0]['Location']}
    team_features = ['Pts', 'eFG', 'TOV', 'ORB', 'FTR', 'ORT']
    for col in team_features:
        col_mean = games_played_df[col].mean()
        cols[col] = col_mean
        
        opp_col = next_opp_df[f'Opp{col}'].mean()
        cols[f'Opp{col}'] = opp_col
        
    cols['Month'] = cols['Date'].month
    cols['DayOfWeek'] = cols['Date'].weekday()
    cols['DaysOfRest'] = (cols['Date'] - datetime.strptime(str(games_played_df['Date'].values[-1])[:10], '%Y-%m-%d').date()).days

    return cols

def predNextGame(team_id, today, sorted_df):
    model = sorted_df.loc[sorted_df['Team'] == team_id]['Model'].values[0]
    games_played_df = pd.read_excel(f'data/CurrentSeason/{team_id}.xlsx', sheet_name='Games')
    next_games_df = pd.read_excel(f'data/CurrentSeason/{team_id}.xlsx', sheet_name='Next')

    next_game_date = next_games_df.iloc[0]['Date'].date()
    if next_game_date == today:
        next_opp = next_games_df.iloc[0]['OppID']
        next_opp_df = pd.read_excel(f'data/CurrentSeason/{next_opp}.xlsx', sheet_name='Games')

        cols = getNextGameCols(games_played_df, next_games_df, next_opp_df, next_opp)
        game = pd.DataFrame([cols])

        features = ['Month', 'DayOfWeek', 'DaysOfRest', 'Streak', 'Pace', 'Location',
                    'Pts',  'eFG', 'TOV', 'ORB', 'FTR', 'ORT', 'OppID',
                    'OppPts', 'OppeFG', 'OppTOV', 'OppORB', 'OppFTR', 'OppORT']
        prediction = model.predict(game[features])[0]
        pred = {'Team': team_id, 'OppID': next_opp, 'Prediction': prediction}
    else:
        pred = None

    return pred

def predToday(model):
    sorted_df, current_sorted_df = getTeamsCurrentPreds(model)
    today = datetime.today().date()

    today_preds = []
    for team in sorted_df['Team']:
        pred = predNextGame(team, today, sorted_df)
        if pred:
            today_preds.append(pred)

    preds_df = pd.DataFrame(today_preds)
    return preds_df

def evaluatePredictions():
    df = pd.read_excel('tonight_preds.xlsx')
    results = np.empty(len(df), dtype=object)
    boxscores_url = 'https://www.basketball-reference.com/boxscores/'
    soup = getSoup(boxscores_url)

    winners = soup.find('div', {'class': 'game_summaries'}).findAll('tr', {'class': 'winner'})
    n_bets, n_wins = 0,0
    for winner in winners:
        team_winner = winner.find('a').get('href')[7:10]

        game, won = False, False        
        if team_winner in list(df['Team']):
            n_bets += 1
            game = True
            row = df.loc[df['Team'] == team_winner]
            index = row.index
            if row['Prediction'].values[0]:
                n_wins += 1
                won = True

        elif team_winner in list(df['OppID']):
            n_bets += 1
            game = True
            row = df.loc[df['OppID'] == team_winner]
            index = row.index
            if not row['Prediction'].values[0]:
                n_wins += 1
                won = True

        if game:
            results[index] = team_winner

    accuracy = n_wins/n_bets
    print(f'Bets: {n_bets}\nWon: {n_wins}\nAccuracy: {n_wins/n_bets}')
    
    df['Results'] = results
    return df, n_bets, n_wins, accuracy