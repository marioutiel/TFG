import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time

from selenium.webdriver import Firefox
from selenium.webdriver.firefox.options import Options
from bs4 import BeautifulSoup as bs

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def getSoup(url, driver=None, timer=None):
    if timer:
        time.sleep(timer)
    
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
    if team_id == 'BKN':
        team_id = 'BRK'

    teams = ['ATL', 'BOS', 'BRK', 'CHI', 'CHO', 'CLE', 'DAL', 'DEN', 'DET', 'GSW', 
             'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA','MIL', 'MIN', 'NOP', 'NYK', 
             'OKC', 'ORL', 'PHI', 'PHO', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS']
    return teams.index(team_id) + 1

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
    current_df = pd.read_excel(f'data/CurrentSeason/{team_id}.xlsx', 'Games')
    last_date = str(current_df['Date'].iloc[-1].date())

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
                if str(new_date) > last_date:
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

    new_games = pd.DataFrame(season_stats)
    next_games = pd.DataFrame(next_games)

    season_games = pd.concat([current_df, new_games])

    return season_games, next_games

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