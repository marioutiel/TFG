# Import libraries
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd



# Create a Dash web application
app = dash.Dash(__name__)

app.css.append_css({'external_url': 'style.css'})

# Define the layout of the web app
app.layout = html.Div(className='container', children=[
    html.H1(className='header' ,children='NBA Visualization Tool'),

    html.H2(children='Team Metric Statistics'),

    html.Div(className='dropwdown-container', children=[
        html.Div(children='''
            Select a team:
        '''),

        dcc.Dropdown(
            id='team-dropdown',
            options=[
                {'label': 'Atlanta Hawks', 'value': 'ATL'},
                {'label': 'Boston Celtics', 'value': 'BOS'},
                {'label': 'Brooklyn Nets', 'value': 'BRK'},
                {'label': 'Chicago Bulls', 'value': 'CHI'},
                {'label': 'Charlotte Hornets', 'value': 'CHO'},
                {'label': 'Cleveland Cavaliers', 'value': 'CLE'},
                {'label': 'Dallas Mavericks', 'value': 'DAL'},
                {'label': 'Denver Nuggets', 'value': 'DEN'},
                {'label': 'Detroit Pistons', 'value': 'DET'},
                {'label': 'Golden State Warriors', 'value': 'GSW'},
                {'label': 'Houston Rockets', 'value': 'HOU'},
                {'label': 'Indiana Pacers', 'value': 'IND'},
                {'label': 'Los Angeles Clippers', 'value': 'LAC'},
                {'label': 'Los Angeles Lakers', 'value': 'LAL'},
                {'label': 'Memphis Grizzlies', 'value': 'MEM'},
                {'label': 'Miami Heat', 'value': 'MIA'},
                {'label': 'Milwaukee Bucks', 'value': 'MIL'},
                {'label': 'Minnesota Timberwolves', 'value': 'MIN'},
                {'label': 'New Orleans Pelicans', 'value': 'NOP'},
                {'label': 'New York Knicks', 'value': 'NYK'},
                {'label': 'Oklahoma City Thunder', 'value': 'OKC'},
                {'label': 'Orlando Magic', 'value': 'ORL'},
                {'label': 'Philadelphia 76ers', 'value': 'PHI'},
                {'label': 'Phoenix Suns', 'value': 'PHO'},
                {'label': 'Portland Trailblazers', 'value': 'POR'},
                {'label': 'Sacramento Kings', 'value': 'SAC'},
                {'label': 'San Antonio Spurs', 'value': 'SAS'},
                {'label': 'Toronto Raptors', 'value': 'TOR'},
                {'label': 'Utah Jazz', 'value': 'UTA'},
                {'label': 'Washington Wizards', 'value': 'WAS'},
            ],
            value='ATL'
        ),
    ]),

    html.Div(className='dropdown-container', children=[
        html.Div(children='''
            Select season to show:
        '''),

        dcc.Dropdown(
            id='year-dropdown',
            options=[
                {'label': '2023-24', 'value':'2024'},
                {'label': '2022-23', 'value':'2023'},
                {'label': '2021-22', 'value':'2022'},
                {'label': '2020-21', 'value':'2021'},
                {'label': '2019-20', 'value':'2020'}
            ],
            value='2024'
        ),
    ]),

    html.Div(className='dropdown-container', children=[
        html.Div(children='''
            Select metric to visualize:
        '''),

        dcc.Dropdown(
            id='metric-dropdown',
            options=[
                {'label': 'Points (Pts)', 'value': 'Pts'},
                {'label': 'Effective Field Goal Percentage (eFG)', 'value': 'eFG'},
                {'label': 'Turnovers (TOV)', 'value': 'TOV'},
                {'label': 'Offensive Rebounds (ORB)', 'value': 'ORB'},
                {'label': 'Free Throw Rate (FTR)', 'value': 'FTR'},
            ],
            value='Pts'
        ),
    ]),

    html.Div(className='graph-container', children=[
        dcc.Graph(
            id='game-metric-plot',
        ),
        
        dcc.Graph(
            id='opp-metric-plot'
        )
    ]),

    html.Div(className='graph-container', children=[
        dcc.Graph(
            id='metric-plot-win',
        ),
        
        dcc.Graph(
            id='metric-plot-local'
        )
    ]),

    html.H2(children='Team Evolution Statistics'),

    html.Div(className='graph-container', children=[
        dcc.Graph(
            id='win-evolution-plot',
        ),
    ]),

    html.Div(className='graph-container', children=[
        dcc.Graph(
            id='streak-evolution-plot',
        ),
    ]),

    html.Div(className='footer', children='Created by Mario Utiel')
])

# Define callback to update the plot based on the selected metric
@app.callback(
    [Output('game-metric-plot', 'figure'),
     Output('opp-metric-plot', 'figure'),
     Output('metric-plot-win', 'figure'),
     Output('metric-plot-local', 'figure'),
     Output('win-evolution-plot', 'figure'),
     Output('streak-evolution-plot', 'figure')],
    [Input('metric-dropdown', 'value'),
     Input('team-dropdown', 'value'),
     Input('year-dropdown', 'value')]
)
def update_plot(metric, team, year):
    if year == '2024':
        file_path = f'TFG/data/CurrentSeason/{team}.xlsx'
        team_df = pd.read_excel(file_path)
    else:
        file_path = f'TFG/data/TeamsData/{team}.xlsx'
        team_df = pd.read_excel(file_path, year)
    

    metric_plot = px.box(
        team_df,
        x=metric,
        points='all',
        title=metric,
        labels={metric: metric}
    )

    opp_plot = px.box(
        team_df,
        x='Opp'+metric,
        points='all',
        title='Opponent '+metric,
        color_discrete_sequence=['red']
    )

    win_metric_plot = px.box(
        team_df,
        y=metric,
        x='Target',
        title=f'{metric} by Win',
        labels={metric: metric, 'Win':'Win'}
    )

    local_metric_plot = px.box(
        team_df,
        y=metric,
        x='Location',
        title=f'{metric} by Location',
        labels={metric: metric, 'Win':'Win'},
        color_discrete_sequence=['red']
    )

    win_plot = px.line(
        team_df,
        x='Game',
        y=team_df['Target'].eq(x for x in [0,1]).cumsum(),
        color='Target',
        title=f'{team} Wins Evolution',
        labels={'Game': 'Game', 'y': 'Total Wins'}
    )

    streak_plot = px.line(
        team_df,
        x='Game',
        y='Streak',
        title=f'{team} Streak Evolution',
        labels={'Game': 'Game', 'y': 'Streak'}
    )

    return metric_plot, opp_plot, win_metric_plot, local_metric_plot, win_plot, streak_plot

# Run the web app
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')
