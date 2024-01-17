import pandas as pd
import plotly.express as px

import dash
from dash import dcc, html
from dash.dependencies import Input, Output


app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.Div(className='header', children=[
        html.H1('NBA Team Dashboard'),
        
        html.Div(children=[
            html.Label('Select Team:'),
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

            html.Div(className='column', children=[
                html.Label('Select Metric:'),
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
                )
            ], style={'margin':'0%'}),

            html.Div(className='column', children=[
                html.Label('Select Season:'),
                dcc.Dropdown(
                    id='season-dropdown',
                    options=[
                        {'label': '2023-24', 'value':'2024'},
                        {'label': '2022-23', 'value':'2023'},
                        {'label': '2021-22', 'value':'2022'},
                        {'label': '2020-21', 'value':'2021'},
                        {'label': '2019-20', 'value':'2020'}
                    ],
                    value='2024'
                )
            ], style={'margin': '0%'}),
            
        ])
    ]),

    html.Div(children=[
        html.Div(className='column', children=[
            dcc.Graph(id='boxplot-team'),
            dcc.Graph(id='boxplot-opponent'),
            dcc.Graph(id='metric-distribution')
        ]),

        html.Div(className='column', children=[
            dcc.Graph(id='win-piechart'),
            dcc.Graph(id='win-lineplot'),
            dcc.Graph(id='streak-lineplot')
        ]),
    ]),

    html.Div(className='footer', children='''
        Created by Mario Utiel
        © 2024 NBA Dashboard
    ''')
])

# Define callback to update the plot based on the selected metric
@app.callback(
    [Output('boxplot-team', 'figure'),
     Output('boxplot-opponent', 'figure'),
     Output('metric-distribution', 'figure'),
     Output('win-piechart', 'figure'),
     Output('win-lineplot', 'figure'),
     Output('streak-lineplot', 'figure')],
    [Input('team-dropdown', 'value'),
     Input('season-dropdown', 'value'),
     Input('metric-dropdown', 'value')]
)
def update_plot(team, year, metric):
    if year == '2024':
        file_path = f'data/CurrentSeason/{team}.xlsx'
        team_df = pd.read_excel(file_path)
    else:
        file_path = f'data/TeamsData/{team}.xlsx'
        team_df = pd.read_excel(file_path, year)
    

    metric_plot = px.box(
        team_df,
        x=metric,
        points='all',
        title=metric,
        labels={metric: metric}
    )
    metric_plot.update_layout(title_x=0.5, title_font_size=20)

    opp_plot = px.box(
        team_df,
        x='Opp'+metric,
        points='all',
        title='Opponent '+metric,
        color_discrete_sequence=['red']
    )
    opp_plot.update_layout(title_x=0.5, title_font_size=20)

    win_local_metric_plot = px.box(
        team_df,
        y=metric,
        x='Target',
        color='Location',
        title=f'{metric} by Win (colored by Location)',
        labels={metric: metric, 'Win':'Win'}
    )
    win_local_metric_plot.update_layout(title_x=0.5, title_font_size=20)

    win_percentage_pie_chart = px.pie(
        team_df,
        values=team_df['Target'].value_counts(),
        names=team_df['Target'].unique(),
        title='Win Percentage',
    )
    win_percentage_pie_chart.update_layout(title_x=0.5, title_font_size=20)

    win_plot = px.line(
        team_df,
        x='Game',
        y=team_df['Target'].eq(1).cumsum(),
        title=f'Wins Evolution',
        labels={'Game': 'Game', 'y': 'Total Wins'}
    )
    win_plot.update_layout(title_x=0.5, title_font_size=20)

    streak_plot = px.line(
        team_df,
        x='Game',
        y='Streak',
        title=f'Streak Evolution',
        labels={'Game': 'Game', 'y': 'Streak'}
    )
    streak_plot.update_layout(title_x=0.5, title_font_size=20)

    return (metric_plot, opp_plot, win_local_metric_plot,
           win_percentage_pie_chart, win_plot, streak_plot)

if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1', port='8080')