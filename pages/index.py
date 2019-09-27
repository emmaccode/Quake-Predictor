import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px

from app import app
column1 = dbc.Col(
    [
        dcc.Markdown(
            """

            # The Quake Locater
            🧠 The Quake Predictor uses machine learning to predict where
            earthquakes of certain magnitudes will be.

            """
        ),
        dcc.Markdown(
            """
            👩🏾‍🔬 Using this tool, the location and risk of a high or low magnitude
            earthquake can be predicted, and investigated.

            """
        ),
        dcc.Link(dbc.Button('Start Predicting!', color='secondary'), href='/predictions')
    ],
    md=4,
)

gapminder = px.data.gapminder()

column2 = dbc.Col(
    [
        html.Img(src='/assets/quakelogo.png'),
        #dcc.Graph(figure=fig),
    ]
)

layout = dbc.Row([column1, column2])
