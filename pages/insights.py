import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd

from app import app
df = pd.read_csv("https://opendata.socrata.com/api/views/77jn-2ym9/rows.csv")
df = df.dropna()
df = df.drop(columns="updated")
df = df.drop(columns="id")
df = df.drop(columns="Location 1")
df = df.rename(columns={"magType": "Magnitude Type"})
df['State'] = df['place'].str.split(',').str[1]
df = df.drop(columns="place")
import plotly.graph_objects as go

import numpy as np
import plotly.express as px
tips = px.data.tips()
fig = px.histogram(df, x="State", color="Magnitude Type",title="U.S. EarthQuakes by State and Magnitude")
column1 = dbc.Col(
    [
        dcc.Markdown(
            """

            ## Understanding Earthquakes
            To understand earthquakes, we need to understand magnitude.
            magnitude is a measurement of vibration from the surfact of the
            Earth's crust, usually taken on a Richter Scale. There are several
            different classifications of earthquake magnitude, which can be
            found [here.](https://earthquake.usgs.gov/learn/topics/mag-intensity/magnitude-types.php) \
            Natural earthquakes occur near fault lines, where tectonic plates
            converge against once another, shifting and sliding, causing the
            Earth's crust to vibrate.
            """

        ),
        dcc.Markdown(
            """
            ## Fault-lines
            The more, and greater the fault lines in a particular region
            contribute to the likelihood of an Earthquake occurring. For example,
            Earthquakes in the United States can frequently be found in locations
            like California and Hawaii, which are located along fault-lines.
            """

        ),
        dcc.Markdown(
            """
            ## An initial question:
            If the most major faultline in the United States is in California,
            there should be strong potential of an algorithm guessing California
            everytime, and getting a high accuracy (majority-class baseline).
            However strong in theory this hypothesis is, suprisingly, it proved
            to not be true. With a baseline accuracy score of around 51 percent,
            and a model score of 91 percent, it's clear that a model is needed
            to properly predict this target.
            """

        ),
    ],
    md=4,
)


column2 = dbc.Col(
    [
        dcc.Graph(figure=fig),
        html.Img(src='/assets/tectonic_map.png'),
    ]
)

layout = dbc.Row([column1, column2])
