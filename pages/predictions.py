import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import scikit-learn
from dash.dependencies import Input, Output

from app import app
column1 = dbc.Col(
    [
        dcc.Markdown(
            """

            ## Predict the location of an Earthquake
            The model's train accuracy usually sits around 90 percent accuracy.
            The predictor will allow you to find out information about the
            likelihood of being in danger

            """
        ),
    ],
    md=4,
)
import pandas as pd
import numpy as np
import plotly.graph_objects as go
"""# Reading and cleaning"""
df = pd.read_csv("https://opendata.socrata.com/api/views/77jn-2ym9/rows.csv")
df = df.dropna()
df = df.drop(columns="updated")
df = df.drop(columns="id")
df = df.drop(columns="Location 1")
df = df.replace("CA","California")
df = df.drop(columns="time")
df['State'] = df['place'].str.split(',').str[1]
df = df.drop(columns="place")
"""# Selection
Selecting the target, my features, getting some importances, exploring and making tweaks to the data, and finally Train/Test splitting my data.
"""
from sklearn.model_selection import train_test_split
test, train = train_test_split(df)

target = "State"
import plotly.express as px

"""Obviously California is the world leader here, Interestingly, some of the state data for California is stored as CA"""


# The problem I could face with something like this is the potential of the
#       outcome almost always being California

# Next I want to see how useless the "type" column is:
# fig2 = px.histogram(df, x="type")
# That is interesting, and there is potential for more outcomes other than
  # California, so this could be beneficial, and possibly a dash categorical.

target = "State"
Xtrain = train.drop(columns=target)
ytrain = train[target]
Xtest = test.drop(columns=target)
ytest = test[target]
"""# Model"""
import category_encoders as ce
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import xgboost as xgb
# Pipeline:
pipeline = make_pipeline(
    ce.OrdinalEncoder(),
    IterativeImputer(),
    xgb.XGBClassifier(
                n_estimators=1200,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.7,
                colsample_bytree=0.7,
                objective='reg:linear')
)
# Fit the pipeline
pipeline.fit(Xtrain,ytrain)
# Empty DataFrame:
#<---- Dash Components ---->
# -Magnitude Type Dropdown
QuakeTdropdown = dcc.Dropdown(
    options=[
        {'label': 'md', 'value': 'md'},
        {'label': 'ml', 'value': 'ml'},
        {'label': 'mw', 'value': 'mw'},
        {'label': 'Md', 'value': 'Md'},
        {'label': 'mb', 'value': 'mb'},
        {'label': 'mh', 'value': 'mh'},
        {'label': 'Ml', 'value': 'Ml'},
        {'label': 'Mb', 'value': 'Mb'},
        {'label': 'mlg', 'value': 'mlg'}
    ],
    id='magtypedrp',
    value='ml'
)
# -Magnitude Range Slider
magslider = dcc.Slider(
    min=-1.0,
    max=4.2,
    step=0.2,
    id='magslide',
    value=.6,
    marks={
        -1.0: '-1',
        0: '0',
        2: '2',
        4: '4'
    },
)
# -Depth Range Slider
depthslider = dcc.Slider(
    min=-3.5,
    max=211.0,
    step=10,
    id='depthslide',
    value=9,
    marks={
        0: '0',
        25: '25',
        50: '50',
        75: '75',
        100: '100',
        125: '125',
        150: '150',
        175: '175',
        200: '200'
    },
)
# -Nst slider
nstslider = dcc.Slider(
    min=-0,
    max=166,
    step=10,
    id='nstslide',
    value=50,
    marks={
        0: '0',
        25: '25',
        50: '50',
        75: '75',
        100: '100'
    },
)
gapslider = dcc.Slider(
    min=12.0,
    max=365.4,
    step=15,
    id='gapslide',
    value=14,
    marks={
        0: '0',
        25: '25',
        50: '50',
        75: '75',
        100: '100',
        125: '125',
        150: '150',
        175: '175',
        200: '200',
        225: '225',
        250: '250',
        275: '275',
        300: '300',
        325: '325',
        350: '350',
        365: '365'
    },
)
dminslider = dcc.Slider(
    min=-.00000004,
    max=2.5,
    step=0.1,
    id='dminslide',
    value=.2,
    marks={
        .5: '.5',
        1: '1',
        1.5: '1.5',
        2: '2',
        2.5: '2.5'
    },
)
rmsslider = dcc.Slider(
    min=0,
    max=1.5,
    step=0.2,
    id='rmsslide',
    value=.6,
    marks={
        0: '0',
        .5: '.5',
        1: '1',
        1.5: '1.5'
    },
)
netdropdown = dcc.Dropdown(
    options=[
        {'label': 'nc', 'value': 'nc'},
        {'label': 'ci', 'value': 'ci'},
        {'label': 'hv', 'value': 'hv'},
        {'label': 'uw', 'value': 'uw'},
        {'label': 'pr', 'value': 'pr'},
        {'label': 'nn', 'value': 'nn'},
        {'label': 'mb', 'value': 'mb'},
        {'label': 'uu', 'value': 'uu'},
        {'label': 'nm', 'value': 'nm'},
        {'label': 'ismpkansas', 'value': 'ismpkansas'},
        {'label': 'ld', 'value': 'ld'},
        {'label': 'se', 'value': 'se'}
    ],
    id='netdrop',
    value='ld'
)
typedropdown = dcc.Dropdown(
    options=[
        {'label': 'earthquake', 'value': 'earthquake'},
        {'label': 'explosion', 'value': 'explosion'},
        {'label': 'quarry blast', 'value': 'quarry blast'},
        {'label': 'rockslide', 'value': 'rockslide'},
        {'label': 'chemical explosion', 'value': 'chemical explosion'},
    ],
    id='typedrop',
    value='earthquake'
)
# 'md' 'ml' 'Md' 'mw' 'mb' 'mh' 'Ml' 'Mb' 'mlg'
column2 = dbc.Col(
    [
    html.A("Magnitude Type", href='https://earthquake.usgs.gov/learn/topics/mag-intensity/magnitude-types.php', target="_blank",className=".myButton"),
    QuakeTdropdown,
    html.A("Vibration Cause"),
    typedropdown,
    html.A("Net"),
    netdropdown,
    html.A("Magnitude Scale",href='https://en.wikipedia.org/wiki/Seismic_magnitude_scales'),
    magslider,
    html.A("Quake Depth"),
    depthslider,
    html.A("NST"),
    nstslider,
    html.A("Quake gap"),
    gapslider,
    html.A("Dmin"),
    dminslider,
    html.A("RMS"),
    rmsslider,
    html.H1(id='predictiondiv'),
    ]
)
# Callback function---->
@app.callback(
    Output('predictiondiv', 'children'),
    [Input('depthslide', 'value'),
    Input('magslide', 'value'),
    Input('magtypedrp', 'value'),
    Input('nstslide', 'value'),
    Input('gapslide', 'value'),
    Input('dminslide', 'value'),
    Input('rmsslide', 'value'),
    Input('netdrop', 'value'),
    Input('typedrop', 'value'),
    ]
)
#The actual predictor-
def predict(depth, mag, magType, nst, gap, dmin, rms, net, type):
    predx = pd.DataFrame({'depth':[depth],'mag':[mag],'magType':[magType],'nst':[nst],'gap': [gap],'dmin': [dmin], 'rms': [rms], 'net': [net],'type': [type]})
    y_pred = pipeline.predict(predx)
    y_pred = str(y_pred)
    y_pred = y_pred.strip("[]")
    y_pred = y_pred.strip('""')
    return("Your predicted state is: ",y_pred)
# And finally the layout ---->
layout = dbc.Row([column1, column2])
