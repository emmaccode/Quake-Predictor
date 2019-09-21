import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
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
        dcc.Markdown(
            """

            ## Predict the magnitude
            The model's train accuracy usually sits around 90 percent accuracy.
            The predictor will allow you to find out information about the
            likelihood of being in danger.
            """
        ),
    ],
    md=4,
)
import pandas as pd
import numpy as np
import scipy.stats as scs
import itertools
import operator

"""# Reading and cleaning"""
from sklearn.metrics import accuracy_score
df = pd.read_csv("https://opendata.socrata.com/api/views/77jn-2ym9/rows.csv")
df = df.dropna()
df = df.drop(columns="updated")
df = df.drop(columns="id")
df = df.drop(columns="Location 1")
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

df = df.replace("CA","California")

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
pipeline = make_pipeline(
    ce.OrdinalEncoder(),
    IterativeImputer(),
    RandomForestClassifier(n_estimators=3000, random_state=32, n_jobs=-2)
)
pipeline.fit(Xtrain,ytrain)
ypr = pipeline.predict(Xtest)
accuracy_rf = accuracy_score(ytest,ypr)
print(ypr)
# Hey that's pretty high I would say.
# That's suprisingly balanced with the histogram in mind.
#    Especially with the baseline being 59, all things considered, I think
#    This'll work.
# I'm interested to see one more thing though....

df.head(2)

typeeffect = pd.DataFrame({'time': ["10/28/2015 12:42:31 PM", "10/28/2015 12:42:31 PM"],
                           'depth': [0.85, 0.85], 'mag': [1.50,1.50],
                          'magType': ["md","md"], 'nst': [8.0,8.0],
                           'gap': [144.0,144.0], 'dmin': [.02065,.02065],
                           'rms': [0.04,0.04], 'net': ["nc","nc"],
                           'type': ['earthquake',"quarry blast"]})

# The only difference between these 2 sample X features is the type, i'm
# Interested to see if the prediction will be different based on type alone.
testpr = pipeline.predict(typeeffect)

typeeffect["PR"] = testpr

typeeffect.head(3)
import plotly.graph_objects as go
import numpy as np


# Create figure
fig = go.Figure()

column2 = dbc.Col(
    [

    ]
)

layout = dbc.Row([column1, column2])
