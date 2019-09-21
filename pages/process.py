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

            # Information
            Hi, my name is Emmett Boudreau, and I'm a Data Scientist. This app
            uses a Random Forest Classifier model with an accuracy of about 90 percent
            to predict what state an
            earthquake will take place in based on the specifications of the
            said Earthquake. For more information on the development of this
            model, and the development of this application, I have a post on
            Medium outlining the whole thing.

            """
        ),
        dcc.Link(dbc.Button('Open on medium', color='secondary'), href='/predictions'),
        dcc.Markdown(
            """
            Additionally, it is open-source! You can view the source for this
            project on my github!

            """
        ),
        dcc.Link(dbc.Button('Open source', color='secondary'), href='/predictions'),
        dcc.Markdown(
            """
            If you would like to see more of my Data Science
            projects, you can also visit my portfolio

            """
        ),
        dcc.Link(dbc.Button('Visit my Portfolio', color='secondary'), href='http://emmettboudreau.com/')
    ],
)

layout = dbc.Row([column1])
