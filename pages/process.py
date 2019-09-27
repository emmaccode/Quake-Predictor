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
            uses an XGBoost Classifier model with an accuracy of about 90 percent
            to predict what state an
            earthquake will take place in based on the specifications of the
            said Earthquake. For more information on the development of this
            model, and the development of this application, I have a post on
            Medium outlining the project.

            """
        ),
        html.A("Medium Post", href='https://plot.ly', target="_blank",className=".myButton"),
        dcc.Markdown(
            """
            Additionally, it is open-source! You can view the source for this
            project on my github!

            """
        ),
        html.A("Source on github", href='https://github.com/emmettgb/Quake-Predictor', target="_blank",className=".myButton"),
        dcc.Markdown(
            """
            If you would like to see more of my Data Science
            projects, you can also visit my portfolio

            """
        ),
        html.A("My Portfolio", href='https://emmettboudreau.com', target="_blank",className=".myButton"),
    ],
)

layout = dbc.Row([column1])
