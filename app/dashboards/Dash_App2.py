# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 10:39:33 2018

@author: jimmybow
"""
from dash import Dash
from dash.dependencies import Input, State, Output
from .Dash_fun import apply_layout_with_auth
import dash_core_components as dcc
import dash_html_components as html
from flask_login import current_user
url_base = '/dash/app2/'

layout = html.Div([
    dcc.Input(id='my-id', value='initial value', type='text'),
    html.Div(id='my-div')
])



def Add_Dash(server):
    app = Dash(server=server, url_base_pathname=url_base)
    apply_layout_with_auth(app, layout)

    if current_user and current_user.is_authenticated:
        @app.callback(
            Output(component_id='my-div', component_property='children'),
            [Input(component_id='my-id', component_property='value')]
        )
        def update_output_div(input_value):
            return 'You\'ve entered "{}"'.format(input_value)

    return app.server