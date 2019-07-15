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
import os
import plotly.graph_objs as go
import pandas as pd

url_base = '/dashboards/app1/'

mapbox_access_token = 'pk.eyJ1Ijoic2ViYXN0aWVubCIsImEiOiJjanhmOG0xOW0wd3huNDBvOXJtbnJ5N3A3In0.52bda0Z2xvkRi7cUEOc4yQ'

df = pd.read_csv(
    'https://raw.githubusercontent.com/plotly/datasets/master/Nuclear%20Waste%20Sites%20on%20American%20Campuses.csv')
site_lat = df.lat
site_lon = df.lon
locations_name = df.text

data = [
    go.Scattermapbox(
        lat=site_lat,
        lon=site_lon,
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=17,
            color='rgb(255, 0, 0)',
            opacity=0.7
        ),
        text=locations_name,
        hoverinfo='text'
    ),
    go.Scattermapbox(
        lat=site_lat,
        lon=site_lon,
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=8,
            color='rgb(242, 177, 172)',
            opacity=0.7
        ),
        hoverinfo='none'
    )]

layout = go.Layout(
    title='Nuclear Waste Sites on Campus',
    autosize=True,
    hovermode='closest',
    showlegend=False,
    height=700,
    mapbox=go.layout.Mapbox(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=go.layout.mapbox.Center(
            lat=38,
            lon=-94
        ),
        pitch=0,
        zoom=3,
        style='light'
    ),
)

fig = go.Figure(data=data, layout=layout)


layout = html.Div([
    html.Div(
        [
            html.Div(
                [dcc.Graph(id="map", figure=fig)],
                className="pretty_container seven columns",
            ),
            html.Div(
                [dcc.Graph(id="individual_graph",
                           figure={
                               'data': [{
                                   'y': [1, 4, 3]
                               }],
                               'layout': {
                                   'height': 700
                               }
                           }
                           )],
                className="pretty_container five columns",
            ),
        ],
        className="row flex-display", style={"width": "100%"}
        )
    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},)

def Add_Dash(server):
    app = Dash(server=server, url_base_pathname=url_base)
    apply_layout_with_auth(app, layout)

    mapbox_access_token = "pk.eyJ1Ijoic2ViYXN0aWVubCIsImEiOiJjanhmOG0xOW0wd3huNDBvOXJtbnJ5N3A3In0.52bda0Z2xvkRi7cUEOc4yQ"


    if current_user and current_user.is_authenticated:
        @app.callback(
            [Output('map', 'figure'),
             ],
            [Input('get', 'n_clicks')])
        def update_map_callback(n_clicks):
            df = pd.read_csv(
                'https://raw.githubusercontent.com/plotly/datasets/master/2011_february_us_airport_traffic.csv')

            df['text'] = df['airport'] + '' + df['city'] + ', ' + df['state'] + '' + 'Arrivals: ' + df['cnt'].astype(
                str)

            scl = [[0, "rgb(5, 10, 172)"], [0.35, "rgb(40, 60, 190)"], [0.5, "rgb(70, 100, 245)"], \
                   [0.6, "rgb(90, 120, 245)"], [0.7, "rgb(106, 137, 247)"], [1, "rgb(220, 220, 220)"]]

            data = [go.Scattergeo(
                locationmode='ISO-3',
                lon=df['long'],
                lat=df['lat'],
                text=df['text'],
                mode='markers',
                marker=dict(
                    size=8,
                    opacity=0.8,
                    reversescale=True,
                    autocolorscale=False,
                    symbol='square',
                    line=dict(
                        width=1,
                        color='rgba(102, 102, 102)'
                    ),
                    colorscale=scl,
                    cmin=0,
                    color=df['cnt'],
                    cmax=df['cnt'].max(),
                    colorbar=dict(
                        title="Incoming flights<br>February 2011"
                    )
                ))]

            layout1 = dict(
                title='Most trafffor airport names)',
                geo=dict(
                    scope='world',
                    projection=dict(type='albers usa'),
                    showland=True,
                    landcolor="rgb(250, 250, 250)",
                    subunitcolor="rgb(217, 217, 217)",
                    countrycolor="rgb(217, 217, 217)",
                    countrywidth=0.5,
                    subunitwidth=0.5
                ),
            )

            return dict(data=data, layout=layout1)
    
    return app.server