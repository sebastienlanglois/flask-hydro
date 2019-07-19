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
import sqlite3
import geopandas as gpd
from geopandas import GeoDataFrame
import json
from shapely.geometry import mapping
import dash_table

url_base = '/dashboards/app1/'
mapbox_access_token = 'pk.eyJ1Ijoic2ViYXN0aWVubCIsImEiOiJjanhmOG0xOW0wd3huNDBvOXJtbnJ5N3A3In0.52bda0Z2xvkRi7cUEOc4yQ'


conn = sqlite3.connect('app/HYDRO-dev2.db')
df = pd.read_sql('SELECT * FROM META_STATION_BASSIN', conn)
df2 = pd.read_sql('SELECT * FROM META_TS', conn)
gdf = gpd.GeoDataFrame.from_features(df['GEOM'].apply(lambda x: json.loads(x)['features'][0]))

lons= gdf['geometry'].apply(lambda x : x.centroid.x)
lats= gdf['geometry'].apply(lambda x : x.centroid.y)

text=['Numéro de station: ' + str(c)+'<br>Nom de station: '+'{}'.format(r) +'<br>Superficie: '+'{}'.format(str(k))
      for c, r, k in zip(df.ID_POINT, df.NOM_STATION, df.SUPERFICIE)]

data = [
    go.Scattermapbox(
        lat=lats,
        lon=lons,
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=11,
            color='rgb(20, 0, 255)',
            opacity=0.7
        ),
        text=text,
        hoverinfo='text',
        legendgroup="Stations/Bassins de débits",
        showlegend=True,
        name="Stations/Bassins de débits"
    ),
]

layout = go.Layout(
    # title='Nuclear Waste Sites on Campus',
    autosize=True,
    legend_orientation="h",
    legend=dict(x=.05, y=1.07),
    hovermode='closest',
    showlegend=True,
    height=600,
    margin=go.layout.Margin(
        l=0,
        r=25,
        b=40,
        t=0,
        pad=0
    ),
    paper_bgcolor='#f6f6f6',
    mapbox=go.layout.Mapbox(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=go.layout.mapbox.Center(
        lat=52.4638,
        lon=-73.98),
        pitch=0,
        zoom=4,
    ),
)

fig = go.Figure(data=data, layout=layout)


layout = html.Div([
    html.Div(
            [
            html.Div(
                [dcc.Graph(id="map", figure=fig, )],
                className="pretty_container seven columns",
                style={"width": "100%", 'plot_bgcolor': '#f6f6f6'}
            ),
                html.Div(
                    [dash_table.DataTable(
                        data=df.drop(columns='GEOM').to_dict('records'),
                        style_cell={'padding': '5px'},
                        style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': 'rgb(248, 248, 248)'
                            }
                        ],
                        style_header={
                            'backgroundColor': 'rgb(230, 230, 230)',
                            'fontWeight': 'bold'
                        },
                        filter_action="native",
                        sort_action="native",
                        sort_mode="multi",
                        row_selectable="multi",
                        page_action="none",
                        style_table={'overflowX': 'scroll', 'overflowY': 'scroll',
                                     'maxWidth': '100%','maxHeight': '562px'},
                        columns=[{'id': c, 'name': c} for c in df.drop(columns='GEOM')],
                    )
                    ],
                    className="pretty_container five columns",
                )
                ,
        ],
        className="row flex-display", style={'backgroundColor': '#f6f6f6'}
        ),
    html.Div(
        [
            html.Div(
                [dcc.Graph(id="graph", figure=fig, )],
                className="pretty_container seven columns",
                style={"width": "100%", 'plot_bgcolor': '#f6f6f6'}
            ),
            html.Div(
                [dash_table.DataTable(
                    data=df2.to_dict('records'),
                    style_cell={'padding': '5px'},
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(248, 248, 248)'
                        }
                    ],
                    style_header={
                        'backgroundColor': 'rgb(230, 230, 230)',
                        'fontWeight': 'bold'
                    },
                    filter_action="native",
                    sort_action="native",
                    sort_mode="multi",
                    row_selectable="multi",
                    page_action="none",
                    style_table={'overflowX': 'scroll', 'overflowY': 'scroll',
                                 'maxWidth': '100%', 'maxHeight': '562px'},
                    columns=[{'id': c, 'name': c} for c in df2],
                )
                ],
                className="pretty_container five columns",
            )
            ,
        ],
        className="row flex-display", style={'backgroundColor': '#f6f6f6'}
    )
    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column", 'backgroundColor': '#f6f6f6'})

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