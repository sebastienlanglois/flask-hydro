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
from plotly import tools
import datetime
import dateutil.parser
import pytz
from collections import Iterable                            # < py38
import numpy as np
import time


url_base = '/dashboards/app1/'
mapbox_access_token = 'pk.eyJ1Ijoic2ViYXN0aWVubCIsImEiOiJjanhmOG0xOW0wd3huNDBvOXJtbnJ5N3A3In0.52bda0Z2xvkRi7cUEOc4yQ'



conn = sqlite3.connect('app/HYDRO-dev2.db')
df = pd.read_sql('SELECT * FROM META_STATION_BASSIN', conn)
cols = list(df.columns)
cols = cols[1:] + [cols[0]]
df = df[cols]


df2 = pd.read_sql("""SELECT s.NUMERO_STATION, m.* 
                     FROM META_TS m
                     INNER JOIN META_STATION_BASSIN s
                     on m.ID_POINT = s.ID_POINT
                     WHERE m.ID_POINT = 1000""", conn)
conn.close()

cols = list(df2.columns)
cols = [cols[0]] + cols[3:] + cols[1:2]
df2 = df2[cols]



gdf = gpd.GeoDataFrame.from_features(df['GEOM'].apply(lambda x: json.loads(x)['features'][0]))
lons= gdf['geometry'].apply(lambda x : x.centroid.x)
lats= gdf['geometry'].apply(lambda x : x.centroid.y)

# lons = df['LONGITUDE']
# lats = df['LATITUDE']
text=['Numéro de station: ' + str(c)+'<br>Nom de station: '+'{}'.format(r) +'<br>Superficie: '+'{}'.format(str(k))
      for c, r, k in zip(df.ID_POINT, df.NOM_STATION, df.SUPERFICIE)]



def flatten(items):
    """Yield items from any nested iterable; see Reference."""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x

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

layout2 = go.Layout(
    # title='Nuclear Waste Sites on Campus',
    autosize=False,
    legend_orientation="h",
    legend=dict(x=.05, y=1.07),
    hovermode='closest',
    showlegend=True,
    height=600,
    margin=go.layout.Margin(
        l=0,
        r=0,
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
        style= 'light'
    ),
)

fig = go.Figure(data=data, layout=layout2)


layout = html.Div([
    html.Div(
            [
                html.Div([
                    html.Button('Select/Deselect all...',
                                id='btn-5',
                                n_clicks_timestamp=0),
                    dash_table.DataTable(
                        id='bassins',
                        data=df.drop(columns=['GEOM']).to_dict('records'),
                        style_cell={'padding': '2px'},
                        style_as_list_view=True,
                        style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': 'rgb(248, 248, 248)'
                            }
                        ],
                        style_header={
                            'backgroundColor': '#071389',
                            'fontWeight': 'bold',
                            'color': '#fff',
                        },
                        filter_action="native",
                        sort_action="native",
                        sort_mode="multi",
                        row_selectable="multi",
                        page_action="none",
                        fixed_columns={ 'headers': True, 'data': 1 },
                        style_table={'maxWidth': '100%', 'maxHeight': '550px',
                                     },
                        columns=[{'id': c, 'name': c} for c in df.drop(columns='GEOM')],
                    )
                    ],
                    className="pretty_container four columns",
                    style={
                        'overflowX': 'scroll', 'overflowY': 'scroll',
                    }
                ),

                html.Div(
                    [dcc.Graph(id="map", figure=fig, ),

                     ],
                    className="pretty_container five columns",
                    style={'plot_bgcolor': '#f6f6f6'}
                ),


        ],
        className="row flex-display", style={'backgroundColor': '#f6f6f6'}
        ),
    html.Div(
        [
            html.Div([
                    html.Button('Select/Deselect all...',
                        id='btn-6',
                        n_clicks_timestamp=0),
                    dash_table.DataTable(
                        id="meta_ts",
                        data=df2.to_dict('records'),
                        filter_action="native",
                        sort_action="native",
                        sort_mode="multi",
                        row_selectable="multi",
                        style_cell={'padding': '5px'},
                        style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': 'rgb(248, 248, 248)'
                            }
                        ],
                        style_header={
                            'backgroundColor': '#071389',
                            'fontWeight': 'bold',
                            'color': '#fff',
                        },
                        page_action="none",
                        fixed_columns={ 'headers': True, 'data': 1 },
                        style_table={'maxWidth': '100%', 'maxHeight': '562px'},
                        columns=[{'id': c, 'name': c} for c in df2],
                ),
                    html.Div(id='test')

                ],
                className="pretty_container four columns",
                style={
                    'overflowX': 'scroll', 'overflowY': 'scroll',
                },

            ),
            html.Div(
               [dcc.Loading(id="loading-1",
                            children=[dcc.Store(id='storage'),
                                      dcc.Graph(id="graph",
                                                style={'Height': '500'},
                                                figure=[])],
                            type="circle",
                            style={"width": "100%",
                                   "margin-top": "230px",
                                   "margin-bottom": "240px"}),
                 html.Div(
                     children=dcc.RangeSlider(id='my-range-slider',),
                     id='slider-keeper',
                     style={"width": "100%",
                            "margin-top": "10px",
                            "margin-bottom": "40px"}),
                     html.Div([
                         html.Button('Importer...',
                                     id='btn-1',
                                     n_clicks_timestamp=0),
                         html.Button('Exporter...',
                                     id='btn-2',
                                     n_clicks_timestamp=0),
                         html.Button('Suivant...',
                                     id='btn-4',
                                     n_clicks_timestamp=0),
                         html.Button('Précédent...',
                                     id='btn-3',
                                     n_clicks_timestamp=0)
                     ], className=" row")
                ],

                className="pretty_container eight columns",
                style={'plot_bgcolor': '#f6f6f6'}
            ),

        ],
        className="row flex-display", style={'backgroundColor': '#f6f6f6'}
    )
    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column", 'backgroundColor': '#f6f6f6'})




def update_graph_fct(rows, selected_rows):
    dff = pd.DataFrame(rows)
    conn = sqlite3.connect('app/HYDRO-dev2.db')

    df1 = pd.read_sql('SELECT * FROM DON_TS WHERE ID_SERIE in ({})'
                      .format(", ".join(str(x) for x in dff['ID_SERIE'].iloc[selected_rows])),                      conn)
    conn.close()
    df_pivot = df1.pivot_table(values='VALUE', index='DATE', columns='ID_SERIE')
    df_pivot.index = pd.to_datetime(df_pivot.index)
    data = []

    for col in df_pivot.columns:
        print(col)
        trace = go.Scatter(
            x=df_pivot.index,
            y=df_pivot[col],
            mode='lines',
            name=col
        )
        data.append(trace)

    layout = dict(
        margin=go.layout.Margin(l=30, r=0, t=20, b=30),
        plot_bgcolor='#f6f6f6',
        paper_bgcolor='#f6f6f6',
        xaxis=dict(
            rangeslider=dict(
                visible=False
            ),
            type='date'
        )
    )

    return go.Figure(data=data, layout=layout)


def dropdown_graph_fct(fig, min_year, max_year):
    # x = pd.to_datetime(fig['data'][0]['x'])
    # y = (fig['data'][1]['y'])



    data = []
    for col in fig['data']:
        x = pd.to_datetime(col['x'])
        y = np.array(col['y'])
        # print(x > datetime.datetime(min_year,1,1))
        # print(x <= datetime.datetime(max_year,12,31))

        bool_array = [(x > datetime.datetime(min_year, 1, 1)) &
                      (x <= datetime.datetime(max_year, 12, 31))]
        print(bool_array[0])
        trace = go.Scatter(
            x=x[bool_array[0]],
            y=y[bool_array[0]],
            mode='lines',
            name=col['name']
        )
        data.append(trace)

    layout = dict(
        margin=go.layout.Margin(l=30, r=0, t=20, b=30),
        plot_bgcolor='#f6f6f6',
        paper_bgcolor='#f6f6f6',
        height=500,
        xaxis=dict(
            rangeslider=dict(
                visible=False
            ),
            type='date'
        )
    )

    return go.Figure(data=data, layout=layout)


def Add_Dash(server):
    app = Dash(server=server, url_base_pathname=url_base)
    apply_layout_with_auth(app, layout)


    # @app.callback(Output('storage', 'data'),
    #               [Input('btn-1', 'n_clicks_timestamp')])
    # def update_btn_storage(btn):
    #     value = btn / 1000
    #     return value


    # @app.callback([Output('my-range-slider', 'min'),
    #                Output('my-range-slider', 'max'),
    #                Output('my-range-slider', 'step'),
    #                Output('my-range-slider', 'value'),
    #                Output('my-range-slider', 'marks')],
    #               [Input('graph', 'figure')],
    #               [State('storage', 'data')])
    # def update_slider_example(fig, btn):
    #     print(fig)
    #     duration = time.mktime(datetime.datetime.now().timetuple()) - btn / 1000
    #     print(duration)
    #     if fig and duration <= 2:
    #         dates = pd.to_datetime(fig['data'][0]['x'])
    #         min1 = int(dates.year.min())
    #         max1 = int(dates.year.max())
    #         step = int(1),
    #         value1 = [int(dates.year.min()), int(dates.year.max())]
    #         if max1 - min1> 90:
    #             step_display = 5
    #         elif max1 - min1 > 70:
    #             step_display = 4
    #         elif max1 - min1 > 50:
    #             step_display = 3
    #         elif max1 - min1 > 30:
    #             step_display = 2
    #         else:
    #             step_display = 1
    #
    #         marks = {int(i): str(i) for i in np.arange(dates.year.min(),dates.year.max(), step_display)}
    #
    #         print(min1)
    #         print(max1)
    #         print(step)
    #         print(value1)
    #         print(marks)
    #         range_options = [min1, max1, step,  value1, marks]
    #     else:
    #         range_options = [1900, int(datetime.datetime.now().year), 1, [], []]
    #     return range_options


    @app.callback(
        [Output('storage', 'data'),
         Output('my-range-slider', 'min'),
         Output('my-range-slider', 'max'),
         Output('my-range-slider', 'step'),
         Output('my-range-slider', 'value'),
         Output('my-range-slider', 'marks'),
         ],
        [Input('btn-1', 'n_clicks_timestamp'),
         Input('btn-2', 'n_clicks_timestamp'),
         Input('btn-3', 'n_clicks_timestamp'),
         Input('btn-4', 'n_clicks_timestamp')],
        [State('my-range-slider', 'min'),
         State('my-range-slider', 'max'),
         State('my-range-slider', 'step'),
         State('my-range-slider', 'value'),
         State('my-range-slider', 'marks'),
         State('meta_ts', 'selected_rows'),
         State('meta_ts', 'data'),
         State('storage', 'data')])
    def update_figure(btn1, btn2, btn3, btn4, min1, max1, step, value1, marks,
                      selected_rows, rows, storage):
        time_now = time.mktime(datetime.datetime.now().timetuple())
        duration = time_now - btn1 / 1000
        duration2 = time_now - btn2 / 1000
        duration3 = time_now - btn3 / 1000
        duration4 = time_now - btn4 / 1000

        darr = np.array([duration, duration2, duration3, duration4])


        fig = storage

        if not min1:
            min1 = 1900
        if not max1:
            max1 = int(datetime.datetime.now().year)
        if not step:
            step = 1
        if not value1:
            value1 = []
        if not marks:
            marks = []

        if darr.argmin() ==2:
            if value1 and value1[0] > min1:
                value1[0] = value1[0] - 1
            if value1[1] - value1[0] >=1:
                value1[1] = value1[1] - 1

        if darr.argmin() ==3:
            if value1 and value1[1] < max1:
                value1[1] = value1[1] + 1
            if value1[1] - value1[0] >=1:
                value1[0] = value1[0] + 1

        if duration <=2:
            fig = []
            # case btm
            if selected_rows:
                fig = update_graph_fct(rows, selected_rows)
                dates = pd.to_datetime(fig['data'][0]['x'], utc = True)
                dates = dates.tz_convert('America/Montreal')
                min1 = int(dates.year.min())
                max1 = int(dates.year.max())
                step = int(1),
                value1 = [int(dates.year.min()), int(dates.year.max())]
                if max1 - min1 > 90:
                    step_display = 5
                elif max1 - min1 > 70:
                    step_display = 4
                elif max1 - min1 > 50:
                    step_display = 3
                elif max1 - min1 > 30:
                    step_display = 2
                else:
                    step_display = 1

                marks = {int(i): str(i) for i in np.arange(dates.year.min(), dates.year.max(), step_display)}


        # else:
        #     print(fig)
        #     # y = (fig['data'][0]['y'])
        #     # print(y)
        #     # if fig:
        #     #     fig = dropdown_graph_fct(fig, range_slider[0], range_slider[1])
        return [fig, min1, max1, step,  value1, marks]


    @app.callback(
        Output('graph', 'figure'),
        [Input('my-range-slider', 'value')],
        [State('storage', 'data')])
    def update_figure(value, fig):
        # y = (fig['data'][0]['y'])
        if fig:
            fig = dropdown_graph_fct(fig, value[0], value[1])
        else:
            fig = {
                'layout': {
                    'height': 500
                }
            }
        return fig


    @app.callback(Output('meta_ts', 'style_data_conditional'),
                  [Input('meta_ts', 'selected_rows'),
                   Input('meta_ts', 'derived_viewport_selected_rows')])
    def display_click_data(selected_rows, derived_viewport_selected_rows):
        style_data = []
        if selected_rows:
            style_data = [{
                "if": {"row_index": i},
                "backgroundColor": "#7BCFED",
                'color': 'white'
            } for i in derived_viewport_selected_rows]
        return style_data


    @app.callback(Output('meta_ts', 'data'),
                  [Input('bassins', 'selected_rows'),
                   Input('bassins', 'data')])
    def update_meta_ts(selected_rows, rows):
        df1 = pd.DataFrame()

        dff = pd.DataFrame(rows)
        if selected_rows:
            conn = sqlite3.connect('app/HYDRO-dev2.db')
            df1 = pd.read_sql("""SELECT s.NUMERO_STATION, m.*  
                                 FROM META_TS m
                                 INNER JOIN META_STATION_BASSIN s
                                 on m.ID_POINT = s.ID_POINT
                                 WHERE m.ID_POINT  in ({}) ORDER BY ID_POINT"""
                              .format(", ".join(str(x) for x in dff['ID_POINT'].iloc[selected_rows]))
                              , conn)

            conn.close()
            cols = list(df1.columns)
            cols = [cols[0]] + cols[3:] + cols[1:2]
            df1 = df1[cols]
        return df1.to_dict('records')


    @app.callback(Output('bassins', 'selected_rows'),
                  [Input('map', 'selectedData'),
                   Input('map', 'clickData'),
                   Input('btn-5', 'n_clicks_timestamp')],
                  [State('bassins', "derived_virtual_data"),
                   State('bassins', 'selected_rows')])
    def update_map_pts(selected_data, click_data, btn, derived_virtual_data, selected_rows):
        liste_points = []
        if selected_rows is None:
            selected_rows = []
        if selected_data:
            liste_points = [i['pointNumber'] for i in selected_data['points']]
        if click_data:
            liste_points.append(click_data['points'][0]['pointNumber'])
        print(selected_rows)
        for val in liste_points:
            if selected_rows and val in selected_rows:
                selected_rows.remove(val)
            else:
                selected_rows.append(val)
        duration = time.mktime(datetime.datetime.now().timetuple()) - btn / 1000
        print(selected_rows)
        if duration < 2:
            if derived_virtual_data is None:
                selected_rows =  []
            elif len(selected_rows) > 0:
                selected_rows = []
            else:
                selected_rows = [i for i in range(len(derived_virtual_data))]

        return selected_rows


    @app.callback(Output('map', 'figure'),
                  [Input('bassins', 'selected_rows')],
                  [State('map', 'figure'),
                   State('bassins', 'data')])
    def reset_map_pts(selected_rows, fig, rows):
        if selected_rows:
            df = pd.DataFrame(rows)
            print(selected_rows)
            # now the colors
            clrred = 'rgb(222,0,0)'
            clrgrn = 'rgb(20,0,255)'
            clrs = [clrred if x in selected_rows else clrgrn for x in range(df.shape[0])]
            size = [13 if x in selected_rows else 11 for x in range(df.shape[0])]
            fig['data'][0]['selectedpoints'] = None
            fig['data'][0]['marker'] = dict(color=clrs,
                                            size=size,
                                            opacity=0.7)
        return fig


    @app.callback(
        Output('meta_ts', "selected_rows"),
        [Input('btn-6', 'n_clicks_timestamp')],
        [State('meta_ts', "derived_viewport_indices"),
         State('meta_ts', "selected_rows")]
    )
    def select_all(n_clicks, derived_viewport_indices, selected_rows):
        print(derived_viewport_indices)
        if derived_viewport_indices is None:
            selected_rows =  []
            print('a')
        elif selected_rows and len(selected_rows) > 0:
            selected_rows = []
            print('b')
        else:
            selected_rows = derived_viewport_indices
            print(selected_rows)
        return selected_rows

    return app.server


