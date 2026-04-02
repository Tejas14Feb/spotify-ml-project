"""
dashboard/app.py
----------------
Interactive Plotly Dash app for the Spotify Hit Predictor.

Run:  python dashboard/app.py
Open: http://localhost:8050
"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib
import json
import os

# ── Load assets ──────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

df      = pd.read_csv(os.path.join(BASE, 'data', 'spotify_clean.csv'))
model   = joblib.load(os.path.join(BASE, 'models', 'best_model.pkl'))
le      = joblib.load(os.path.join(BASE, 'models', 'genre_encoder.pkl'))

with open(os.path.join(BASE, 'models', 'features.json')) as f:
    FEATURES = json.load(f)

GENRES   = sorted(df['track_genre'].dropna().unique())
AUDIO_FT = ['danceability', 'energy', 'valence', 'loudness',
            'acousticness', 'speechiness', 'instrumentalness', 'liveness', 'tempo']

# ── App layout ────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title='Spotify Hit Predictor')

app.layout = html.Div([

    # Header
    html.Div([
        html.H1('Spotify Hit Predictor', style={'margin': 0, 'fontSize': 28, 'fontWeight': 500}),
        html.P('End-to-end ML pipeline · XGBoost · 114k tracks',
               style={'color': '#888', 'margin': '4px 0 0', 'fontSize': 14}),
    ], style={'padding': '24px 32px 0', 'borderBottom': '1px solid #eee', 'paddingBottom': 16}),

    # KPI row
    html.Div(id='kpi-row', style={
        'display': 'flex', 'gap': 16, 'padding': '20px 32px',
        'borderBottom': '1px solid #eee'
    }),

    # Main content
    html.Div([

        # Left panel — predictor
        html.Div([
            html.H3('Will this song be a hit?', style={'fontSize': 16, 'fontWeight': 500, 'marginBottom': 16}),

            html.Label('Genre', style={'fontSize': 13, 'color': '#666'}),
            dcc.Dropdown(
                id='genre-input',
                options=[{'label': g, 'value': g} for g in GENRES],
                value='pop',
                clearable=False,
                style={'marginBottom': 12}
            ),

            *[html.Div([
                html.Label(f'{ft.replace("_", " ").title()}',
                           style={'fontSize': 13, 'color': '#666', 'marginBottom': 4}),
                dcc.Slider(
                    id=f'slider-{ft}',
                    min=round(df[ft].min(), 2),
                    max=round(df[ft].max(), 2),
                    step=round((df[ft].max() - df[ft].min()) / 100, 3),
                    value=round(df[ft].median(), 3),
                    marks=None,
                    tooltip={'placement': 'bottom', 'always_visible': True}
                )
            ], style={'marginBottom': 14}) for ft in AUDIO_FT],

            html.Label('Explicit', style={'fontSize': 13, 'color': '#666'}),
            dcc.RadioItems(
                id='explicit-input',
                options=[{'label': ' Yes', 'value': 1}, {'label': ' No', 'value': 0}],
                value=0,
                inline=True,
                style={'marginBottom': 20, 'fontSize': 14}
            ),

            html.Div(id='prediction-output', style={
                'textAlign': 'center', 'padding': 20,
                'borderRadius': 12, 'background': '#f8f8f8',
                'border': '1px solid #eee'
            }),

        ], style={'width': '340px', 'flexShrink': 0, 'padding': '20px 24px',
                  'borderRight': '1px solid #eee', 'overflowY': 'auto'}),

        # Right panel — charts
        html.Div([

            html.Div([
                html.Div([
                    html.H3('Feature distributions by popularity', style={'fontSize': 15, 'fontWeight': 500, 'margin': 0}),
                    dcc.Dropdown(
                        id='feature-select',
                        options=[{'label': f, 'value': f} for f in AUDIO_FT],
                        value='danceability',
                        clearable=False,
                        style={'width': 200}
                    ),
                ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginBottom': 8}),
                dcc.Graph(id='dist-chart', style={'height': 280}),
            ], style={'marginBottom': 16}),

            html.Div([
                html.H3('Top genres by hit rate', style={'fontSize': 15, 'fontWeight': 500, 'marginBottom': 8}),
                dcc.Graph(id='genre-chart', style={'height': 300}),
            ]),

        ], style={'flex': 1, 'padding': '20px 24px', 'overflowY': 'auto'}),

    ], style={'display': 'flex', 'height': 'calc(100vh - 130px)', 'overflow': 'hidden'}),

], style={'fontFamily': 'system-ui, sans-serif', 'height': '100vh', 'overflow': 'hidden'})


# ── Callbacks ──────────────────────────────────────────────────────────────────

@callback(Output('kpi-row', 'children'), Input('genre-input', 'value'))
def update_kpis(_):
    total   = len(df)
    hits    = df['is_hit'].sum()
    avg_pop = df['popularity'].mean()
    genres  = df['track_genre'].nunique()

    cards = [
        ('Total tracks', f'{total:,}'),
        ('Hit songs (top 25%)', f'{hits:,}'),
        ('Avg popularity', f'{avg_pop:.1f} / 100'),
        ('Genres covered', str(genres)),
    ]
    return [html.Div([
        html.Div(label, style={'fontSize': 12, 'color': '#888', 'marginBottom': 4}),
        html.Div(value, style={'fontSize': 22, 'fontWeight': 500}),
    ], style={
        'background': '#f8f8f8', 'borderRadius': 8, 'padding': '12px 16px',
        'minWidth': 140
    }) for label, value in cards]


@callback(
    Output('prediction-output', 'children'),
    Output('prediction-output', 'style'),
    [Input('genre-input', 'value'),
     Input('explicit-input', 'value')] +
    [Input(f'slider-{ft}', 'value') for ft in AUDIO_FT]
)
def predict(genre, explicit, *audio_vals):
    try:
        genre_enc = le.transform([genre])[0]
    except Exception:
        genre_enc = 0

    duration_s = df['duration_s'].median()
    row = dict(zip(AUDIO_FT, audio_vals))
    row['explicit']      = explicit
    row['duration_s']    = duration_s
    row['genre_encoded'] = genre_enc

    X = pd.DataFrame([row])[FEATURES]
    prob = model.predict_proba(X)[0][1]
    is_hit = prob >= 0.5

    color  = '#E1F5EE' if is_hit else '#F1EFE8'
    border = '#1D9E75' if is_hit else '#B4B2A9'
    label  = 'HIT' if is_hit else 'NOT A HIT'
    emoji  = '' if is_hit else ''

    style = {
        'textAlign': 'center', 'padding': 20,
        'borderRadius': 12, 'background': color,
        'border': f'1px solid {border}'
    }
    children = [
        html.Div(emoji, style={'fontSize': 32, 'marginBottom': 8}),
        html.Div(label, style={'fontSize': 20, 'fontWeight': 500, 'color': '#1D9E75' if is_hit else '#888'}),
        html.Div(f'Confidence: {prob:.1%}', style={'fontSize': 14, 'color': '#666', 'marginTop': 6}),
    ]
    return children, style


@callback(Output('dist-chart', 'figure'), Input('feature-select', 'value'))
def update_dist(feature):
    fig = px.violin(
        df.assign(label=df['is_hit'].map({0: 'Not a hit', 1: 'Hit'})),
        x='label', y=feature, color='label',
        color_discrete_map={'Hit': '#1D9E75', 'Not a hit': '#888780'},
        box=True, points=False
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis_title='',
        yaxis_title=feature
    )
    return fig


@callback(Output('genre-chart', 'figure'), Input('genre-input', 'value'))
def update_genre_chart(_):
    genre_stats = (
        df.groupby('track_genre')
          .agg(hit_rate=('is_hit', 'mean'), count=('is_hit', 'size'))
          .query('count > 100')
          .nlargest(15, 'hit_rate')
          .reset_index()
    )
    genre_stats['hit_rate_pct'] = (genre_stats['hit_rate'] * 100).round(1)

    fig = px.bar(
        genre_stats, x='hit_rate_pct', y='track_genre',
        orientation='h',
        color='hit_rate_pct',
        color_continuous_scale=[[0, '#E1F5EE'], [1, '#0F6E56']],
        text='hit_rate_pct'
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(
        margin=dict(l=0, r=40, t=10, b=0),
        showlegend=False, coloraxis_showscale=False,
        plot_bgcolor='white', paper_bgcolor='white',
        xaxis_title='Hit rate (%)', yaxis_title='',
        yaxis={'categoryorder': 'total ascending'}
    )
    return fig


if __name__ == '__main__':
    print('\nSpotify Hit Predictor Dashboard')
    print('Open: http://localhost:8050\n')
    app.run(debug=True, port=8050)
