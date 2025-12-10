import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -- App Initialization --
app = dash.Dash(__name__, title="F1 Analytics")
server = app.server

# -- Constants & Helpers --
ERA_ORDER = [
    "High-downforce NA (1987–2005)",
    "V8 frozen-engine (2006–2013)",
    "Hybrid V6 turbo (2014–2021)",
    "Ground-effect hybrid (2022–present)",
]

TEAM_COLORS = {
    "Ferrari": "#DC0000",
    "McLaren": "#FF8700",
    "Red Bull": "#0600EF",
    "Mercedes": "#00D2BE",
    "Williams": "#005AFF",
    "Renault": "#FFF500",
    "Benetton": "#00A0DD",
    "Lotus": "#FFB800",
    "Jordan": "#F9D600",
    "Brawn": "#B8FD6E",
    "Toro Rosso": "#0032FF",
}

def map_era(year):
    if 1987 <= year <= 2005:
        return ERA_ORDER[0]
    elif 2006 <= year <= 2013:
        return ERA_ORDER[1]
    elif 2014 <= year <= 2021:
        return ERA_ORDER[2]
    elif year >= 2022:
        return ERA_ORDER[3]
    return "Pre-era"

def get_color(name):
    return TEAM_COLORS.get(name, "#AAAAAA")

def hex_to_rgba(hex_color, opacity):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return f"rgba({r},{g},{b},{opacity})"
    return hex_color

# -- Data Loading --
DATA_DIR = "data"
print("Loading data...")

races = pd.read_csv(f"{DATA_DIR}/races.csv")
results = pd.read_csv(f"{DATA_DIR}/results.csv")
constructors = pd.read_csv(f"{DATA_DIR}/constructors.csv")
constructor_standings = pd.read_csv(f"{DATA_DIR}/constructor_standings.csv")
drivers = pd.read_csv(f"{DATA_DIR}/drivers.csv")

print("Loading lap times...")
lap_times = pd.read_csv(f"{DATA_DIR}/lap_times.csv", usecols=["raceId", "driverId", "lap", "position"])
RACES_WITH_LAPS = set(lap_times["raceId"].unique())
print("Data loaded.")

# -- Data Pre-processing --

races_meta = races[["raceId", "year", "name", "date"]].sort_values(["year", "date"])
races_small = races[["raceId", "year", "date"]].copy()
races_small["era"] = races_small["year"].apply(map_era)
constructors_small = constructors[["constructorId", "name"]].copy()
drivers_meta = drivers[["driverId", "code", "surname"]].copy()
drivers_meta["label"] = drivers_meta["code"].fillna(drivers_meta["surname"])

# -- Generate Static Dominance Figure (Compressed for Side Panel) --
# Short names for cleaner display in side panel
ERA_SHORT = {
    "High-downforce NA (1987–2005)": "NA",
    "V8 frozen-engine (2006–2013)": "V8",
    "Hybrid V6 turbo (2014–2021)": "Hybrid",
    "Ground-effect hybrid (2022–present)": "Ground",
}

# Helper: Era Durations for Temporal Alignment
ERA_TIMING = {
    ERA_ORDER[0]: {"start": 1987, "end": 2005},
    ERA_ORDER[1]: {"start": 2006, "end": 2013},
    ERA_ORDER[2]: {"start": 2014, "end": 2021},
    ERA_ORDER[3]: {"start": 2022, "end": 2024}, # Assuming 2024 is current data end
}

# Calculate Midpoints and Widths for Bar Charts
era_plot_data = {}
for e, t in ERA_TIMING.items():
    duration = t["end"] - t["start"] + 1
    mid = t["start"] + (duration / 2) - 0.5 # Center aligned
    era_plot_data[e] = {"mid": mid, "width": duration * 0.95} # 0.95 to leave tiny gap

def create_dominance_figure():
    print("Generating Dominance Figure...")
    
    races_era = races_small[races_small["era"].isin(ERA_ORDER)].copy()
    results_races = results.merge(races_era[["raceId", "year", "era"]], on="raceId", how="inner")
    results_races["is_win"] = (results_races["positionOrder"] == 1).astype(int)
    
    wins_per_season = results_races.groupby(["year", "era", "constructorId"], as_index=False)["is_win"].sum()
    wins_per_era = wins_per_season.groupby(["era", "constructorId"], as_index=False)["is_win"].sum().rename(columns={"is_win": "total_wins"})
    wins_per_era = wins_per_era.merge(constructors_small, on="constructorId", how="left")
    
    # Points Time Series
    standings_join = constructor_standings.merge(races_era[["raceId", "year", "era"]], on="raceId", how="inner")
    points_per_season = standings_join.groupby(["year", "era", "constructorId"], as_index=False)["points"].max()
    points_per_season = points_per_season.merge(constructors_small, on="constructorId", how="left")
    
    focus_teams = ["Ferrari", "McLaren", "Williams", "Mercedes", "Red Bull", "Benetton", "Renault"]
    points_focus = points_per_season[points_per_season["name"].isin(focus_teams)].sort_values(["year", "name"])

    # HHI
    hhi_data = []
    for era in ERA_ORDER:
        df = wins_per_era[wins_per_era["era"] == era].copy()
        total = df["total_wins"].sum()
        hhi = ((df["total_wins"]/total)**2).sum() if total > 0 else 0
        hhi_data.append({
            "era": era, 
            "hhi": hhi,
            "mid": era_plot_data[era]["mid"],
            "width": era_plot_data[era]["width"]
        })
    hhi_df = pd.DataFrame(hhi_data)

    # Layout: 3 Rows, Shared X-Axis (Time)
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Points Trajectory", "Dom. Concentration (HHI)", "Era Wins Share"),
        vertical_spacing=0.08,
        shared_xaxes=True, # Key change
        specs=[[{"type": "scatter"}], [{"type": "bar"}], [{"type": "bar"}]]
    )

    # 1. Points
    for team in focus_teams:
        df = points_focus[points_focus["name"] == team]
        if not df.empty:
            fig.add_trace(go.Scatter(x=df["year"], y=df["points"], mode="lines", name=team, line=dict(color=get_color(team), width=2), showlegend=False), row=1, col=1)
    
    # 2. HHI (Time Aligned)
    fig.add_trace(go.Bar(
        x=hhi_df["mid"], y=hhi_df["hhi"], width=hhi_df["width"],
        marker_color="purple", name="HHI", showlegend=False,
        hovertemplate="<b>%{text}</b><br>HHI: %{y:.2f}<extra></extra>",
        text=[ERA_SHORT[e] for e in hhi_df["era"]] # Show short name on hover
    ), row=2, col=1)

    # 3. Total Wins (Stacked Bar Time Aligned)
    for team in TEAM_COLORS.keys():
        team_wins = wins_per_era[wins_per_era["name"] == team].copy()
        if not team_wins.empty:
            # Map plotting data
            team_wins["mid"] = team_wins["era"].apply(lambda e: era_plot_data[e]["mid"])
            team_wins["width"] = team_wins["era"].apply(lambda e: era_plot_data[e]["width"])
            
            fig.add_trace(go.Bar(
                x=team_wins["mid"], y=team_wins["total_wins"], width=team_wins["width"],
                name=team, marker_color=get_color(team), showlegend=True
            ), row=3, col=1)
    fig.update_layout(barmode='stack')

    # Add Vertical Separation Lines (Era Boundaries) - Spanning entire figure
    boundaries = [2005.5, 2013.5, 2021.5]
    for b in boundaries:
        fig.add_shape(
            type="line", x0=b, x1=b, y0=0, y1=1.10, # Extended up to headings
            xref="x", yref="paper",
            line=dict(color="#555", width=1, dash="dash")
        )

    # Add Era Headings at the TOP
    for era in ERA_ORDER:
        mid = era_plot_data[era]["mid"]
        txt = ERA_SHORT[era]
        fig.add_annotation(
            x=mid, y=1.12, # Moved up to clear the subplot title
            text=f"<b>{txt}</b>",
            xref="x", yref="paper",
            showarrow=False,
            font=dict(size=12, color="#fff")
        )

    fig.update_layout(
        template="plotly_dark", 
        margin=dict(t=80, l=40, r=40, b=50), # Increased top margin
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
        font=dict(size=10),
        xaxis3=dict(title="Year", tickmode="linear", dtick=5)
    )
    return fig

dominance_fig = create_dominance_figure()

# -- Layout (Split Screen) --
app.layout = html.Div(
    style={
        "fontFamily": "Segoe UI, sans-serif", "backgroundColor": "#121212", "color": "#eee",
        "height": "100vh", "width": "100vw", "display": "flex", "flexDirection": "row", "overflow": "hidden"
    },
    children=[
        # LEFT PANEL: Context (40%)
        html.Div(
            style={"width": "40%", "height": "100%", "borderRight": "1px solid #333", "padding": "10px", "display": "flex", "flexDirection": "column", "boxSizing": "border-box"},
            children=[
                html.H3("Historical Context", style={"margin": "10px 0 10px 0", "textAlign": "center", "color": "#aaa"}),
                dcc.Graph(figure=dominance_fig, style={"flex": "1"}, config={"displayModeBar": False})
            ]
        ),
        
        # RIGHT PANEL: Interaction (60%)
        html.Div(
            style={"width": "60%", "height": "100%", "padding": "10px", "display": "flex", "flexDirection": "column"},
            children=[
                html.H3("Race Analysis (Lap-by-Lap)", style={"margin": "0 0 15px 0", "textAlign": "center"}),
                
                # Controls
                html.Div(
                    style={"display": "flex", "gap": "15px", "justifyContent": "center", "marginBottom": "10px"},
                    children=[
                        html.Div([
                            html.Label("Year", style={"fontSize": "0.8em"}),
                            dcc.Dropdown(
                                id="year-dropdown",
                                options=[{"label": y, "value": y} for y in sorted(races_meta["year"].unique(), reverse=True)],
                                value=2021, clearable=False, style={"width": "100px", "color": "#000"}
                            )
                        ]),
                        html.Div([
                            html.Label("Event", style={"fontSize": "0.8em"}),
                            dcc.Dropdown(
                                id="race-dropdown", value=None, clearable=False, style={"width": "250px", "color": "#000"}
                            )
                        ]),
                    ]
                ),
                
                # Main Chart
                dcc.Loading(
                    type="cube",
                    children=[
                        dcc.Graph(id="lap-chart", style={"height": "85vh"}, config={"responsive": True})
                    ]
                )
            ]
        )
    ]
)

# -- Callbacks --
@app.callback(
    [Output("race-dropdown", "options"), Output("race-dropdown", "value")],
    [Input("year-dropdown", "value")]
)
def update_races(selected_year):
    # Filter races that match year AND have lap data
    df = races_meta[
        (races_meta["year"] == selected_year) & 
        (races_meta["raceId"].isin(RACES_WITH_LAPS))
    ]
    
    if df.empty:
        return [], None

    options = [{"label": f"{row['name']} ({row['date']})", "value": row["raceId"]} for _, row in df.iterrows()]
    val = options[-1]["value"] if options else None
    return options, val

@app.callback(
    Output("lap-chart", "figure"),
    [Input("race-dropdown", "value")]
)
def update_chart(race_id):
    if not race_id: return go.Figure()

    df_race = lap_times[lap_times["raceId"] == race_id].copy()
    df_race = df_race.merge(drivers_meta, on="driverId", how="left").sort_values(["driverId", "lap"])
    
    last_lap = df_race["lap"].max()
    final_pos = df_race[df_race["lap"] == last_lap].sort_values("position")
    winner_id = final_pos.iloc[0]["driverId"] if not final_pos.empty else None
    
    race_info = races_meta[races_meta["raceId"] == race_id].iloc[0]
    driver_ids = df_race["driverId"].unique()

    # -- 1. Initial Data (FULL RACE) --
    fig = go.Figure()
    for did in driver_ids:
        d = df_race[df_race["driverId"] == did]
        is_winner = (did == winner_id)
        
        fig.add_trace(go.Scatter(
            x=d["lap"], y=d["position"], # Full data
            mode="lines", 
            name=d["label"].iloc[0],
            line=dict(width=3 if is_winner else 1, color="#FFF" if is_winner else None),
            opacity=1.0 if is_winner else 0.4,
            hovertemplate=f"<b>{d['label'].iloc[0]}</b><br>L%{{x}} P%{{y}}<extra></extra>"
        ))

    # -- 2. Frames (All Laps 1..Last) --
    frames = []
    
    # Create valid frame names list for the Play button
    frame_names = [str(l) for l in range(1, last_lap + 1)]
    
    for l in range(1, last_lap + 1):
        frame_data = []
        for did in driver_ids:
            d = df_race[df_race["driverId"] == did]
            d_lap = d[d["lap"] <= l]
            frame_data.append(go.Scatter(x=d_lap["lap"], y=d_lap["position"]))
        
        frames.append(go.Frame(data=frame_data, name=str(l)))
    
    fig.frames = frames

    # -- 3. Animation Controls --
    fig.update_layout(
        title=f"{race_info['year']} {race_info['name']}",
        template="plotly_dark",
        yaxis=dict(autorange="reversed", title="Position", fixedrange=True, range=[22, 0]), 
        xaxis=dict(title="Lap", fixedrange=True, range=[1, last_lap]),
        hovermode="x unified",
        margin=dict(t=40, b=160, l=40, r=20), # Maximized bottom margin
        legend=dict(orientation="h", y=-0.5, x=0.5, xanchor="center"), # Pushed way down
        
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=1.15, x=1.1, xanchor="right", yanchor="top",
            pad=dict(t=0, r=10),
            buttons=[
                dict(
                    label="▶", 
                    method="animate", 
                    # Play ALL frames from start
                    args=[frame_names, dict(frame=dict(duration=100, redraw=False), fromcurrent=True, mode="immediate")]
                ),
                dict(
                    label="⏸", 
                    method="animate", 
                    args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))]
                )
            ]
        )],
        
        sliders=[dict(
            active=last_lap - 1, # Set slider to END
            yanchor="top", xanchor="left",
            currentvalue=dict(font=dict(size=15), prefix="Lap: ", visible=True, xanchor="right"),
            transition=dict(duration=300, easing="cubic-in-out"),
            pad=dict(b=10, t=50),
            len=0.9, x=0.1, y=0,
            steps=[dict(label=str(l), method="animate", args=[[str(l)], dict(mode="immediate", frame=dict(duration=100, redraw=False), transition=dict(duration=0))]) for l in range(1, last_lap + 1)]
        )]
    )
    return fig

if __name__ == "__main__":
    app.run(debug=True, port=8050)
