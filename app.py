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
    "Aston Martin": "#225941",
    "Alpine": "#0090FF",
    "Alfa Romeo": "#900000",
    "AlphaTauri": "#2B4562",
    "Haas": "#FFFFFF",
    "Williams": "#005AFF",
    "Renault": "#FFF500",
    "Racing Point": "#F596C8",
    "Toro Rosso": "#469BFF",
    "Force India": "#F596C8",
    "Sauber": "#9B0000",
    "Lotus F1": "#FFB800",
    "Benetton": "#00A0DD",
    "Lotus": "#FFB800",
    "Jordan": "#F9D600",
    "Brawn": "#B8FD6E",
    "BAR": "#F0F0F0",
    "Toyota": "#E3001B",
    "Jaguar": "#004225",
    "Minardi": "#000000",
    "Super Aguri": "#FFFFFF",
    "Prost": "#0000FF",
    "Arrows": "#F59600",
    "Stewart": "#FFFFF0",
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

def get_color(name, theme="Dark"):
    color = TEAM_COLORS.get(name, "#AAAAAA")
    # Dynamic override for Haas/Williams visibility
    if name == "Haas":
        return "#7B7B7B" if theme == "Light" else "#FFFFFF"
    if name == "Williams" and theme == "Dark":
        return "#005AFF" # Default
    if name == "Williams" and theme == "Light":
        return "#005AFF" # Ensure visibility
    return color

def hex_to_rgba(hex_color, opacity):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return f"rgba({r},{g},{b},{opacity})"
    return hex_color

# -- Data Loading --
# Data source: https://www.kaggle.com/datasets/melissamonfared/formula-1/data
DATA_DIR = "data"
print("Loading data...")

races = pd.read_csv(f"{DATA_DIR}/races.csv")
results = pd.read_csv(f"{DATA_DIR}/results.csv")
constructors = pd.read_csv(f"{DATA_DIR}/constructors.csv")
constructor_standings = pd.read_csv(f"{DATA_DIR}/constructor_standings.csv")
drivers = pd.read_csv(f"{DATA_DIR}/drivers.csv")

print("Loading lap times...")
lap_times = pd.read_csv(f"{DATA_DIR}/lap_times.csv", usecols=["raceId", "driverId", "lap", "position", "milliseconds"])
RACES_WITH_LAPS = set(lap_times["raceId"].unique())
print("Data loaded.")

# -- Data Pre-processing --
races_meta = races[["raceId", "year", "name", "date"]].sort_values(["year", "date"])
races_small = races[["raceId", "year", "date"]].copy()
races_small["era"] = races_small["year"].apply(map_era)
constructors_small = constructors[["constructorId", "name"]].copy()
drivers_meta = drivers[["driverId", "code", "surname"]].copy()
drivers_meta["label"] = drivers_meta["code"].replace(r"\\N", pd.NA, regex=True).fillna(drivers_meta["surname"])

# -- Generate Static Dominance Figure (Compressed for Side Panel) --
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
    ERA_ORDER[3]: {"start": 2022, "end": 2024},
}

# Calculate Midpoints and Widths for Bar Charts
era_plot_data = {}
for e, t in ERA_TIMING.items():
    duration = t["end"] - t["start"] + 1
    mid = t["start"] + (duration / 2) - 0.5 # Center aligned
    era_plot_data[e] = {"mid": mid, "width": duration * 0.95} # 0.95 to leave tiny gap

def create_dominance_figure(theme="Dark"):
    print(f"Generating Dominance Figure ({theme})...")
    is_light = (theme == "Light")
    template = "plotly_white" if is_light else "plotly_dark"
    text_color = "#111" if is_light else "#FFF"
    
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

    # Layout: 3 Rows
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Points Trajectory", "Dom. Concentration (HHI)", "Era Wins Share"),
        vertical_spacing=0.08,
        shared_xaxes=True, 
        specs=[[{"type": "scatter"}], [{"type": "bar"}], [{"type": "bar"}]]
    )

    # 1. Points
    for team in focus_teams:
        df = points_focus[points_focus["name"] == team]
        if not df.empty:
            c = get_color(team, theme)
            fig.add_trace(go.Scatter(x=df["year"], y=df["points"], mode="lines", name=team, line=dict(color=c, width=2), showlegend=False), row=1, col=1)
    
    # 2. HHI
    fig.add_trace(go.Bar(
        x=hhi_df["mid"], y=hhi_df["hhi"], width=hhi_df["width"],
        marker_color="purple", name="HHI", showlegend=False,
        hovertemplate="<b>%{text}</b><br>HHI: %{y:.2f}<extra></extra>",
        text=[ERA_SHORT[e] for e in hhi_df["era"]] 
    ), row=2, col=1)

    # 3. Total Wins
    for team in TEAM_COLORS.keys():
        team_wins = wins_per_era[wins_per_era["name"] == team].copy()
        if not team_wins.empty:
            team_wins["mid"] = team_wins["era"].apply(lambda e: era_plot_data[e]["mid"])
            team_wins["width"] = team_wins["era"].apply(lambda e: era_plot_data[e]["width"])
            c = get_color(team, theme)
            
            fig.add_trace(go.Bar(
                x=team_wins["mid"], y=team_wins["total_wins"], width=team_wins["width"],
                name=team, marker_color=c, showlegend=True
            ), row=3, col=1)
    fig.update_layout(barmode='stack')

    # Add Vertical Separation Lines
    boundaries = [2005.5, 2013.5, 2021.5]
    line_c = "#BBB" if is_light else "#555"
    for b in boundaries:
        fig.add_shape(
            type="line", x0=b, x1=b, y0=0, y1=1.10, 
            xref="x", yref="paper",
            line=dict(color=line_c, width=1, dash="dash")
        )

    # Add Era Headings
    for era in ERA_ORDER:
        mid = era_plot_data[era]["mid"]
        txt = ERA_SHORT[era]
        fig.add_annotation(
            x=mid, y=1.12, 
            text=f"<b>{txt}</b>",
            xref="x", yref="paper",
            showarrow=False,
            font=dict(size=12, color=text_color)
        )

    fig.update_layout(
        template=template, 
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=120, l=40, r=40, b=50), # Increased top margin for Era Headers
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
        font=dict(size=10, color=text_color),
        xaxis3=dict(title="Year", tickmode="linear", dtick=5),
        yaxis=dict(title="Points"),
        yaxis2=dict(title="HHI"),
        yaxis3=dict(title="Wins")
    )
    return fig

# -- Layout (Split Screen) --
app.layout = html.Div(
    id="main-container",
    style={
        "fontFamily": "Segoe UI, sans-serif", 
        "backgroundColor": "#121212", 
        "color": "#eee",
        "height": "100vh", "width": "100vw", "display": "flex", "flexDirection": "row", "overflow": "hidden"
    },
    children=[
        # LEFT PANEL: Context (40%)
        html.Div(
            style={"width": "40%", "height": "100%", "borderRight": "1px solid #333", "padding": "10px", "display": "flex", "flexDirection": "column", "boxSizing": "border-box"},
            children=[
                # Header with Theme Toggle
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "10px"},
                    children=[
                        html.H3("Historical Context", style={"margin": "0", "color": "inherit"}),
                        dcc.RadioItems(
                            id="theme-switch",
                            options=[
                                {"label": "🌙 Dark", "value": "Dark"},
                                {"label": "☀️ Light", "value": "Light"},
                            ],
                            value="Dark",
                            inline=True,
                            style={"fontSize": "0.9em"}
                        )
                    ]
                ),
                dcc.Graph(id="dominance-graph", style={"flex": "1"}, config={"displayModeBar": False})
            ]
        ),
        
        # RIGHT PANEL: Interaction (60%)
        html.Div(
            style={"width": "60%", "height": "100%", "padding": "10px", "display": "flex", "flexDirection": "column", "boxSizing": "border-box"},
            children=[
                # Top Row: Controls + Lap Chart
                html.Div(
                    style={"height": "50%", "display": "flex", "flexDirection": "column"},
                    children=[
                        html.H3("Race Analysis (Lap-by-Lap)", style={"margin": "0 0 5px 0", "textAlign": "center"}),
                         # Controls
                        html.Div(
                            style={"display": "flex", "gap": "15px", "justifyContent": "center", "marginBottom": "5px"},
                            children=[
                                html.Div([
                                    html.Label("Year", style={"fontSize": "0.8em"}),
                                    dcc.Dropdown(
                                        id="year-dropdown",
                                        options=[
                                            {"label": y, "value": y} 
                                            for y in sorted(races_meta[races_meta["raceId"].isin(RACES_WITH_LAPS)]["year"].unique(), reverse=True)
                                        ],
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
                        dcc.Loading(
                            type="cube",
                            children=[
                                dcc.Graph(id="lap-chart", style={"height": "40vh"}, config={"responsive": True})
                            ]
                        )
                    ]
                ),
                
                # Bottom Row: Consistency
                html.Div(
                    style={"height": "50%", "borderTop": "1px solid #333", "paddingTop": "10px"},
                    children=[
                        html.H4("Top 5 Drivers: Lap Time Consistency", style={"margin": "0 0 5px 0", "textAlign": "center"}),
                        dcc.Loading(
                            type="cube",
                            children=[
                                dcc.Graph(id="violin-graph", style={"height": "40vh"}, config={"responsive": True})
                            ]
                        )
                    ]
                )
            ]
        )
    ]
)

# -- Callbacks --

# 1. Update Theme (Global Styles & Dominance Chart)
@app.callback(
    [Output("main-container", "style"), Output("dominance-graph", "figure")],
    [Input("theme-switch", "value")]
)
def update_theme(theme):
    is_light = (theme == "Light")
    bg_color = "#FFFFFF" if is_light else "#121212"
    text_color = "#111111" if is_light else "#EEEEEE"
    
    style = {
        "fontFamily": "Segoe UI, sans-serif", 
        "backgroundColor": bg_color, 
        "color": text_color,
        "height": "100vh", "width": "100vw", "display": "flex", "flexDirection": "row", "overflow": "hidden"
    }
    
    fig = create_dominance_figure(theme)
    return style, fig

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
    [Output("lap-chart", "figure"), Output("violin-graph", "figure")],
    [Input("race-dropdown", "value"), Input("theme-switch", "value")]
)
def update_chart(race_id, theme):
    if not race_id: return go.Figure(), go.Figure()
    
    is_light = (theme == "Light")
    template = "plotly_white" if is_light else "plotly_dark"
    winner_color = "#000000" if is_light else "#FFFFFF"

    # 1. Get Lap Data
    df_race = lap_times[lap_times["raceId"] == race_id].copy()
    
    # 2. Get Team Data for this Race (Driver -> Team)
    race_results = results[results["raceId"] == race_id][["driverId", "constructorId", "positionOrder"]]
    race_results = race_results.merge(constructors_small, on="constructorId", how="left")
    
    # Merge Metadata
    df_race = df_race.merge(drivers_meta, on="driverId", how="left")
    df_race = df_race.merge(race_results, on="driverId", how="left")
    df_race = df_race.sort_values(["driverId", "lap"])
    
    last_lap = df_race["lap"].max()
    final_pos = df_race[df_race["lap"] == last_lap].sort_values("position")
    winner_id = final_pos.iloc[0]["driverId"] if not final_pos.empty else None
    
    # Identify "Lead Driver" per team (Highest finisher)
    final_stats = df_race[df_race["lap"] == last_lap][["driverId", "name", "position"]].sort_values("position")
    
    best_in_team = {}
    for _, row in final_stats.iterrows():
        team = row["name"]
        if team not in best_in_team:
            best_in_team[team] = row["driverId"]
            
    race_info = races_meta[races_meta["raceId"] == race_id].iloc[0]
    
    # SORT DRIVERS BY FINAL POSITION (For Tooltip Order)
    driver_sort_map = race_results.set_index("driverId")["positionOrder"].to_dict()
    
    # Get unique drivers present in LAP DATA
    driver_ids = sorted(df_race["driverId"].unique(), key=lambda x: driver_sort_map.get(x, 999))

    # -- 1. Initial Data (FULL RACE) --
    fig = go.Figure()
    for did in driver_ids:
        d = df_race[df_race["driverId"] == did]
        is_winner = (did == winner_id)
        team_name = d["name"].iloc[0] if "name" in d.columns else "Unknown"
        color = get_color(team_name, theme)
        
        # Line Style Logic
        is_best = (did == best_in_team.get(team_name))
        dash_style = "solid" if is_best else "dot"
        
        # Winner gets THICK WINNER COLOR (White/Black) LINE
        # Teammate logic applies to color lines
        line_color = winner_color if is_winner else color
        opacity = 1.0 if is_winner else 0.8
        width = 4 if is_winner else 2
        final_dash = "solid" if is_winner else dash_style

        fig.add_trace(go.Scatter(
            x=d["lap"], y=d["position"],
            mode="lines", 
            name=d["label"].iloc[0],
            line=dict(width=width, color=line_color, dash=final_dash),
            opacity=opacity,
            hovertemplate=f"<b>{d['label'].iloc[0]}</b> ({team_name})<br>L%{{x}} P%{{y}}<extra></extra>"
        ))

    # -- 2. Frames (All Laps 1..Last) --
    frames = []
    frame_names = [str(l) for l in range(1, last_lap + 1)]
    
    for l in range(1, last_lap + 1):
        frame_data = []
        for did in driver_ids:
            d = df_race[df_race["driverId"] == did]
            d_lap = d[d["lap"] <= l]
            
            # Re-apply logic
            is_winner = (did == winner_id)
            team_name = d["name"].iloc[0] if "name" in d.columns else "Unknown"
            color = get_color(team_name, theme)
            is_best = (did == best_in_team.get(team_name))
            dash_style = "solid" if is_best else "dot"
            
            line_color = winner_color if is_winner else color
            width = 4 if is_winner else 2
            final_dash = "solid" if is_winner else dash_style
            
            frame_data.append(go.Scatter(
                x=d_lap["lap"], y=d_lap["position"],
                mode="lines",
                line=dict(width=width, color=line_color, dash=final_dash)
            ))
        
        frames.append(go.Frame(data=frame_data, name=str(l)))
    
    fig.frames = frames

    # -- 3. Animation Controls --
    fig.update_layout(
        title=dict(text=f"{race_info['year']} {race_info['name']}", y=0.98),
        template=template,
        yaxis=dict(autorange="reversed", title="Position", fixedrange=True, range=[22, 0]), 
        xaxis=dict(title="Lap", fixedrange=True, range=[1, last_lap]),
        hovermode="closest",
        margin=dict(t=150, b=50, l=40, r=20), # Increased Top for Legend
        legend=dict(orientation="h", y=1.25, x=0.5, xanchor="center"), # Legend at Top
        
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=1.1, x=1.05, xanchor="right", yanchor="top",
            pad=dict(t=0, r=10),
            buttons=[
                dict(
                    label="▶", 
                    method="animate", 
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
            active=last_lap - 1,
            yanchor="top", xanchor="left",
            currentvalue=dict(font=dict(size=15), prefix="Lap: ", visible=True, xanchor="right"),
            transition=dict(duration=300, easing="cubic-in-out"),
            pad=dict(b=10, t=50, l=20, r=5), # Reduced margins to expand slider
            len=1.0, x=0, y=0,
            steps=[dict(label=str(l), method="animate", args=[[str(l)], dict(mode="immediate", frame=dict(duration=100, redraw=False), transition=dict(duration=0))]) for l in range(1, last_lap + 1)]
        )]
    )

    # -- 4. Violin Chart (Top 5 Consistency) --
    fig_violin = go.Figure()
    
    top_5_ids = final_pos.head(5)["driverId"].tolist()
    
    for did in top_5_ids:
        d = df_race[df_race["driverId"] == did].copy()
        d["seconds"] = pd.to_numeric(d["milliseconds"], errors="coerce") / 1000.0
        
        # Filter outliers: Use IQR method
        q1 = d["seconds"].quantile(0.25)
        q3 = d["seconds"].quantile(0.75)
        iqr = q3 - q1
        upper_fence = q3 + (1.5 * iqr)
        
        if not pd.isna(upper_fence):
            d = d[d["seconds"] < upper_fence]
        
        driver_label = d["label"].iloc[0]
        # Team Color Logic
        team_name = d["name"].iloc[0] if "name" in d.columns else "Unknown"
        color = get_color(team_name, theme)
        
        fig_violin.add_trace(go.Violin(
            x=d["label"],
            y=d["seconds"],
            name=driver_label,
            box_visible=True,
            meanline_visible=True,
            points="all",
            jitter=0.05,
            pointpos=-1.8,
            line_color=color, # Apply Team Color
            opacity=0.8
        ))

    fig_violin.update_layout(
        template=template,
        yaxis=dict(title="Lap Time (s)", showgrid=True),
        xaxis=dict(title="Driver"),
        margin=dict(t=20, b=40, l=40, r=20),
        showlegend=False
    )

    return fig, fig_violin

if __name__ == "__main__":
    app.run(debug=True, port=8050)
