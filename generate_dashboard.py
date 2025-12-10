import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# ----------------- Configuration -----------------

DATA_DIR = "data"
OUTPUT_FILE = "f1_era_dominance_dashboard_novel.html"

pio.templates.default = "plotly_dark"

# Approximate team colours for recognisable teams
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
DEFAULT_COLOR = "#AAAAAA"

ERA_ORDER = [
    "High-downforce NA (1987–2005)",
    "V8 frozen-engine (2006–2013)",
    "Hybrid V6 turbo (2014–2021)",
    "Ground-effect hybrid (2022–present)",
]


# ----------------- Helpers -----------------

def map_era(year: int) -> str:
    if 1987 <= year <= 2005:
        return ERA_ORDER[0]
    elif 2006 <= year <= 2013:
        return ERA_ORDER[1]
    elif 2014 <= year <= 2021:
        return ERA_ORDER[2]
    elif year >= 2022:
        return ERA_ORDER[3]
    else:
        return "Pre-era"


def get_color(team_name: str) -> str:
    return TEAM_COLORS.get(team_name, DEFAULT_COLOR)


def compute_hhi(shares: pd.Series) -> float:
    shares = shares.fillna(0)
    return float((shares ** 2).sum())


# ----------------- Main -----------------

def main():
    print("Loading data...")

    try:
        races = pd.read_csv(f"{DATA_DIR}/races.csv")
        results = pd.read_csv(f"{DATA_DIR}/results.csv")
        constructors = pd.read_csv(f"{DATA_DIR}/constructors.csv")
        constructor_standings = pd.read_csv(f"{DATA_DIR}/constructor_standings.csv")
    except FileNotFoundError:
        print("Error: Data files not found in 'data/' directory.")
        return

    # ---- Pre‑processing ----

    races_small = races[["raceId", "year", "date"]].copy()
    constructors_small = constructors[["constructorId", "name"]].copy()

    races_small["era"] = races_small["year"].apply(map_era)
    races_small = races_small[races_small["era"].isin(ERA_ORDER)].copy()

    # Merge race metadata into results
    results_races = results.merge(
        races_small[["raceId", "year", "era"]],
        on="raceId",
        how="inner",
    )
    results_races["is_win"] = (results_races["positionOrder"] == 1).astype(int)

    # 1. Wins per constructor per era ----------------------------------------

    wins_per_season = (
        results_races.groupby(["year", "era", "constructorId"], as_index=False)["is_win"]
        .sum()
        .rename(columns={"is_win": "season_wins"})
    )

    wins_per_era = (
        wins_per_season.groupby(["era", "constructorId"], as_index=False)["season_wins"]
        .sum()
        .rename(columns={"season_wins": "total_wins"})
    )

    wins_per_era = wins_per_era.merge(constructors_small, on="constructorId", how="left")

    # Top 10 teams per era
    top_teams_per_era = {}
    for era in ERA_ORDER:
        df_era = (
            wins_per_era[wins_per_era["era"] == era]
            .sort_values("total_wins", ascending=False)
            .head(10)
        )
        top_teams_per_era[era] = df_era

    # 2. Season‑end points per constructor (using max points per season) -----

    standings_join = constructor_standings.merge(
        races_small[["raceId", "year", "era"]],
        on="raceId",
        how="inner",
    )

    points_per_season = (
        standings_join.groupby(["year", "era", "constructorId"], as_index=False)["points"]
        .max()
    )
    points_per_season = points_per_season.merge(
        constructors_small, on="constructorId", how="left"
    )

    focus_teams = ["Ferrari", "McLaren", "Williams", "Mercedes", "Red Bull", "Benetton", "Renault"]
    points_focus = (
        points_per_season[points_per_season["name"].isin(focus_teams)]
        .sort_values(["year", "name"])
        .copy()
    )

    # 3. Era‑level dominance (HHI on wins share) -----------------------------

    hhi_rows = []
    for era in ERA_ORDER:
        df_era = wins_per_era[wins_per_era["era"] == era].copy()
        total = df_era["total_wins"].sum()
        if total > 0:
            df_era["share"] = df_era["total_wins"] / total
            hhi = compute_hhi(df_era["share"])
        else:
            hhi = 0.0
        hhi_rows.append({"era": era, "hhi_wins": hhi, "total_wins_era": total})

    hhi_df = pd.DataFrame(hhi_rows)

    # 4. Data for sunburst: hierarchy Era -> Constructor ---------------------

    sunburst_df = wins_per_era[wins_per_era["total_wins"] > 0].copy()
    # To avoid overcrowding, group constructors with very few wins into "Other"
    MIN_WINS_FOR_NODE = 3
    sunburst_df["constructor_label"] = sunburst_df["name"]
    sunburst_df.loc[sunburst_df["total_wins"] < MIN_WINS_FOR_NODE, "constructor_label"] = "Other"

    # Aggregate again after grouping "Other"
    sunburst_agg = (
        sunburst_df.groupby(["era", "constructor_label"], as_index=False)["total_wins"]
        .sum()
        .rename(columns={"constructor_label": "constructor_group"})
    )

    # 5. Data for Sankey: Era -> Top constructor in that era -----------------

    sankey_rows = []
    for era in ERA_ORDER:
        df_era = wins_per_era[wins_per_era["era"] == era]
        if df_era.empty:
            continue
        top_row = df_era.sort_values("total_wins", ascending=False).iloc[0]
        sankey_rows.append(
            {
                "era": era,
                "constructor": top_row["name"],
                "wins": int(top_row["total_wins"]),
            }
        )
    sankey_df = pd.DataFrame(sankey_rows)

    # ----------------- Plotting -----------------

    print("Generating dashboard...")

    # Layout grid:
    # Row 1: 4x bar (wins by era)
    # Row 2: 2x1 (left = sunburst, right = sankey)
    # Row 3: 1x (time series)
    # Row 4: 1x (HHI)

    fig = make_subplots(
        rows=4,
        cols=4,
        specs=[
            [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
            [
                {"type": "domain", "colspan": 2},
                None,
                {"type": "domain", "colspan": 2},
                None,
            ],
            [{"colspan": 4, "type": "scatter"}, None, None, None],
            [{"colspan": 4, "type": "bar"}, None, None, None],
        ],
        subplot_titles=(
            "Wins: " + ERA_ORDER[0],
            "Wins: " + ERA_ORDER[1],
            "Wins: " + ERA_ORDER[2],
            "Wins: " + ERA_ORDER[3],
            "Era → constructor hierarchy (wins, aggregated)",
            "Era → top constructor flow (wins)",
            "Constructor points trajectory (selected teams)",
            "Era dominance concentration (HHI of wins)",
        ),
        vertical_spacing=0.10,
        row_heights=[0.26, 0.26, 0.30, 0.18],
    )

    # --- Row 1: bar charts per era (top 10) ---

    for i, era in enumerate(ERA_ORDER):
        df = top_teams_per_era[era].sort_values("total_wins", ascending=True)
        fig.add_trace(
            go.Bar(
                y=df["name"],
                x=df["total_wins"],
                orientation="h",
                name=era,
                showlegend=False,
                marker_color=[get_color(n) for n in df["name"]],
                hovertemplate="<b>%{y}</b><br>Wins in era: %{x}<extra></extra>",
            ),
            row=1,
            col=i + 1,
        )
        fig.update_yaxes(title_text="", row=1, col=i + 1)
        fig.update_xaxes(title_text="Wins", row=1, col=i + 1)

    # --- Row 2 left: sunburst Era -> constructor_group ---

    labels = []
    parents = []
    values = []
    colors = []

    # Root node (optional, but helps structure)
    root_label = "F1 constructors by era"
    labels.append(root_label)
    parents.append("")
    values.append(0)
    colors.append("rgba(0,0,0,0)")  # transparent

    # Era nodes
    for era in ERA_ORDER:
        labels.append(era)
        parents.append(root_label)
        era_total = int(
            sunburst_agg[sunburst_agg["era"] == era]["total_wins"].sum()
        )
        values.append(era_total)
        colors.append("rgba(150,150,150,0.4)")

    # Constructor group nodes
    for _, row in sunburst_agg.iterrows():
        labels.append(row["constructor_group"])
        parents.append(row["era"])
        values.append(int(row["total_wins"]))
        colors.append(get_color(row["constructor_group"]))

    sunburst_trace = go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        hovertemplate="<b>%{label}</b><br>Wins: %{value}<extra></extra>",
    )

    fig.add_trace(sunburst_trace, row=2, col=1)

    # --- Row 2 right: Sankey Era -> top constructor ---

    if not sankey_df.empty:
        # Build node list: eras + unique constructors
        sankey_labels = list(ERA_ORDER)
        constructor_nodes = sorted(sankey_df["constructor"].unique().tolist())
        sankey_labels.extend(constructor_nodes)

        label_to_index = {lab: i for i, lab in enumerate(sankey_labels)}

        sources = []
        targets = []
        values_link = []
        link_colors = []

        # Helper to convert hex to rgba for transparency
        def hex_to_rgba(hex_color, opacity):
            hex_color = hex_color.lstrip('#')
            if len(hex_color) == 6:
                r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                return f"rgba({r},{g},{b},{opacity})"
            return hex_color # Fallback

        for _, row_s in sankey_df.iterrows():
            era = row_s["era"]
            cons = row_s["constructor"]
            wins_val = row_s["wins"]
            sources.append(label_to_index[era])
            targets.append(label_to_index[cons])
            values_link.append(wins_val)
            # Use rgba for transparency support in Plotly
            base_color = get_color(cons)
            link_colors.append(hex_to_rgba(base_color, 0.5))

        node_colors = []
        for lab in sankey_labels:
            if lab in constructor_nodes:
                node_colors.append(get_color(lab))
            else:
                node_colors.append("rgba(200,200,200,0.8)")

        sankey_trace = go.Sankey(
            arrangement="snap",
            node=dict(
                pad=10,
                thickness=15,
                label=sankey_labels,
                color=node_colors,
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values_link,
                color=link_colors,
                hovertemplate="<b>%{source.label} → %{target.label}</b><br>Wins: %{value}<extra></extra>",
            ),
        )

        fig.add_trace(sankey_trace, row=2, col=3)

    # --- Row 3: time series lines with era shading ---

    for team in focus_teams:
        df_team = points_focus[points_focus["name"] == team]
        if df_team.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=df_team["year"],
                y=df_team["points"],
                mode="lines+markers",
                name=team,
                line=dict(color=get_color(team), width=3),
                marker=dict(size=6),
                hovertemplate="<b>"
                + team
                + "</b><br>Year: %{x}<br>Points: %{y}<extra></extra>",
            ),
            row=3,
            col=1,
        )

    min_year = int(points_focus["year"].min())
    max_year = int(points_focus["year"].max())

    era_ranges = [
        (1987, 2005, ERA_ORDER[0]),
        (2006, 2013, ERA_ORDER[1]),
        (2014, 2021, ERA_ORDER[2]),
        (2022, max_year, ERA_ORDER[3]),
    ]

    for i, (start, end, label) in enumerate(era_ranges):
        opacity = 0.10 if i % 2 == 0 else 0.18
        # Use add_shape directly to avoid add_vrect crash on Sunburst traces
        # specific to mixed subplots. Targeting x5/y5 (Row 3, Col 1).
        fig.add_shape(
            type="rect",
            x0=start - 0.5,
            x1=end + 0.5,
            y0=0,
            y1=1,
            xref="x5",
            yref="y5 domain",
            fillcolor="white",
            opacity=opacity,
            layer="below",
            line_width=0,
        )

    fig.update_xaxes(title_text="Season", row=3, col=1)
    fig.update_yaxes(title_text="Constructor points", row=3, col=1)

    # --- Row 4: HHI bar chart ---

    fig.add_trace(
        go.Bar(
            x=hhi_df["era"],
            y=hhi_df["hhi_wins"],
            name="HHI (wins)",
            marker_color="purple",
            hovertemplate="<b>%{x}</b><br>HHI: %{y:.3f}<extra></extra>",
        ),
        row=4,
        col=1,
    )

    fig.update_xaxes(title_text="Era", row=4, col=1)
    fig.update_yaxes(
        title_text="Herfindahl index (0–1, higher = more concentrated)",
        row=4,
        col=1,
    )

    # --- Global layout ---

    fig.update_layout(
        title=dict(
            text="Formula 1: Evolution and Structure of Constructor Dominance (1987–present)",
            font=dict(size=22),
            x=0.5,
        ),
        height=1100,
        margin=dict(l=60, r=40, t=90, b=50),
        legend=dict(
            orientation="h",
            y=0.50,
            x=0.5,
            xanchor="center",
            bgcolor="rgba(0,0,0,0.3)",
        ),
        template="plotly_dark",
    )

    print(f"Saving to {OUTPUT_FILE}...")
    fig.write_html(OUTPUT_FILE)
    print("Done.")


if __name__ == "__main__":
    main()
