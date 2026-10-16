import os
import math
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import io
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Gas Price Simulator",
    page_icon="⛽",
    layout="wide",
)


try:
    TOKEN = st.secrets["APIFY_API_TOKEN"].strip()
except Exception:
    TOKEN = os.getenv("APIFY_API_TOKEN", "").strip()

HISTORY_FILE = Path("station_prices.csv")

# Verify this actor's input schema and dataset fields against the
# current provider documentation if the provider changes its API.
ACTOR_URL = (
    "https://api.apify.com/v2/acts/"
    "johnvc~fuelprices/run-sync-get-dataset-items"
)

HISTORY_COLUMNS = [
    "station_id",
    "station",
    "address",
    "latitude",
    "longitude",
    "price",
    "reported_at",
    "observed_at",
    "distance_miles",
    "price_type",
]


def empty_history():
    return pd.DataFrame(columns=HISTORY_COLUMNS)


def haversine_miles(lat1, lon1, lat2, lon2):
    earth_radius = 3958.8
    lat1, lon1, lat2, lon2 = map(
        math.radians, [lat1, lon1, lat2, lon2]
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )
    a = min(1.0, max(0.0, a))
    return 2 * earth_radius * math.asin(math.sqrt(a))


def load_history():
    if not HISTORY_FILE.exists():
        return empty_history()

    try:
        df = pd.read_csv(HISTORY_FILE)
        for column in HISTORY_COLUMNS:
            if column not in df.columns:
                df[column] = np.nan

        for column in [
            "latitude",
            "longitude",
            "price",
            "distance_miles",
        ]:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        for column in ["reported_at", "observed_at"]:
            df[column] = pd.to_datetime(
                df[column], errors="coerce", utc=True
            )

        df["station_id"] = df["station_id"].astype("string")
        df["station"] = df["station"].astype("string")
        df["address"] = df["address"].fillna("").astype(str)
        df["price_type"] = df["price_type"].fillna("Unknown").astype(str)

        df = df.dropna(
            subset=["station_id", "station", "price", "observed_at"]
        )
        df = df[df["price"] > 0]
        return df[HISTORY_COLUMNS].sort_values("observed_at")

    except Exception as exc:
        st.error(f"Could not read {HISTORY_FILE}: {exc}")
        return empty_history()


def fetch_live_stations(latitude, longitude, radius_miles, max_age_days, price_type):
    if not TOKEN:
        raise ValueError(
            "Missing APIFY_API_TOKEN. Add your token to the .env file."
        )

    payload = {
        "search": f"{latitude}, {longitude}",
        "fuel": 1,  # regular gasoline for this provider's documented schema
        "lang": "en",
        "maxAge": max_age_days,
    }

    response = requests.post(
        ACTOR_URL,
        params={"token": TOKEN},
        json=payload,
        timeout=180,
    )

    if response.status_code == 401:
        raise ValueError("API token rejected. Check your .env file.")
    if response.status_code == 402:
        raise ValueError("Provider reports insufficient funds or usage allowance.")
    if response.status_code == 429:
        raise ValueError("Rate limit reached. Wait before trying again.")

    response.raise_for_status()
    records = response.json()

    if not isinstance(records, list):
        raise ValueError(
            "Unexpected API response: expected a list of station records. "
            "Check the provider's current actor input/output schema."
        )

    now = datetime.now(timezone.utc).isoformat()
    rows = []

    for item in records:
        if not isinstance(item, dict):
            continue

        station_id = (
            item.get("id")
            or item.get("station_id")
            or item.get("place_id")
        )
        station_name = item.get("name") or item.get("station")

        if not station_id or not station_name:
            continue

        if price_type == "Cash":
            price = item.get("price_cash")
            reported_at = item.get("price_cash_postedTime")
        else:
            price = item.get("price_credit")
            reported_at = item.get("price_credit_postedTime")

        if price is None:
            if price_type == "Cash":
                price = item.get("price_credit")
                reported_at = item.get("price_credit_postedTime")
            else:
                price = item.get("price_cash")
                reported_at = item.get("price_cash_postedTime")

        try:
            price = float(price)
        except (TypeError, ValueError):
            continue

        if not np.isfinite(price) or price <= 0:
            continue

        address_parts = [
            item.get("address_line1"),
            item.get("address_line2"),
            item.get("address_locality"),
            item.get("address_region"),
            item.get("address_postalCode"),
        ]
        address = ", ".join(
            str(part).strip()
            for part in address_parts
            if part and str(part).strip()
        )

        # Coordinates can vary by provider schema; leave unknown rather than guess.
        lat = item.get("latitude", item.get("lat"))
        lon = item.get("longitude", item.get("lng", item.get("lon")))

        try:
            lat = float(lat) if lat is not None else np.nan
            lon = float(lon) if lon is not None else np.nan
        except (TypeError, ValueError):
            lat, lon = np.nan, np.nan

        distance = item.get("distance")
        try:
            distance = float(distance) if distance is not None else np.nan
        except (TypeError, ValueError):
            distance = np.nan

        # If station coordinates are available, calculate distance locally.
        if np.isfinite(lat) and np.isfinite(lon):
            distance = haversine_miles(latitude, longitude, lat, lon)

        if np.isfinite(distance) and distance > radius_miles:
            continue

        rows.append({
            "station_id": str(station_id),
            "station": str(station_name),
            "address": address,
            "latitude": lat,
            "longitude": lon,
            "price": price,
            "reported_at": reported_at,
            "observed_at": now,
            "distance_miles": distance,
            "price_type": price_type,
        })

    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)


def simulate_prices(history, station_id, days_ahead=30, n_sims=5000, seed=42):
    """
    Historical-return bootstrap Monte Carlo model.

    NOT YET WORKING
    """
    station_history = history[
        history["station_id"].astype(str) == str(station_id)
    ].copy()

    station_history["observed_at"] = pd.to_datetime(
        station_history["observed_at"], utc=True, errors="coerce"
    )
    station_history["price"] = pd.to_numeric(
        station_history["price"], errors="coerce"
    )

    station_history = station_history.dropna(
        subset=["observed_at", "price"]
    )
    station_history = station_history[station_history["price"] > 0]

    if station_history.empty:
        return None

    # A station may be checked several times a day; average each calendar day.
    daily = (
        station_history.set_index("observed_at")["price"]
        .resample("D")
        .mean()
        .dropna()
    )

    if len(daily) < 10:
        return None

    # Avoid treating gaps longer than one day as daily observations:
    # only calculate changes between adjacent calendar days.
    daily_returns = np.log(daily).diff().dropna()
    adjacent = daily.index.to_series().diff().dt.total_seconds().div(86400)
    daily_returns = daily_returns[adjacent.reindex(daily_returns.index).eq(1)]

    returns = daily_returns.to_numpy(dtype=float)
    returns = returns[np.isfinite(returns)]

    if len(returns) < 9:
        return None

    # Shrink drift to limit overconfidence from short histories.
    drift = float(np.clip(np.mean(returns) * 0.25, -0.005, 0.005))
    volatility = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0

    rng = np.random.default_rng(seed)
    start_price = float(daily.iloc[-1])

    # Bootstrap historical returns; drift is a small regularized adjustment.
    sampled = rng.choice(
        returns,
        size=(n_sims, days_ahead),
        replace=True,
    )
    paths = start_price * np.exp(np.cumsum(sampled + drift, axis=1))
    paths = np.column_stack([np.full(n_sims, start_price), paths])

    ending = paths[:, -1]
    return {
        "paths": paths,
        "start_price": start_price,
        "median": float(np.percentile(ending, 50)),
        "p10": float(np.percentile(ending, 10)),
        "p90": float(np.percentile(ending, 90)),
        "probability_up": float(np.mean(ending > start_price) * 100),
        "probability_down": float(np.mean(ending < start_price) * 100),
        "last_date": daily.index[-1],
        "daily_volatility": volatility,
        "observed_days": len(daily),
    }


st.title("Gas Price Monte Carlo Simulator")
st.write(
    "Compare real reported regular-gas prices and simulate possible future "
    "prices from saved station history."
)

with st.sidebar:
    st.header("Search settings")
    latitude = st.number_input(
        "Your latitude", min_value=-90.0, max_value=90.0,
        value=30.3322, format="%.6f"
    )
    longitude = st.number_input(
        "Your longitude", min_value=-180.0, max_value=180.0,
        value=-81.6557, format="%.6f"
    )
    radius_miles = st.slider("Search radius (miles)", 1, 50, 15)
    price_type = st.selectbox("Price type", ["Credit", "Cash"])
    max_age_days = st.selectbox(
        "Maximum age of reported prices", [1, 3, 7, 14], index=2
    )
    st.caption(
        "Replace the example coordinates with your own. Live provider "
    )

history = load_history()

if st.button("Fetch live station prices", type="primary"):
    with st.spinner("Fetching station prices..."):
        try:
            live = fetch_live_stations(
                latitude,
                longitude,
                radius_miles,
                max_age_days,
                price_type,
            )
            if live.empty:
                st.warning(
                    "No usable stations were returned. Check your API access, "
                    "coordinates, price type, and the provider's current schema."
                )
            else:
                history = pd.concat([history, live], ignore_index=True)
                history = history.drop_duplicates(
                    subset=["station_id", "observed_at", "price_type"],
                    keep="last",
                )
                history.to_csv(HISTORY_FILE, index=False)
                st.success(
                    f"Saved {len(live)} station observations to "
                    f"{HISTORY_FILE.resolve()}."
                )
                st.rerun()
        except requests.RequestException as exc:
            st.error(f"Network/API request failed: {exc}")
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")

history = load_history()

if history.empty:
    st.info(
        "No station history yet. Add your Apify token to .env, then fetch "
        "live station prices. Historical simulation becomes available after "
        "at least 10 distinct daily prices per station."
    )
    st.stop()

# Show latest observation per station, using the requested price type when possible.
history = history.sort_values("observed_at")
latest = history.groupby("station_id", as_index=False).tail(1).copy()

latest["reported_at"] = pd.to_datetime(
    latest["reported_at"], utc=True, errors="coerce"
)
now = pd.Timestamp.now(tz="UTC")
latest["report_age_hours"] = (
    now - latest["reported_at"]
).dt.total_seconds() / 3600

latest["report_status"] = np.select(
    [
        latest["report_age_hours"].isna(),
        latest["report_age_hours"] <= 24,
        latest["report_age_hours"] <= 72,
    ],
    ["Unknown", "Recent", "Aged"],
    default="Stale",
)

if np.isfinite(latitude) and np.isfinite(longitude):
    coords_ok = latest["latitude"].notna() & latest["longitude"].notna()
    latest.loc[coords_ok, "distance_miles"] = latest.loc[coords_ok].apply(
        lambda r: haversine_miles(
            latitude, longitude, r["latitude"], r["longitude"]
        ),
        axis=1,
    )

nearby = latest[
    latest["distance_miles"].isna()
    | (latest["distance_miles"] <= radius_miles)
].copy()

if nearby.empty:
    st.warning("No stations remain in the selected radius.")
    st.stop()

forecasts = []
for _, row in nearby.iterrows():
    result = simulate_prices(
        history,
        row["station_id"],
        days_ahead=7,
        n_sims=1000,
        seed=42,
    )
    forecasts.append(result["median"] if result else np.nan)

nearby["seven_day_median"] = forecasts
nearby["seven_day_change"] = nearby["seven_day_median"] - nearby["price"]

c1, c2, c3 = st.columns(3)
c1.metric("Stations", len(nearby))
c2.metric("Lowest reported price", f"${nearby['price'].min():.3f}/gal")
c3.metric("Stations with enough history", int(nearby["seven_day_median"].notna().sum()))

st.subheader("Latest station prices")
table = nearby[
    [
        "station", "address", "price", "distance_miles",
        "reported_at", "report_status", "seven_day_median",
        "seven_day_change",
    ]
].rename(
    columns={
        "station": "Station",
        "address": "Address",
        "price": "Latest price ($/gal)",
        "distance_miles": "Distance (mi)",
        "reported_at": "Price report time",
        "report_status": "Report freshness",
        "seven_day_median": "7-day simulated median",
        "seven_day_change": "7-day change ($/gal)",
    }
)
st.dataframe(table.sort_values("Latest price ($/gal)"), use_container_width=True, hide_index=True)

price_fig = px.bar(
    nearby.sort_values("price"),
    x="station",
    y="price",
    hover_data=["address", "distance_miles", "report_status"],
    labels={"station": "Station", "price": "Price ($/gallon)"},
    title="Latest saved station prices",
)
st.plotly_chart(price_fig, use_container_width=True)

st.subheader("Monte Carlo future price simulation")
days_ahead = st.slider("Forecast horizon (days)", 1, 90, 30)
n_sims = st.select_slider(
    "Number of simulations",
    options=[1000, 2500, 5000, 10000],
    value=5000,
)

station_options = nearby[["station_id", "station", "address"]].drop_duplicates("station_id").copy()
station_options["label"] = (
    station_options["station"].astype(str)
    + " — "
    + station_options["address"].astype(str)
)
selected_label = st.selectbox("Choose a station to simulate", station_options["label"].tolist())
selected = station_options[station_options["label"] == selected_label].iloc[0]

result = simulate_prices(
    history,
    selected["station_id"],
    days_ahead=days_ahead,
    n_sims=n_sims,
    seed=42,
)

if result is None:
    st.warning(
        "Not enough history for this station. The simulator requires at least "
        "10 distinct daily price observations and 9 adjacent-day price changes. "
        "Collect observations on different days; repeated fetches on one day "
        "do not substitute for historical data."
    )
else:
    start_price = result["start_price"]
    median_price = result["median"]
    low_price = result["p10"]
    high_price = result["p90"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Last daily average", f"${start_price:.3f}/gal")
    m2.metric(
        f"Median at day {days_ahead}",
        f"${median_price:.3f}/gal",
        delta=f"${median_price - start_price:+.3f}",
    )
    m3.metric("10th–90th percentile", f"${low_price:.3f}–${high_price:.3f}")
    m4.metric("Simulated chance of increase", f"{result['probability_up']:.1f}%")

    paths = result["paths"]
    dates = pd.date_range(
        start=result["last_date"],
        periods=days_ahead + 1,
        freq="D",
    )
    p10 = np.percentile(paths, 10, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p90 = np.percentile(paths, 90, axis=0)

    fig = go.Figure()

    sample_indices = np.linspace(
        0, n_sims - 1, min(100, n_sims), dtype=int
    )
    for idx in sample_indices:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=paths[idx],
                mode="lines",
                line=dict(width=0.7, color="rgba(100,100,100,0.18)"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=dates, y=p90, mode="lines",
            line=dict(width=0),
            name="90th percentile",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates, y=p10, mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(50,120,200,0.18)",
            name="Central 80% range",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates, y=p50, mode="lines",
            line=dict(width=3),
            name="Median simulated price",
        )
    )
    fig.add_hline(
        y=start_price,
        line_dash="dash",
        annotation_text="Last daily average",
    )
    fig.update_layout(
        title=f"Possible future prices — {selected['station']}",
        xaxis_title="Date",
        yaxis_title="Price per gallon ($)",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write(
        f"Based on {result['observed_days']} distinct daily observations, "
        f"the median simulated ending price is ${median_price:.3f}/gal. "
        f"The central 80% of simulated outcomes is "
        f"${low_price:.3f}–${high_price:.3f}/gal."
    )
    st.caption(
        "This bootstrap model assumes historical daily changes remain useful "
        "for future scenarios. It does not explicitly model crude-oil shocks, "
        "refinery outages, hurricanes, taxes, seasonality, or changes in station "
        "pricing policy. Percentile ranges are model outputs, not guaranteed "
        "confidence intervals or calibrated real-world probabilities."
    )

st.subheader("Saved observations")
st.download_button(
    "Download station history CSV",
    history.to_csv(index=False),
    file_name="station_prices.csv",
    mime="text/csv",
)

@st.cache_data(ttl=3600)
def load_two_month_history():
    """Load the latest two months of weekly U.S. regular gas prices."""
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=GASREGW"

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    history = pd.read_csv(io.StringIO(response.text))
    history.columns = ["date", "price"]

    history["date"] = pd.to_datetime(
        history["date"], errors="coerce"
    )
    history["price"] = pd.to_numeric(
        history["price"], errors="coerce"
    )

    history = history.dropna(subset=["date", "price"])
    history = history[history["price"] > 0]

    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=62)

    history = history[history["date"] >= cutoff]
    return history.sort_values("date").reset_index(drop=True)


st.header("Two-Month Gas Price History and Forecast")

try:
    history = load_two_month_history()

    if len(history) < 5:
        st.warning(
            "There is not enough recent historical data to make a forecast."
        )
    else:
        # Show actual historical prices.
        historical_chart = go.Figure()

        historical_chart.add_trace(
            go.Scatter(
                x=history["date"],
                y=history["price"],
                mode="lines+markers",
                name="Historical U.S. average",
            )
        )

        historical_chart.update_layout(
            title="Regular Gasoline Prices — Past Two Months",
            xaxis_title="Date",
            yaxis_title="Price per Gallon ($)",
            hovermode="x unified",
        )

        st.plotly_chart(
            historical_chart,
            use_container_width=True,
        )

        st.caption(
            "Source: FRED GASREGW, weekly U.S. regular-gasoline average. "
            "This is a national benchmark, not station-specific history."
        )

        # Choose a station price from the existing station table.
        # The existing dashboard must have a DataFrame named `nearby`
        # containing station_id and price columns.
        station_options = nearby["station_id"].astype(str).tolist()

        selected_station_id = st.selectbox(
            "Choose a station for the forecast",
            station_options,
            format_func=lambda station_id: (
                nearby.loc[
                    nearby["station_id"].astype(str) == station_id,
                    "station"
                ].iloc[0]
            ),
        )

        selected_row = nearby[
            nearby["station_id"].astype(str) == selected_station_id
        ].iloc[0]

        current_price = float(selected_row["price"])

        weeks = st.slider(
            "Forecast horizon (weeks)",
            min_value=1,
            max_value=12,
            value=4,
        )

        # Calculate weekly percentage changes from the historical data.
        weekly_returns = (
            history["price"].pct_change().dropna().to_numpy()
        )
        weekly_returns = weekly_returns[
            np.isfinite(weekly_returns)
        ]

        if len(weekly_returns) < 4:
            st.warning("There are too few historical price changes.")
        else:
            # Monte Carlo scenario simulation.
            simulations = 5000
            rng = np.random.default_rng()

            sampled_changes = rng.choice(
                weekly_returns,
                size=(simulations, weeks),
                replace=True,
            )

            # Conservative drift: use only 25% of the recent average change.
            drift = float(
                np.clip(np.mean(weekly_returns) * 0.25, -0.02, 0.02)
            )

            future_prices = current_price * np.cumprod(
                1 + sampled_changes + drift,
                axis=1,
            )

            dates = pd.date_range(
                start=pd.Timestamp.now().normalize()
                + pd.Timedelta(days=7),
                periods=weeks,
                freq="7D",
            )

            low = np.percentile(future_prices, 10, axis=0)
            median = np.percentile(future_prices, 50, axis=0)
            high = np.percentile(future_prices, 90, axis=0)

            forecast_chart = go.Figure()

            forecast_chart.add_trace(
                go.Scatter(
                    x=dates,
                    y=high,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    name="90th percentile",
                )
            )

            forecast_chart.add_trace(
                go.Scatter(
                    x=dates,
                    y=low,
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    name="Central 80% scenario range",
                )
            )

            forecast_chart.add_trace(
                go.Scatter(
                    x=dates,
                    y=median,
                    mode="lines+markers",
                    name="Median scenario",
                )
            )

            forecast_chart.add_hline(
                y=current_price,
                line_dash="dash",
                annotation_text="Current station price",
            )

            forecast_chart.update_layout(
                title=f"Possible Future Prices — {selected_row['station']}",
                xaxis_title="Forecast Date",
                yaxis_title="Price per Gallon ($)",
                hovermode="x unified",
            )

            st.plotly_chart(
                forecast_chart,
                use_container_width=True,
            )

            final_prices = future_prices[:, -1]

            col1, col2, col3 = st.columns(3)

            col1.metric(
                f"Median scenario in {weeks} weeks",
                f"${np.percentile(final_prices, 50):.3f}",
                delta=(
                    f"${np.percentile(final_prices, 50) - current_price:+.3f}"
                ),
            )

            col2.metric(
                "Central 80% scenario range",
                f"${np.percentile(final_prices, 10):.3f}–"
                f"${np.percentile(final_prices, 90):.3f}",
            )

            col3.metric(
                "Chance of a higher price",
                f"{np.mean(final_prices > current_price) * 100:.1f}%",
            )

            st.info(
                "Take the results with a grain of salt. T"
                "hey are simulated scenarios, not guaranteed predictions. "
                "The model applies national historical percentage changes "
                "to the selected station's current price."
            )

except Exception as error:
    st.error(
        "Could not load historical gasoline prices. "
        f"Check your internet connection. Details: {error}"
    )

