"""
Nassau Candy Distributor — Core Analysis Module
Handles data prep, feature engineering, ML training & optimization logic.
"""

import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

# ── Factory metadata ─────────────────────────────────────────────────────────
FACTORIES = {
    "Lot's O' Nuts":     {"lat": 32.881893, "lon": -111.768036, "state": "Arizona"},
    "Wicked Choccy's":   {"lat": 32.076176, "lon": -81.088371,  "state": "Georgia"},
    "Sugar Shack":       {"lat": 48.119140, "lon": -96.181150,  "state": "Minnesota"},
    "Secret Factory":    {"lat": 41.446333, "lon": -90.565487,  "state": "Illinois"},
    "The Other Factory": {"lat": 35.117500, "lon": -89.971107,  "state": "Tennessee"},
}

PRODUCT_FACTORY = {
    "Wonka Bar - Nutty Crunch Surprise":    "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows":            "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious":       "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate":           "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel":    "Wicked Choccy's",
    "Laffy Taffy":                          "Sugar Shack",
    "SweeTARTS":                            "Sugar Shack",
    "Nerds":                                "Sugar Shack",
    "Fun Dip":                              "Sugar Shack",
    "Fizzy Lifting Drinks":                 "Sugar Shack",
    "Everlasting Gobstopper":               "Secret Factory",
    "Lickable Wallpaper":                   "Secret Factory",
    "Wonka Gum":                            "Secret Factory",
    "Hair Toffee":                          "The Other Factory",
    "Kazookles":                            "The Other Factory",
}

STATE_COORDS = {
    "Alabama": (32.7,-86.7),"Alaska":(64.2,-153.4),"Arizona":(34.3,-111.1),
    "Arkansas":(34.8,-92.2),"California":(36.8,-119.4),"Colorado":(39.0,-105.5),
    "Connecticut":(41.6,-72.7),"Delaware":(39.0,-75.5),"Florida":(27.8,-81.7),
    "Georgia":(32.2,-83.4),"Hawaii":(20.0,-156.0),"Idaho":(44.4,-114.5),
    "Illinois":(40.0,-89.2),"Indiana":(40.3,-86.1),"Iowa":(42.0,-93.2),
    "Kansas":(38.5,-98.4),"Kentucky":(37.5,-85.3),"Louisiana":(31.1,-91.8),
    "Maine":(45.2,-69.0),"Maryland":(39.0,-76.8),"Massachusetts":(42.2,-71.5),
    "Michigan":(44.4,-85.4),"Minnesota":(46.4,-93.1),"Mississippi":(32.7,-89.7),
    "Missouri":(38.5,-92.5),"Montana":(47.0,-110.0),"Nebraska":(41.5,-99.9),
    "Nevada":(39.3,-116.6),"New Hampshire":(43.7,-71.6),"New Jersey":(40.1,-74.5),
    "New Mexico":(34.8,-106.2),"New York":(42.2,-74.9),"North Carolina":(35.5,-79.8),
    "North Dakota":(47.5,-100.5),"Ohio":(40.4,-82.8),"Oklahoma":(35.6,-96.9),
    "Oregon":(44.0,-120.5),"Pennsylvania":(40.6,-77.2),"Rhode Island":(41.7,-71.5),
    "South Carolina":(33.9,-80.9),"South Dakota":(44.4,-100.3),"Tennessee":(35.9,-86.7),
    "Texas":(31.5,-99.3),"Utah":(39.3,-111.1),"Vermont":(44.1,-72.7),
    "Virginia":(37.8,-78.2),"Washington":(47.4,-120.5),"West Virginia":(38.6,-80.6),
    "Wisconsin":(44.3,-89.8),"Wyoming":(43.0,-107.6),
    "District of Columbia":(38.9,-77.0),
}

REGION_COORDS = {
    "Atlantic": (40.0, -75.0),
    "Gulf":     (29.5, -90.0),
    "Interior": (41.0, -88.0),
    "Pacific":  (37.5, -121.0),
}


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi, dlambda = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


def load_and_prepare(path="Nassau_Candy_Distributor.csv"):
    df = pd.read_csv(path)
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    df["Ship Date"]  = pd.to_datetime(df["Ship Date"],  dayfirst=True)
    df["Lead Time"]  = (df["Ship Date"] - df["Order Date"]).dt.days
    df["Factory"]    = df["Product Name"].map(PRODUCT_FACTORY)
    df["Order Year"]  = df["Order Date"].dt.year
    df["Order Month"] = df["Order Date"].dt.month
    df["Margin_pct"]  = (df["Gross Profit"] / df["Sales"].replace(0, np.nan)) * 100

    def sc(state):
        return STATE_COORDS.get(state, (39.5, -98.4))

    df["State Lat"] = df["State/Province"].apply(lambda s: sc(s)[0])
    df["State Lon"] = df["State/Province"].apply(lambda s: sc(s)[1])

    df["Distance_km"] = df.apply(
        lambda r: haversine(
            FACTORIES[r["Factory"]]["lat"], FACTORIES[r["Factory"]]["lon"],
            r["State Lat"], r["State Lon"]
        ) if r["Factory"] in FACTORIES else np.nan,
        axis=1,
    )
    # Relative lead time within ship-mode group (normalise out year effect)
    df["LT_relative"] = df.groupby(["Ship Mode","Order Year"])["Lead Time"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9)
    )
    return df


def build_features(df):
    from sklearn.preprocessing import LabelEncoder
    feat = df[["Ship Mode","Region","Division","Factory",
               "Distance_km","Units","Sales","Cost","Order Month"]].copy()
    encoders = {}
    for col in ["Ship Mode","Region","Division","Factory"]:
        le = LabelEncoder()
        feat[col] = le.fit_transform(feat[col].astype(str))
        encoders[col] = le
    feat = feat.fillna(feat.mean(numeric_only=True))
    return feat.values, df["Lead Time"].values, encoders, feat.columns.tolist()


def train_models(X, y):
    import warnings; warnings.filterwarnings("ignore")
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.preprocessing import StandardScaler

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler()
    Xs_tr, Xs_te = sc.fit_transform(X_tr), sc.transform(X_te)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest":     RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
    }
    results, fitted = {}, {}
    for name, m in models.items():
        xtr, xte = (Xs_tr, Xs_te) if name == "Linear Regression" else (X_tr, X_te)
        m.fit(xtr, y_tr)
        p = m.predict(xte)
        results[name] = {
            "RMSE": round(float(np.sqrt(mean_squared_error(y_te, p))), 2),
            "MAE":  round(float(mean_absolute_error(y_te, p)), 2),
            "R²":   round(float(r2_score(y_te, p)), 4),
        }
        fitted[name] = (m, sc if name == "Linear Regression" else None)

    best = max(results, key=lambda k: results[k]["R²"])
    return fitted, results, best


def factory_distance_profile(df):
    """Avg distance from each factory to each region."""
    rows = []
    for factory, fc in FACTORIES.items():
        for region, (rlat, rlon) in REGION_COORDS.items():
            dist = haversine(fc["lat"], fc["lon"], rlat, rlon)
            rows.append({"Factory": factory, "Region": region, "Distance_km": round(dist, 0)})
    return pd.DataFrame(rows)


def simulate_all_factories(df, product_name, ship_mode_filter=None):
    """Return a DataFrame comparing all factories for a given product."""
    sub = df[df["Product Name"] == product_name].copy()
    if ship_mode_filter:
        sub = sub[sub["Ship Mode"] == ship_mode_filter]
    if sub.empty:
        return pd.DataFrame()

    current_factory = PRODUCT_FACTORY.get(product_name)
    rows = []
    for factory, fc in FACTORIES.items():
        dist = haversine(
            fc["lat"], fc["lon"],
            sub["State Lat"].mean(), sub["State Lon"].mean(),
        )
        # Approximate lead-time delta based on distance change from current
        curr_dist = haversine(
            FACTORIES[current_factory]["lat"], FACTORIES[current_factory]["lon"],
            sub["State Lat"].mean(), sub["State Lon"].mean(),
        ) if current_factory in FACTORIES else dist
        # ~0.05 day per extra km (heuristic)
        lt_delta = (dist - curr_dist) * 0.05
        actual_lt = sub["Lead Time"].mean()
        rows.append({
            "Factory":              factory,
            "Is Current":          factory == current_factory,
            "Avg Distance (km)":   round(dist, 0),
            "Estimated Lead Time (days)": round(actual_lt + lt_delta, 1),
            "Avg Gross Profit ($)": round(sub["Gross Profit"].mean(), 2),
            "Total Revenue ($)":   round(sub["Sales"].sum(), 2),
            "Orders":              len(sub),
        })
    return pd.DataFrame(rows).sort_values("Avg Distance (km)")


def generate_recommendations(df):
    """For every product find best factory by distance and score."""
    recs = []
    for product, current_factory in PRODUCT_FACTORY.items():
        sub = df[df["Product Name"] == product]
        if sub.empty:
            continue
        avg_lat = sub["State Lat"].mean()
        avg_lon = sub["State Lon"].mean()
        curr_dist = haversine(
            FACTORIES[current_factory]["lat"], FACTORIES[current_factory]["lon"],
            avg_lat, avg_lon
        )
        best_factory, best_dist = current_factory, curr_dist
        for alt, fc in FACTORIES.items():
            d = haversine(fc["lat"], fc["lon"], avg_lat, avg_lon)
            if d < best_dist:
                best_dist, best_factory = d, alt

        lt = sub["Lead Time"].mean()
        lt_delta = (best_dist - curr_dist) * 0.05
        recs.append({
            "Product":                     product,
            "Division":                    sub["Division"].iloc[0],
            "Current Factory":             current_factory,
            "Recommended Factory":         best_factory,
            "Current Avg Distance (km)":   round(curr_dist, 0),
            "Recommended Avg Distance (km)": round(best_dist, 0),
            "Distance Reduction (km)":     round(curr_dist - best_dist, 0),
            "Distance Reduction (%)":      round((curr_dist - best_dist)/curr_dist*100, 1),
            "Est. Lead Time Improvement (days)": round(-lt_delta, 1),
            "Current Lead Time (days)":    round(lt, 1),
            "Avg Gross Profit ($)":        round(sub["Gross Profit"].mean(), 2),
            "Total Orders":                len(sub),
            "Reassignment Needed":         best_factory != current_factory,
        })
    recs_df = pd.DataFrame(recs)
    if not recs_df.empty:
        recs_df = recs_df.sort_values("Distance Reduction (%)", ascending=False).reset_index(drop=True)
    return recs_df
