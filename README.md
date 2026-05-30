# 🍬 Nassau Candy Distributor — Factory Reallocation & Shipping Optimization

> **Unified Mentor Capstone Project** | Decision Intelligence for Supply Chain Optimization

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red.svg)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org)
[![Plotly](https://img.shields.io/badge/Plotly-5.18+-blue.svg)](https://plotly.com)

---

## 📌 Problem Statement

Nassau Candy Distributor currently assigns products to factories using static legacy rules. This leads to:
- Suboptimal shipping distances and inflated lead times
- Margin erosion from logistics inefficiencies
- No system to simulate reassignment scenarios before execution

This project builds a **decision-intelligence system** that:
- Predicts shipping outcomes under different factory configurations
- Recommends which products should move to alternative factories
- Balances shipping efficiency and profitability

---

## 📁 Project Structure

```
nassau_candy_project/
├── app.py                          # Main Streamlit dashboard
├── requirements.txt                # Python dependencies
├── Nassau_Candy_Distributor.csv    # Dataset
├── analysis/
│   ├── __init__.py
│   └── core.py                     # Data prep, ML, optimization engine
└── README.md
```

---

## 🏭 Factory Locations

| Factory | State | Coordinates |
|---------|-------|-------------|
| Lot's O' Nuts | Arizona | 32.88°N, 111.77°W |
| Wicked Choccy's | Georgia | 32.08°N, 81.09°W |
| Sugar Shack | Minnesota | 48.12°N, 96.18°W |
| Secret Factory | Illinois | 41.45°N, 90.57°W |
| The Other Factory | Tennessee | 35.12°N, 89.97°W |

---

## 🔗 Product–Factory Mapping (Current)

| Product | Factory | Division |
|---------|---------|---------|
| Wonka Bar - Nutty Crunch Surprise | Lot's O' Nuts | Chocolate |
| Wonka Bar - Fudge Mallows | Lot's O' Nuts | Chocolate |
| Wonka Bar - Scrumdiddlyumptious | Lot's O' Nuts | Chocolate |
| Wonka Bar - Milk Chocolate | Wicked Choccy's | Chocolate |
| Wonka Bar - Triple Dazzle Caramel | Wicked Choccy's | Chocolate |
| Laffy Taffy | Sugar Shack | Sugar |
| SweeTARTS | Sugar Shack | Sugar |
| Nerds | Sugar Shack | Sugar |
| Fun Dip | Sugar Shack | Sugar |
| Fizzy Lifting Drinks | Sugar Shack | Other |
| Everlasting Gobstopper | Secret Factory | Sugar |
| Lickable Wallpaper | Secret Factory | Other |
| Wonka Gum | Secret Factory | Other |
| Hair Toffee | The Other Factory | Sugar |
| Kazookles | The Other Factory | Other |

---

## 📊 Dataset

- **10,194 orders** | 15 unique products | 4 US regions
- **Date range:** Jan 2024 – Dec 2025 (order dates)
- **Fields:** Order ID, Order/Ship Dates, Ship Mode, Customer Location, Division, Region, Product, Sales, Units, Gross Profit, Cost

---

## 🤖 ML Models

| Model | Purpose |
|-------|---------|
| Linear Regression | Baseline lead-time prediction |
| Random Forest Regressor | Non-linear feature interactions |
| Gradient Boosting Regressor | Sequential error correction |

**Note:** The primary optimization signal is haversine distance (factory → customer state centroid). Lead-time delta is estimated at **0.05 days per km** of distance deviation.

---

## 🚀 Running the App

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/nassau-candy-optimizer.git
cd nassau-candy-optimizer
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📈 Dashboard Modules

| Module | Description |
|--------|-------------|
| 📊 EDA & Overview | Revenue trends, lead time distributions, distance heatmaps |
| 🏭 Factory Simulator | Per-product factory comparison across all 5 facilities |
| 🔄 What-If Scenarios | Side-by-side current vs proposed factory assignment |
| 🎯 Recommendations | Ranked, scored reassignment suggestions with Sankey flow |
| 🤖 ML Performance | Feature importances, actual vs predicted, model metrics |

---

## 🎯 Key Findings

- **13 of 15 products** can reduce shipping distance through reassignment
- **SweeTARTS** (Sugar Shack → The Other Factory): **71.3% distance reduction**
- **Secret Factory** and **The Other Factory** are centrally located — ideal hubs
- **Sugar Shack** (Minnesota) is geographically disadvantaged for most US customers
- **Lot's O' Nuts** (Arizona) adds ~2,000+ km for East Coast customers

---

## 📋 KPIs Tracked

| KPI | Description |
|-----|-------------|
| Distance Reduction (%) | Primary optimization signal |
| Est. Lead Time Improvement (days) | Operational gain |
| Avg Gross Profit ($) | Profitability preservation |
| Composite Score | Weighted speed + profit blend |
| Recommendation Coverage | % of products with actionable suggestions |

---

## 👤 Author

**[Your Name]**
Unified Mentor Capstone — Factory Reallocation & Shipping Optimization
