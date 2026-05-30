# Factory Reallocation & Shipping Optimization for Nassau Candy Distributor
## Research Paper — Analytical Findings & Recommendations

---

## Executive Summary

This research analyzes 10,194 orders placed with Nassau Candy Distributor between January 2024 and December 2025. Using haversine distance calculations, machine learning models, and a scenario simulation engine, we identify that **13 of 15 products are suboptimally assigned** to factories, with potential shipping distance reductions ranging from 5.3% to 71.3%. The central recommendation is a strategic shift toward the **Secret Factory (Illinois)** and **The Other Factory (Tennessee)** as primary distribution hubs, given their geographic centrality to the US customer base.

---

## 1. Dataset Overview

| Attribute | Value |
|-----------|-------|
| Total Orders | 10,194 |
| Unique Products | 15 |
| Factories | 5 |
| Regions | Atlantic, Gulf, Interior, Pacific |
| Ship Modes | Standard Class, Second Class, First Class, Same Day |
| Date Range | Jan 2024 – Dec 2025 (orders) |

### 1.1 Revenue Distribution
- **Chocolate division** leads in average margin (~67%)
- **Sugar division** has the highest order volume
- **California, New York, Texas** are top customer states by order count

---

## 2. Current Factory Performance Analysis

### 2.1 Average Shipping Distance by Factory (km to customers)

| Factory | Avg Distance to Customers (km) | Rank |
|---------|-------------------------------|------|
| Secret Factory | ~1,369 | 1st (best) |
| The Other Factory | ~1,608 | 2nd |
| Sugar Shack | ~1,750 | 3rd |
| Wicked Choccy's | ~1,865 | 4th |
| Lot's O' Nuts | ~2,062 | 5th (worst) |

### 2.2 Key Insight
**Lot's O' Nuts (Arizona)** ships Chocolate products to predominantly East Coast customers, adding 600–1,200 km versus a central factory. **Sugar Shack (Minnesota)** serves a heavily coast-focused customer base despite being in the upper Midwest — a structural inefficiency.

---

## 3. Lead Time Analysis

### 3.1 Observed Lead Times

The dataset shows lead times clustering into three bands, driven by order-year cohort effects:
- **2024 cohort:** ~908–1,277 days average
- **2025 cohort:** ~1,637–1,642 days average

### 3.2 Lead Time by Ship Mode

| Ship Mode | Avg Lead Time (days) |
|-----------|---------------------|
| Standard Class | 1,314 |
| Second Class | 1,324 |
| Same Day | 1,333 |
| First Class | 1,338 |

**Note:** Counter-intuitively, faster ship modes show slightly higher observed lead times. This is likely a dataset artifact where premium ship modes are used for more distant routes. Distance-based modeling is more reliable for optimization.

---

## 4. Machine Learning Results

### 4.1 Model Performance

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Linear Regression | 266.2 | 214.95 | -0.002 |
| Random Forest | ~285 | ~232 | -0.15 |
| Gradient Boosting | ~268 | ~215 | -0.01 |

### 4.2 Interpretation
The low R² values reflect that lead times in this dataset are dominated by the order-year cohort effect (2024 vs 2025 orders ship to dramatically different future horizons), masking the contribution of operational features. The **distance-based simulation engine** — using haversine geometry — provides more interpretable and actionable optimization signals.

### 4.3 Feature Importance (Random Forest)
Top contributors to variance in lead time predictions:
1. Sales / Cost (proxy for order complexity)
2. Distance_km (logistics signal)
3. Order Month (seasonal routing)
4. Ship Mode (service tier)

---

## 5. Optimization Recommendations

### 5.1 Products to Reassign (ranked by distance reduction)

| Product | Current Factory | Recommended Factory | Distance Reduction |
|---------|----------------|---------------------|--------------------|
| SweeTARTS | Sugar Shack | The Other Factory | **71.3%** |
| Wonka Bar - Fudge Mallows | Lot's O' Nuts | Secret Factory | **70.7%** |
| Wonka Bar - Scrumdiddlyumptious | Lot's O' Nuts | Secret Factory | **68.5%** |
| Wonka Bar - Nutty Crunch Surprise | Lot's O' Nuts | Secret Factory | **67.4%** |
| Fizzy Lifting Drinks | Sugar Shack | The Other Factory | **66.0%** |
| Wonka Bar - Milk Chocolate | Wicked Choccy's | Secret Factory | **65.2%** |
| Wonka Bar - Triple Dazzle Caramel | Wicked Choccy's | Secret Factory | **63.0%** |
| Fun Dip | Sugar Shack | Secret Factory | **54.8%** |
| Laffy Taffy | Sugar Shack | Secret Factory | **51.2%** |
| Everlasting Gobstopper | Secret Factory | The Other Factory | **45.9%** |
| Nerds | Sugar Shack | Wicked Choccy's | **32.7%** |
| Wonka Gum | Secret Factory | The Other Factory | **5.3%** |
| Kazookles | The Other Factory | Secret Factory | **5.3%** |

### 5.2 Products Optimally Assigned (no change needed)
- **Hair Toffee** — The Other Factory ✅
- **Lickable Wallpaper** — Secret Factory ✅

---

## 6. Strategic Insights

### 6.1 Hub Concentration
Moving production toward **Secret Factory (IL)** and **The Other Factory (TN)** consolidates geographic reach. These two facilities sit near the population-weighted center of the US, minimizing average distances across all four regions.

### 6.2 Sugar Shack Exposure
Sugar Shack (Minnesota) is geographically disadvantaged. It currently serves 5 products. Moving **Laffy Taffy, SweeTARTS, Fun Dip, and Fizzy Lifting Drinks** to central factories could reduce distances by 51–71%.

### 6.3 Lot's O' Nuts Bottleneck
Lot's O' Nuts (Arizona) is the furthest factory from the East Coast customer concentration. All three Wonka Bar products it manufactures should be considered for reassignment to Secret Factory.

### 6.4 Profit Stability
Across all recommended reassignments, **gross profit per order is unchanged** since production costs are factory-specific and not affected by our shipping optimization. The gains are entirely in logistics cost reduction.

---

## 7. Conclusions

1. The current factory-product mapping was designed for production specialization, not logistics efficiency.
2. A geographically-centered reassignment strategy — anchored around Secret Factory and The Other Factory — can reduce average shipping distances by **52% across the product portfolio**.
3. The Streamlit simulator enables leadership to validate any scenario before execution, with real-time composite scoring against speed vs profit priorities.
4. Recommended next steps: (a) validate production capabilities at receiving factories, (b) pilot top-3 reassignments, (c) instrument a 90-day A/B test on lead times.

---

*Prepared for: Unified Mentor Capstone Program*
*Dataset Source: Nassau Candy Distributor via Unified Mentor*
