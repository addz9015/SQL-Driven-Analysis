
import pandas as pd
import sqlite3

# ── LOAD CSV INTO IN-MEMORY SQLite DB ─────────
df = pd.read_csv('cleaned_dataset.csv')
feature_df = pd.read_csv('customer_features.csv')
conn = sqlite3.connect(':memory:')
df.to_sql('customers', conn, index=False, if_exists='replace')
print(f"Loaded {len(df)} rows into SQLite table: customers")
print(f"Columns: {df.columns.tolist()}\n")

def run_query(title, sql):
    print("=" * 60)
    print(f"QUERY: {title}")
    print("=" * 60)
    result = pd.read_sql_query(sql, conn)
    print(result.to_string(index=False))
    print()
    return result


# ══════════════════════════════════════════════
# QUERY 1: High-Value vs Low-Value Customers
# What separates them? Which profiles show strongest repeat behavior?
# ══════════════════════════════════════════════

q1 = run_query(
    "High-Value vs Low-Value Customer Separation",
    """
    SELECT
        value_tier,
        COUNT(*)                                        AS customer_count,
        ROUND(AVG(purchase_amount_usd), 2)              AS avg_spend,
        ROUND(AVG(previous_purchases), 2)               AS avg_prev_purchases,
        ROUND(AVG(purchase_freq_numeric), 2)            AS avg_annual_freq,
        ROUND(AVG(estimated_clv_proxy), 2)              AS avg_clv_proxy,
        ROUND(AVG(review_rating), 2)                    AS avg_rating,
        SUM(is_subscriber)                              AS subscribers,
        ROUND(SUM(is_promo_dependent) * 100.0
              / COUNT(*), 1)                            AS promo_pct
    FROM customers
    GROUP BY value_tier
    ORDER BY avg_clv_proxy DESC
    """
)

# ══════════════════════════════════════════════
# QUERY 2: Promo-Dependent vs Organic Buyers
# Who actually buys without a discount?
# ══════════════════════════════════════════════

q2 = run_query(
    "Promo-Dependent vs Organic Buyer Profiles",
    """
    SELECT
        is_promo_dependent,
        CASE WHEN is_promo_dependent = 1
             THEN 'Promo-Dependent' ELSE 'Organic' END  AS buyer_type,
        COUNT(*)                                        AS customer_count,
        ROUND(AVG(purchase_amount_usd), 2)              AS avg_spend,
        ROUND(AVG(previous_purchases), 2)               AS avg_prev_purchases,
        ROUND(AVG(estimated_clv_proxy), 2)              AS avg_clv_proxy,
        ROUND(AVG(review_rating), 2)                    AS avg_rating,
        ROUND(AVG(purchase_freq_numeric), 2)            AS avg_annual_freq,
        SUM(is_subscriber)                              AS subscribers
    FROM customers
    GROUP BY is_promo_dependent
    ORDER BY is_promo_dependent
    """
)

# ══════════════════════════════════════════════
# QUERY 3: Geographic Demand Signal
# Which states show organic demand vs discount-driven volume?
# ══════════════════════════════════════════════

q3 = run_query(
    "Geographic Opportunity — Organic Demand vs Discount-Driven (Top 15 States)",
    """
    SELECT
        location                                        AS state,
        COUNT(*)                                        AS customers,
        ROUND(AVG(purchase_amount_usd), 2)              AS avg_spend,
        ROUND(SUM(is_promo_dependent) * 100.0
              / COUNT(*), 1)                            AS promo_rate_pct,
        ROUND(AVG(estimated_clv_proxy), 2)              AS avg_clv_proxy,
        ROUND(AVG(review_rating), 2)                    AS avg_rating,
        CASE
            WHEN AVG(purchase_amount_usd) > 65
             AND SUM(is_promo_dependent) * 100.0 / COUNT(*) < 40
            THEN 'High Opportunity'
            WHEN AVG(purchase_amount_usd) > 55
             AND SUM(is_promo_dependent) * 100.0 / COUNT(*) < 50
            THEN 'Moderate Opportunity'
            ELSE 'Discount Dependent'
        END                                             AS geo_classification
    FROM customers
    GROUP BY location
    HAVING COUNT(*) >= 5
    ORDER BY avg_clv_proxy DESC
    LIMIT 15
    """
)

# ══════════════════════════════════════════════
# QUERY 4: Category Affinity by Purchase History
# Which categories appear among high-frequency vs low-frequency customers?
# Proxy for entry-point vs retention categories
# ══════════════════════════════════════════════

q4 = run_query(
    "Category Affinity — Entry Point vs Retention Categories",
    """
    SELECT
        category,
        COUNT(*)                                        AS total_customers,
        ROUND(AVG(previous_purchases), 2)               AS avg_prev_purchases,
        ROUND(AVG(purchase_amount_usd), 2)              AS avg_spend,
        ROUND(AVG(estimated_clv_proxy), 2)              AS avg_clv_proxy,
        ROUND(SUM(is_promo_dependent) * 100.0
              / COUNT(*), 1)                            AS promo_rate_pct,
        ROUND(AVG(review_rating), 2)                    AS avg_rating,
        CASE
            WHEN AVG(previous_purchases) >= 28
            THEN 'Retention Category'
            ELSE 'Entry-Point Category'
        END                                             AS category_role
    FROM customers
    GROUP BY category
    ORDER BY avg_prev_purchases DESC
    """
)

# ══════════════════════════════════════════════
# QUERY 5: Ideal Customer Profile
# Data-backed description of the brand's most valuable customer type
# ══════════════════════════════════════════════

q5 = run_query(
    "Ideal Customer Profile — Top Segment Breakdown",
    """
    SELECT
        customer_segment,
        COUNT(*)                                        AS count,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) AS pct_of_total,
        ROUND(AVG(purchase_amount_usd), 2)              AS avg_spend,
        ROUND(AVG(previous_purchases), 2)               AS avg_prev_purchases,
        ROUND(AVG(estimated_clv_proxy), 2)              AS avg_clv_proxy,
        ROUND(AVG(review_rating), 2)                    AS avg_rating,
        ROUND(AVG(age), 1)                              AS avg_age,
        ROUND(AVG(purchase_freq_numeric), 1)            AS avg_annual_freq,
        SUM(is_subscriber)                              AS subscribers,
        ROUND(SUM(CASE WHEN gender = 'Female'
                  THEN 1 ELSE 0 END) * 100.0
              / COUNT(*), 1)                            AS pct_female
    FROM customers
    GROUP BY customer_segment
    ORDER BY avg_clv_proxy DESC
    """
)

# ══════════════════════════════════════════════
# BONUS: Promotional Sunset Target Segment
# High-Value Promo-Dependent — the most dangerous cohort
# ══════════════════════════════════════════════

q_bonus = run_query(
    "Promo Sunset Target — High-Value Promo-Dependent Deep Dive",
    """
    SELECT
        category,
        season,
        frequency_of_purchases,
        COUNT(*)                                        AS count,
        ROUND(AVG(purchase_amount_usd), 2)              AS avg_spend,
        ROUND(AVG(previous_purchases), 2)               AS avg_prev_purchases,
        ROUND(AVG(review_rating), 2)                    AS avg_rating
    FROM customers
    WHERE customer_segment = 'High-Value Promo-Dependent'
    GROUP BY category, season, frequency_of_purchases
    ORDER BY avg_spend DESC
    LIMIT 15
    """
)


# ─────────────────────────────────────────────
# STEP 5: EXPORT
# ─────────────────────────────────────────────

df.to_csv('cleaned_dataset.csv', index=False)
feature_df.to_csv('customer_features.csv', index=False)

# Export to Excel (all engineered features in one file, separate sheets)
with pd.ExcelWriter('customer_analysis.xlsx', engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='Cleaned Full Dataset', index=False)
    feature_df.to_excel(writer, sheet_name='Customer Features', index=False)
    
    # Segment summary sheet
    seg_summary = df.groupby('customer_segment').agg(
        count=('customer_id', 'count'),
        avg_clv=('estimated_clv_proxy', 'mean'),
        avg_spend=('purchase_amount_usd', 'mean'),
        avg_rating=('review_rating', 'mean'),
        promo_pct=('is_promo_dependent', 'mean')
    ).round(2).reset_index()
    seg_summary['promo_pct'] = (seg_summary['promo_pct'] * 100).round(1)
    seg_summary.to_excel(writer, sheet_name='Segment Summary', index=False)

print("DONE. Files saved:")
print("  → cleaned_dataset.csv")
print("  → customer_features.csv")
print("  → customer_analysis.xlsx (3 sheets: full data, features, segment summary)")