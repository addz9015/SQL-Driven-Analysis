
import pandas as pd
import sqlite3

# ── LOAD CSV INTO IN-MEMORY SQLite DB ─────────
df = pd.read_csv('cleaned_dataset.csv')
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
# ══════════════════════════════════════════════

q1 = run_query(
    "High-Value vs Low-Value Customer Separation",
    """
    WITH base_metrics AS (
        SELECT
            customer_id,
            value_tier,
            purchase_amount_usd,
            previous_purchases,
            purchase_freq_numeric,
            estimated_clv_proxy,
            review_rating,
            is_subscriber,
            is_promo_dependent,
            loyalty_score_value,
            loyalty_score_behavior
        FROM customers
    ),

    tier_summary AS (
        SELECT
            value_tier,
            COUNT(*)                                         AS customer_count,
            ROUND(AVG(purchase_amount_usd), 2)               AS avg_spend,
            ROUND(AVG(previous_purchases), 2)                AS avg_prev_purchases,
            ROUND(AVG(purchase_freq_numeric), 2)             AS avg_annual_freq,
            ROUND(AVG(estimated_clv_proxy), 2)               AS avg_clv_proxy,
            ROUND(AVG(review_rating), 2)                     AS avg_rating,
            SUM(is_subscriber)                               AS subscribers,
            ROUND(SUM(is_promo_dependent) * 100.0
                  / COUNT(*), 1)                             AS promo_pct,
            ROUND(AVG(loyalty_score_value), 4)               AS avg_loyalty_value_score,
            ROUND(AVG(loyalty_score_behavior), 4)            AS avg_loyalty_behavior_score
        FROM base_metrics
        GROUP BY value_tier
    )

    SELECT *
    FROM tier_summary
    ORDER BY avg_clv_proxy DESC
    """
)


# ══════════════════════════════════════════════
# QUERY 2: Promo-Dependent vs Organic Buyers
# ══════════════════════════════════════════════

q2 = run_query(
    "Promo-Dependent vs Organic Buyer Profiles",
    """
    WITH buyer_classification AS (
        SELECT
            customer_id,
            purchase_amount_usd,
            previous_purchases,
            estimated_clv_proxy,
            review_rating,
            purchase_freq_numeric,
            is_subscriber,
            is_promo_dependent,
            loyalty_score_value,
            loyalty_score_behavior,
            CASE
                WHEN is_promo_dependent = 1 THEN 'Promo-Dependent'
                ELSE 'Organic'
            END AS buyer_type
        FROM customers
    ),

    buyer_summary AS (
        SELECT
            buyer_type,
            COUNT(*)                                         AS customer_count,
            ROUND(AVG(purchase_amount_usd), 2)               AS avg_spend,
            ROUND(AVG(previous_purchases), 2)                AS avg_prev_purchases,
            ROUND(AVG(estimated_clv_proxy), 2)               AS avg_clv_proxy,
            ROUND(AVG(review_rating), 2)                     AS avg_rating,
            ROUND(AVG(purchase_freq_numeric), 2)             AS avg_annual_freq,
            SUM(is_subscriber)                               AS subscribers,
            ROUND(AVG(loyalty_score_value), 4)               AS avg_loyalty_value_score,
            ROUND(AVG(loyalty_score_behavior), 4)            AS avg_loyalty_behavior_score
        FROM buyer_classification
        GROUP BY buyer_type
    )

    SELECT *
    FROM buyer_summary
    ORDER BY avg_clv_proxy DESC
    """
)


# ══════════════════════════════════════════════
# QUERY 3: Geographic Demand Signal
# ══════════════════════════════════════════════

q3 = run_query(
    "Geographic Opportunity — Organic vs Discount-Driven (Top 15 States)",
    """
    WITH state_metrics AS (
        SELECT
            location                                         AS state,
            COUNT(*)                                         AS customers,
            ROUND(AVG(purchase_amount_usd), 2)               AS avg_spend,
            ROUND(AVG(estimated_clv_proxy), 2)               AS avg_clv_proxy,
            ROUND(AVG(review_rating), 2)                     AS avg_rating,
            ROUND(SUM(is_promo_dependent) * 100.0
                  / COUNT(*), 1)                             AS promo_rate_pct,
            ROUND(AVG(loyalty_score_value), 4)               AS avg_loyalty_value_score,
            ROUND(AVG(loyalty_score_behavior), 4)            AS avg_loyalty_behavior_score
        FROM customers
        GROUP BY location
        HAVING COUNT(*) >= 5
    ),

    state_classified AS (
        SELECT
            *,
            CASE
                WHEN avg_spend > 64
                 AND promo_rate_pct < 45  THEN 'High Opportunity'
                WHEN avg_spend > 58
                 AND promo_rate_pct < 52  THEN 'Moderate Opportunity'
                ELSE 'Discount Dependent'
            END AS geo_classification
        FROM state_metrics
    )

    SELECT *
    FROM state_classified
    ORDER BY avg_clv_proxy DESC
    LIMIT 15
    """
)


# ══════════════════════════════════════════════
# QUERY 4: Category Affinity by Purchase History
# ══════════════════════════════════════════════

q4 = run_query(
    "Category Affinity — Entry Point vs Retention Categories",
    """
    WITH category_metrics AS (
        SELECT
            category,
            COUNT(*)                                         AS total_customers,
            ROUND(AVG(previous_purchases), 2)                AS avg_prev_purchases,
            ROUND(AVG(purchase_amount_usd), 2)               AS avg_spend,
            ROUND(AVG(estimated_clv_proxy), 2)               AS avg_clv_proxy,
            ROUND(SUM(is_promo_dependent) * 100.0
                  / COUNT(*), 1)                             AS promo_rate_pct,
            ROUND(AVG(review_rating), 2)                     AS avg_rating,
            ROUND(AVG(loyalty_score_value), 4)               AS avg_loyalty_value_score,
            ROUND(AVG(loyalty_score_behavior), 4)            AS avg_loyalty_behavior_score
        FROM customers
        GROUP BY category
    ),

    category_classified AS (
        SELECT
            *,
            CASE
                WHEN avg_prev_purchases >= 25.5 THEN 'Retention Category'
                ELSE 'Entry-Point Category'
            END AS category_role
        FROM category_metrics
    )

    SELECT *
    FROM category_classified
    ORDER BY avg_prev_purchases DESC
    """
)


# ══════════════════════════════════════════════
# QUERY 5: Ideal Customer Profile
# ══════════════════════════════════════════════

q5 = run_query(
    "Ideal Customer Profile — Full Segment Breakdown",
    """
    WITH segment_metrics AS (
        SELECT
            customer_segment,
            COUNT(*)                                          AS count,
            ROUND(AVG(purchase_amount_usd), 2)                AS avg_spend,
            ROUND(AVG(previous_purchases), 2)                 AS avg_prev_purchases,
            ROUND(AVG(estimated_clv_proxy), 2)                AS avg_clv_proxy,
            ROUND(AVG(review_rating), 2)                      AS avg_rating,
            ROUND(AVG(age), 1)                                AS avg_age,
            ROUND(AVG(purchase_freq_numeric), 1)              AS avg_annual_freq,
            SUM(is_subscriber)                                AS subscribers,
            ROUND(SUM(CASE WHEN gender = 'Female'
                      THEN 1 ELSE 0 END) * 100.0
                  / COUNT(*), 1)                              AS pct_female,
            ROUND(AVG(loyalty_score_value), 4)                AS avg_loyalty_value_score,
            ROUND(AVG(loyalty_score_behavior), 4)             AS avg_loyalty_behavior_score
        FROM customers
        GROUP BY customer_segment
    ),

    segment_ranked AS (
        SELECT
            *,
            ROUND(count * 100.0 / SUM(count) OVER(), 1)      AS pct_of_total,
            RANK() OVER (ORDER BY avg_clv_proxy DESC)         AS clv_rank
        FROM segment_metrics
    )

    SELECT *
    FROM segment_ranked
    ORDER BY clv_rank
    """
)


# ══════════════════════════════════════════════
# BONUS: Promo Sunset Target Deep Dive
# ══════════════════════════════════════════════

q_bonus = run_query(
    "Promo Sunset Target — High-Value Promo-Dependent Deep Dive",
    """
    WITH target_segment AS (
        SELECT *
        FROM customers
        WHERE customer_segment = 'High-Value Promo-Dependent'
    ),

    segment_breakdown AS (
        SELECT
            category,
            season,
            frequency_of_purchases,
            COUNT(*)                                          AS count,
            ROUND(AVG(purchase_amount_usd), 2)                AS avg_spend,
            ROUND(AVG(previous_purchases), 2)                 AS avg_prev_purchases,
            ROUND(AVG(review_rating), 2)                      AS avg_rating,
            ROUND(AVG(loyalty_score_value), 4)                AS avg_loyalty_value_score,
            ROUND(AVG(loyalty_score_behavior), 4)             AS avg_loyalty_behavior_score
        FROM target_segment
        GROUP BY category, season, frequency_of_purchases
    )

    SELECT *
    FROM segment_breakdown
    ORDER BY avg_spend DESC
    LIMIT 15
    """
)


# ══════════════════════════════════════════════
# EXPORT ALL RESULTS TO EXCEL
# ══════════════════════════════════════════════

with pd.ExcelWriter('sql_results.xlsx', engine='openpyxl') as writer:
    q1.to_excel(writer, sheet_name='Q1 Value Tier Separation', index=False)
    q2.to_excel(writer, sheet_name='Q2 Promo vs Organic', index=False)
    q3.to_excel(writer, sheet_name='Q3 Geographic Signal', index=False)
    q4.to_excel(writer, sheet_name='Q4 Category Affinity', index=False)
    q5.to_excel(writer, sheet_name='Q5 Ideal Customer Profile', index=False)
    q_bonus.to_excel(writer, sheet_name='Bonus Promo Sunset Target', index=False)

print("DONE. SQL results saved to: sql_results.xlsx")

conn.close()
