
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# STEP 1: LOAD & INSPECT
# ─────────────────────────────────────────────

df = pd.read_csv('Dataset.csv')

# Standardize column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(' ', '_')
    .str.replace('[()]', '', regex=True)
    .str.replace('/', '_')
)

print("=" * 60)
print("STEP 1: LOAD & INSPECT")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"\nColumns:\n{df.columns.tolist()}")
print(f"\nDtypes:\n{df.dtypes}")
print(f"\nNull counts:\n{df.isnull().sum()}")
print(f"\nNull % per column:\n{(df.isnull().sum() / len(df) * 100).round(2)}")
print(f"\nDescribe (numeric):\n{df.describe()}")

print("\nCategorical value counts:")
for col in df.select_dtypes(include=['object']).columns:
    print(f"\n{col}:\n{df[col].value_counts().head(10)}")


# ─────────────────────────────────────────────
# STEP 2: CLEAN
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 2: CLEAN")
print("=" * 60)

# Drop duplicates
before = len(df)
df = df.drop_duplicates()
print(f"Duplicates dropped: {before - len(df)}")

# Handle review_rating nulls (only column with nulls — 37 rows, ~1%)
df['review_rating_was_null'] = df['review_rating'].isnull().astype(int)
median_review = df['review_rating'].median()
df['review_rating'] = df['review_rating'].fillna(median_review)
print(f"review_rating: {df['review_rating_was_null'].sum()} nulls imputed with median ({median_review})")

# State abbreviation check for Power BI map compatibility
# Dataset uses full state names (e.g. "Kentucky") — create abbreviated version
state_abbrev = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
    'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
    'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
    'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
    'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
    'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
    'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
    'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
    'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',
    'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
    'Wisconsin': 'WI', 'Wyoming': 'WY'
}
df['state_abbrev'] = df['location'].map(state_abbrev)
unmapped = df['state_abbrev'].isnull().sum()
if unmapped > 0:
    print(f"WARNING: {unmapped} locations could not be mapped to state abbreviations")
else:
    print("State abbreviations mapped successfully for Power BI map compatibility")

print(f"\nPost-clean shape: {df.shape}")
print(f"Remaining nulls:\n{df.isnull().sum()[df.isnull().sum() > 0]}")


# ─────────────────────────────────────────────
# STEP 3: FEATURE ENGINEERING
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 3: FEATURE ENGINEERING")
print("=" * 60)

# ── PROXY DISCLOSURE ──────────────────────────
# IMPORTANT: This dataset has 1 row per transaction, not per customer.
# customer_id appears exactly once. There is no multi-transaction history.
# previous_purchases is a self-reported cumulative field, not computed.
# No timestamps exist. All features are constructed from available variables only.

# ── PROMO & DISCOUNT FEATURES ─────────────────

# promo_dependency_score
# FORMULA: 1 if promo_code_used == 'Yes', else 0
# NOTE: discount_applied and promo_code_used are perfectly correlated (same 1677 Yes).
#       Using promo_code_used as the canonical signal.
# BUSINESS JUSTIFICATION: Identifies whether this customer's purchase was discount-driven.
#                         High promo dependency = potential margin risk on retention.
df['promo_dependency_score'] = (df['promo_code_used'] == 'Yes').astype(float)
df['is_promo_dependent'] = (df['promo_code_used'] == 'Yes').astype(int)

print(f"\npromo_dependency_score distribution:\n{df['promo_dependency_score'].describe()}")
print(f"is_promo_dependent counts:\n{df['is_promo_dependent'].value_counts()}")
print("# BUSINESS JUSTIFICATION: promo_dependency_score — flags discount-driven purchases to identify margin risk")

# ── VALUE FEATURES ────────────────────────────

# Purchase frequency mapped to annual equivalent
# FORMULA: frequency_of_purchases string → estimated annual purchases
# BUSINESS JUSTIFICATION: Converts qualitative frequency to a comparable numeric signal
#                         for computing revenue potential.
freq_map = {
    'Weekly': 52,
    'Fortnightly': 26,
    'Bi-Weekly': 26,
    'Monthly': 12,
    'Quarterly': 4,
    'Every 3 Months': 4,
    'Annually': 1
}
df['purchase_freq_numeric'] = df['frequency_of_purchases'].map(freq_map)
unmapped_freq = df['purchase_freq_numeric'].isnull().sum()
if unmapped_freq > 0:
    print(f"WARNING: {unmapped_freq} frequency values not mapped")
print(f"\npurchase_freq_numeric distribution:\n{df['purchase_freq_numeric'].describe()}")
print("# BUSINESS JUSTIFICATION: purchase_freq_numeric — annual purchase rate proxy for revenue forecasting")

# estimated_clv_proxy
# FORMULA: purchase_amount_usd × purchase_freq_numeric
# NOTE: No timestamps available, so tenure-adjusted CLV is not possible.
#       This represents annualized revenue potential per customer.
# BUSINESS JUSTIFICATION: Approximates the annualized revenue a customer generates.
#                         Core input to value segmentation.
df['estimated_clv_proxy'] = df['purchase_amount_usd'] * df['purchase_freq_numeric']
print(f"\nestimated_clv_proxy distribution:\n{df['estimated_clv_proxy'].describe()}")
print("# BUSINESS JUSTIFICATION: estimated_clv_proxy — annualized revenue potential (purchase_amount × frequency)")

# value_composite (normalized composite score)
# FORMULA: average of normalized purchase_amount, purchase_freq, and previous_purchases
# BUSINESS JUSTIFICATION: Single composite score combining spend size, purchase cadence,
#                         and purchase history depth — avoids over-indexing on one dimension.
def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

df['value_composite'] = (
    normalize(df['purchase_amount_usd']) +
    normalize(df['purchase_freq_numeric']) +
    normalize(df['previous_purchases'])
) / 3

# value_tier — tertile cut on value_composite
# FORMULA: Low (bottom 33%) / Mid (33–66%) / High (top 33%)
# BUSINESS JUSTIFICATION: Segments customers into actionable buckets for targeted strategy.
#                         Tertiles ensure roughly equal group sizes for statistical reliability.
tertiles = df['value_composite'].quantile([0.33, 0.66])
df['value_tier'] = pd.cut(
    df['value_composite'],
    bins=[-np.inf, tertiles[0.33], tertiles[0.66], np.inf],
    labels=['Low', 'Mid', 'High']
)
print(f"\nvalue_tier distribution:\n{df['value_tier'].value_counts()}")
print("# BUSINESS JUSTIFICATION: value_tier — actionable 3-tier segmentation using spend, frequency, and history")

# ── SATISFACTION FEATURE ──────────────────────

# satisfaction_flag
# FORMULA: 1 if review_rating >= median (3.8), else 0
# BUSINESS JUSTIFICATION: High-rating customers are more likely to return organically.
#                         Used to separate genuinely satisfied customers from indifferent ones.
median_rating = df['review_rating'].median()
df['satisfaction_flag'] = (df['review_rating'] >= median_rating).astype(int)
print(f"\nsatisfaction_flag (threshold = {median_rating}):\n{df['satisfaction_flag'].value_counts()}")
print("# BUSINESS JUSTIFICATION: satisfaction_flag — proxy for organic retention likelihood")

# Subscription as loyalty signal
# FORMULA: 1 if subscription_status == 'Yes'
# BUSINESS JUSTIFICATION: Subscribers have explicitly opted in to a longer relationship.
#                         Useful cross-signal with promo dependency.
df['is_subscriber'] = (df['subscription_status'] == 'Yes').astype(int)
print(f"\nis_subscriber counts:\n{df['is_subscriber'].value_counts()}")
print("# BUSINESS JUSTIFICATION: is_subscriber — explicit commitment signal; reduces dependency on discounts")

# ── LOYALTY DEFINITION 1: FREQUENCY-BASED ────

# FORMULA: previous_purchases > median AND promo_code_used == 'No'
# LOGIC: A truly loyal customer comes back often AND doesn't need a discount to do so.
#        Above-median purchase history = proven repeat behavior.
#        No promo = organic motivation.
# ── LOYALTY DEFINITION 1: VALUE-BASED SCORE ──────────────
# Signals: tenure + spend + satisfaction
# BUSINESS JUSTIFICATION: Captures how long, how much, and how happy
# Each signal normalized 0-1 so no single column dominates

df['tenure_score']       = normalize(df['previous_purchases'])
df['spend_score']        = normalize(df['purchase_amount_usd'])
df['satisfaction_score'] = normalize(df['review_rating'])

df['loyalty_score_value'] = (
    df['tenure_score'] +
    df['spend_score'] +
    df['satisfaction_score']
) / 3

print(f"\n=== LOYALTY DEFINITION 1: VALUE-BASED SCORE ===")
print(f"Formula: (tenure + spend + satisfaction) / 3")
print(df['loyalty_score_value'].describe().round(4))
print(f"Avg score - Top value_tier: {df[df['value_tier']=='High']['loyalty_score_value'].mean():.4f}")
print(f"Avg score - Low value_tier: {df[df['value_tier']=='Low']['loyalty_score_value'].mean():.4f}")

# ── LOYALTY DEFINITION 2: BEHAVIOR-BASED SCORE ───────────
# Signals: frequency + tenure + promo independence
# BUSINESS JUSTIFICATION: Captures how often, how long, and whether
#   they buy without needing a discount — pure behavioral loyalty

df['frequency_score']    = normalize(df['purchase_freq_numeric'])
df['promo_independence'] = 1 - df['is_promo_dependent']  # flip: 1 = organic buyer

df['loyalty_score_behavior'] = (
    df['frequency_score'] +
    df['tenure_score'] +
    df['promo_independence']
) / 3

print(f"\n=== LOYALTY DEFINITION 2: BEHAVIOR-BASED SCORE ===")
print(f"Formula: (frequency + tenure + promo_independence) / 3")
print(df['loyalty_score_behavior'].describe().round(4))
print(f"Avg score - Organic buyers:    {df[df['is_promo_dependent']==0]['loyalty_score_behavior'].mean():.4f}")
print(f"Avg score - Promo buyers:      {df[df['is_promo_dependent']==1]['loyalty_score_behavior'].mean():.4f}")

# ── WINNER COMPARISON ─────────────────────────────────────
corr1 = df['loyalty_score_value'].corr(df['estimated_clv_proxy'])
corr2 = df['loyalty_score_behavior'].corr(df['estimated_clv_proxy'])

print(f"\n=== LOYALTY DEFINITION COMPARISON ===")
print(f"Def 1 (Value-Based) correlation with CLV:    {corr1:.4f}")
print(f"Def 2 (Behavior-Based) correlation with CLV: {corr2:.4f}")
print(f"\nDistribution comparison:")
print(df[['loyalty_score_value', 'loyalty_score_behavior']].describe().round(4))

winner = "loyalty_score_value" if abs(corr1) > abs(corr2) else "loyalty_score_behavior"
print(f"\n# WINNER: {winner}")
print("# ARGUMENT: state this after you see the actual correlation numbers from output")

# ── CUSTOMER SEGMENT LABEL ────────────────────

# FORMULA: value_tier × is_promo_dependent → 6 named segments
# BUSINESS JUSTIFICATION: Enables targeted strategy per segment.
#   High-Value Organic = protect and scale
#   High-Value Promo-Dependent = sunset discounts carefully
#   Mid-Value Organic = convert to High
#   Mid-Value Promo-Dependent = test discount reduction
#   Low-Value Organic = low priority but potential
#   Low-Value Promo-Dependent = deprioritize or exit

def segment_label(row):
    tier = str(row['value_tier'])
    promo = row['is_promo_dependent']
    labels = {
        ('High', 0): 'High-Value Organic',
        ('High', 1): 'High-Value Promo-Dependent',
        ('Mid',  0): 'Mid-Value Organic',
        ('Mid',  1): 'Mid-Value Promo-Dependent',
        ('Low',  0): 'Low-Value Organic',
        ('Low',  1): 'Low-Value Promo-Dependent',
    }
    return labels.get((tier, promo), 'Unknown')

df['customer_segment'] = df.apply(segment_label, axis=1)

print(f"\n=== CUSTOMER SEGMENT DISTRIBUTION ===")
seg_summary = df.groupby('customer_segment').agg(
    count=('customer_id', 'count'),
    avg_clv=('estimated_clv_proxy', 'mean'),
    avg_spend=('purchase_amount_usd', 'mean'),
    avg_rating=('review_rating', 'mean')
).sort_values('avg_clv', ascending=False)
print(seg_summary.round(2))
print("# BUSINESS JUSTIFICATION: customer_segment — 6-way actionable segmentation for targeted retention strategy")


# ─────────────────────────────────────────────
# STEP 4: VALIDATE
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4: VALIDATE")
print("=" * 60)

feature_cols = [
    'customer_id',
    'promo_dependency_score',
    'is_promo_dependent',
    'purchase_freq_numeric',
    'estimated_clv_proxy',
    'value_composite',
    'value_tier',
    'tenure_score',
    'spend_score',
    'satisfaction_score',
    'loyalty_score_value',
    'frequency_score',
    'promo_independence',
    'loyalty_score_behavior',
    'satisfaction_flag',
    'is_subscriber',
    'customer_segment'
]

feature_df = df[feature_cols].copy()

print("\nFeature table null check:")
null_counts = feature_df.isnull().sum()
print(null_counts)
assert null_counts.sum() == 0, "FAIL: Nulls found in feature table"
print("PASS: Zero nulls in feature table")

print("\nFeature table stats:")
print(feature_df.describe(include='all'))

print("\nEngineered features summary:")
print(f"  promo_dependency_score  — min: {df['promo_dependency_score'].min()}, max: {df['promo_dependency_score'].max()}, mean: {df['promo_dependency_score'].mean():.3f}")
print(f"  purchase_freq_numeric   — min: {df['purchase_freq_numeric'].min()}, max: {df['purchase_freq_numeric'].max()}, mean: {df['purchase_freq_numeric'].mean():.2f}")
print(f"  estimated_clv_proxy     — min: {df['estimated_clv_proxy'].min()}, max: {df['estimated_clv_proxy'].max()}, mean: {df['estimated_clv_proxy'].mean():.2f}")
print(f"  value_composite         — min: {df['value_composite'].min():.4f}, max: {df['value_composite'].max():.4f}, mean: {df['value_composite'].mean():.4f}")
print(f"  satisfaction_flag       — {df['satisfaction_flag'].sum()} satisfied ({df['satisfaction_flag'].mean()*100:.1f}%)")
print(f"  is_subscriber           — {df['is_subscriber'].sum()} subscribers ({df['is_subscriber'].mean()*100:.1f}%)")
print(f"  loyalty_score_value    — min: {df['loyalty_score_value'].min():.4f}, max: {df['loyalty_score_value'].max():.4f}, mean: {df['loyalty_score_value'].mean():.4f}")
print(f"  loyalty_score_behavior — min: {df['loyalty_score_behavior'].min():.4f}, max: {df['loyalty_score_behavior'].max():.4f}, mean: {df['loyalty_score_behavior'].mean():.4f}")
# ─────────────────────────────────────────────
# ── LOYALTY SCORE COMPARISON TABLE ───────────
loyalty_comparison = df.groupby('customer_segment').agg(
    count=('customer_id', 'count'),
    avg_loyalty_value_score=('loyalty_score_value', 'mean'),
    avg_loyalty_behavior_score=('loyalty_score_behavior', 'mean'),
    avg_clv=('estimated_clv_proxy', 'mean'),
    avg_spend=('purchase_amount_usd', 'mean'),
    promo_pct=('is_promo_dependent', 'mean')
).round(4).reset_index()

loyalty_comparison['promo_pct'] = (loyalty_comparison['promo_pct'] * 100).round(1)
loyalty_comparison = loyalty_comparison.sort_values('avg_clv', ascending=False)

print("\n=== LOYALTY SCORE COMPARISON BY SEGMENT ===")
print(loyalty_comparison.to_string(index=False))

corr1 = df['loyalty_score_value'].corr(df['estimated_clv_proxy'])
corr2 = df['loyalty_score_behavior'].corr(df['estimated_clv_proxy'])
print(f"\nDef 1 (Value-Based) correlation with CLV:    {corr1:.4f}")
print(f"Def 2 (Behavior-Based) correlation with CLV: {corr2:.4f}")
print(f"Winner: {'Def 1 Value-Based' if abs(corr1) > abs(corr2) else 'Def 2 Behavior-Based'}")

# STEP 5: EXPORT
# ─────────────────────────────────────────────

import openpyxl
df.to_csv('cleaned_dataset.csv', index=False)
feature_df.to_csv('customer_features.csv', index=False)

with pd.ExcelWriter('customer_analysis.xlsx', engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='Cleaned Full Dataset', index=False)
    feature_df.to_excel(writer, sheet_name='Customer Features', index=False)

    seg_summary = df.groupby('customer_segment').agg(
        count=('customer_id', 'count'),
        avg_clv=('estimated_clv_proxy', 'mean'),
        avg_spend=('purchase_amount_usd', 'mean'),
        avg_rating=('review_rating', 'mean'),
        promo_pct=('is_promo_dependent', 'mean')
    ).round(2).reset_index()
    seg_summary['promo_pct'] = (seg_summary['promo_pct'] * 100).round(1)
    seg_summary.to_excel(writer, sheet_name='Segment Summary', index=False)

print("DONE. Files saved: cleaned_dataset.csv, customer_features.csv, customer_analysis.xlsx")
loyalty_comparison.to_excel(writer, sheet_name='Loyalty Score Comparison', index=False)
