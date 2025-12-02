import kagglehub
import bz2
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Color map: ensure Positive -> green, Negative -> red
COLOR_MAP = {"Positive": "#51cf66", "Negative": "#ff6b6b"}

print("=" * 80)
print("EXPLORATORY DATA ANALYSIS - AMAZON REVIEWS DATASET")
print("=" * 80)

# --- Download dataset ---
print("\n[1/7] Downloading dataset...")
path = kagglehub.dataset_download("bittlingmayer/amazonreviews")
print(f"✓ Dataset path: {path}")

# --- Load the full dataset ---
print("\n[2/7] Loading dataset from train file...")
file_path = path + "/train.ft.txt.bz2"
with bz2.open(file_path, "rt", encoding="utf-8") as f:
    lines = [line.strip() for line in f]
print(f"✓ Loaded {len(lines):,} reviews")

# --- Create DataFrame and extract labels/reviews ---
print("\n[3/7] Preparing DataFrame...")
df = pd.DataFrame(lines, columns=["raw"])
df["__label__"] = df["raw"].str.extract(r"(__label__\d)")
df["review_text"] = df["raw"].str.replace(r"__label__\d\s", "", regex=True)
df["sentiment"] = df["__label__"].map({"__label__1": "Negative", "__label__2": "Positive"})

# Extract word counts and character counts
df["word_count"] = df["review_text"].apply(lambda x: len(re.findall(r"\b\w+\b", x)))
df["char_count"] = df["review_text"].apply(lambda x: len(x))
df["unique_words"] = df["review_text"].apply(lambda x: len(set(re.findall(r"\b\w+\b", x))))

print(f"✓ DataFrame shape: {df.shape}")
print(f"\nDataFrame preview:")
print(df.head(3))
print(f"\nColumn info:")
print(df.info())

# --- SECTION 1: Sentiment Distribution ---
print("\n" + "=" * 80)
print("SECTION 1: SENTIMENT DISTRIBUTION")
print("=" * 80)

sentiment_counts = df["sentiment"].value_counts()
print(f"\nSentiment Distribution:")
print(sentiment_counts)
print(f"\nPercentage Distribution:")
print(sentiment_counts / len(df) * 100)

# Pie chart
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
colors_for_pie = [COLOR_MAP.get(label, '#888888') for label in sentiment_counts.index]
ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', 
    colors=colors_for_pie, startangle=90, textprops={'fontsize': 12})
ax.set_title('Sentiment Distribution in Amazon Reviews', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_01_sentiment_distribution.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: eda_01_sentiment_distribution.png")
plt.close()

# --- SECTION 2: Review Length Analysis ---
print("\n" + "=" * 80)
print("SECTION 2: REVIEW LENGTH (WORD COUNT) ANALYSIS")
print("=" * 80)

print(f"\nWord Count Statistics:")
print(df["word_count"].describe())
print(f"\nSkewness: {df['word_count'].skew():.4f}")
print(f"Kurtosis: {df['word_count'].kurtosis():.4f}")

# Histogram
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.hist(df["word_count"], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax.set_xlabel('Word Count', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Review Lengths (Word Count)', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('eda_02_word_count_histogram.png', dpi=300, bbox_inches='tight')
print("✓ Saved: eda_02_word_count_histogram.png")
plt.close()

# Boxplot
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.boxplot(df["word_count"], vert=True, widths=0.5)
ax.set_ylabel('Word Count', fontsize=12)
ax.set_title('Boxplot of Review Lengths', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('eda_03_word_count_boxplot.png', dpi=300, bbox_inches='tight')
print("✓ Saved: eda_03_word_count_boxplot.png")
plt.close()

# --- SECTION 3: Word Count by Sentiment ---
print("\n" + "=" * 80)
print("SECTION 3: WORD COUNT BY SENTIMENT")
print("=" * 80)

print(f"\nWord Count Statistics by Sentiment:")
print(df.groupby("sentiment")["word_count"].describe())

# Boxplot by sentiment
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
sns.boxplot(x='sentiment', y='word_count', data=df, palette=COLOR_MAP, ax=ax)
ax.set_xlabel('Sentiment', fontsize=12)
ax.set_ylabel('Word Count', fontsize=12)
ax.set_title('Review Length Distribution by Sentiment', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('eda_04_word_count_by_sentiment_boxplot.png', dpi=300, bbox_inches='tight')
print("✓ Saved: eda_04_word_count_by_sentiment_boxplot.png")
plt.close()

# Violin plot (better for showing distribution shape)
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
sns.violinplot(x='sentiment', y='word_count', data=df, palette=COLOR_MAP, ax=ax)
ax.set_xlabel('Sentiment', fontsize=12)
ax.set_ylabel('Word Count', fontsize=12)
ax.set_title('Review Length Distribution by Sentiment (Violin Plot)', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('eda_05_word_count_by_sentiment_violin.png', dpi=300, bbox_inches='tight')
print("✓ Saved: eda_05_word_count_by_sentiment_violin.png")
plt.close()

# --- SECTION 4: Normality Tests ---
print("\n" + "=" * 80)
print("SECTION 4: NORMALITY TESTS")
print("=" * 80)

# Q-Q Plot
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
stats.probplot(df["word_count"], dist="norm", plot=ax)
ax.set_title('Q-Q Plot: Review Length vs Normal Distribution', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('eda_06_qq_plot.png', dpi=300, bbox_inches='tight')
print("✓ Saved: eda_06_qq_plot.png")
plt.close()

# Shapiro-Wilk test (on a sample to keep it fast)
sample_size = min(5000, len(df))
shapiro_stat, shapiro_p = stats.shapiro(df["word_count"].sample(sample_size, random_state=42))
print(f"\nShapiro-Wilk Test (sample of {sample_size:,}):")
print(f"  Statistic: {shapiro_stat:.6f}")
print(f"  P-value: {shapiro_p:.6e}")
print(f"  → Data is {'NOT ' if shapiro_p < 0.05 else ''}normally distributed (α=0.05)")

# --- SECTION 5: Character Count Analysis ---
print("\n" + "=" * 80)
print("SECTION 5: CHARACTER COUNT ANALYSIS")
print("=" * 80)

print(f"\nCharacter Count Statistics:")
print(df["char_count"].describe())

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.hist(df["char_count"], bins=50, color='coral', edgecolor='black', alpha=0.7)
ax.set_xlabel('Character Count', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Review Lengths (Character Count)', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('eda_07_char_count_histogram.png', dpi=300, bbox_inches='tight')
print("✓ Saved: eda_07_char_count_histogram.png")
plt.close()

# --- SECTION 6: Unique Words Analysis ---
print("\n" + "=" * 80)
print("SECTION 6: UNIQUE WORDS ANALYSIS")
print("=" * 80)

print(f"\nUnique Words Statistics:")
print(df["unique_words"].describe())

# Calculate lexical diversity (unique words / total words)
df["lexical_diversity"] = df["unique_words"] / df["word_count"]

print(f"\nLexical Diversity (unique words / total words) Statistics:")
print(df["lexical_diversity"].describe())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Unique words histogram
ax1.hist(df["unique_words"], bins=50, color='mediumseagreen', edgecolor='black', alpha=0.7)
ax1.set_xlabel('Unique Word Count', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('Distribution of Unique Words per Review', fontsize=13, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Lexical diversity histogram
ax2.hist(df["lexical_diversity"], bins=50, color='orchid', edgecolor='black', alpha=0.7)
ax2.set_xlabel('Lexical Diversity Ratio', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.set_title('Distribution of Lexical Diversity', fontsize=13, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('eda_08_unique_words_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: eda_08_unique_words_analysis.png")
plt.close()

# --- SECTION 7: Correlation Analysis ---
print("\n" + "=" * 80)
print("SECTION 7: CORRELATION ANALYSIS")
print("=" * 80)

# Create numeric sentiment column for correlation
df["sentiment_num"] = (df["sentiment"] == "Positive").astype(int)

correlation_data = df[["word_count", "char_count", "unique_words", "lexical_diversity", "sentiment_num"]]
corr_matrix = correlation_data.corr()

print(f"\nCorrelation Matrix:")
print(corr_matrix)

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0, 
            square=True, ax=ax, cbar_kws={'label': 'Correlation'})
ax.set_title('Correlation Matrix: Review Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_09_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: eda_09_correlation_heatmap.png")
plt.close()

# --- SECTION 8: Comparative Statistics ---
print("\n" + "=" * 80)
print("SECTION 8: COMPARATIVE STATISTICS BY SENTIMENT")
print("=" * 80)

comparative_stats = df.groupby("sentiment")[["word_count", "char_count", "unique_words", "lexical_diversity"]].describe().T
print(f"\nDetailed statistics by sentiment:")
print(comparative_stats)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

metrics = ["word_count", "char_count", "unique_words", "lexical_diversity"]
titles = ["Word Count", "Character Count", "Unique Words", "Lexical Diversity"]

for idx, (ax, metric, title) in enumerate(zip(axes.flat, metrics, titles)):
    sns.boxplot(x='sentiment', y=metric, data=df, palette=COLOR_MAP, ax=ax)
    ax.set_xlabel('Sentiment', fontsize=11)
    ax.set_ylabel(title, fontsize=11)
    ax.set_title(f'{title} by Sentiment', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('eda_10_comparative_metrics.png', dpi=300, bbox_inches='tight')
print("✓ Saved: eda_10_comparative_metrics.png")
plt.close()

# --- SECTION 9: Sample Reviews ---
print("\n" + "=" * 80)
print("SECTION 9: SAMPLE REVIEWS BY SENTIMENT")
print("=" * 80)

for sentiment in ["Positive", "Negative"]:
    print(f"\n{sentiment} Reviews (first 2 samples):")
    reviews = df[df["sentiment"] == sentiment]["review_text"].head(2).tolist()
    for i, review in enumerate(reviews, 1):
        preview = review[:100] + "..." if len(review) > 100 else review
        print(f"  {i}. {preview}")

# --- Summary Statistics Report ---
print("\n" + "=" * 80)
print("SUMMARY STATISTICS REPORT")
print("=" * 80)

summary = {
    'Total Reviews': len(df),
    'Positive Reviews': (df['sentiment'] == 'Positive').sum(),
    'Negative Reviews': (df['sentiment'] == 'Negative').sum(),
    'Avg Words per Review': f"{df['word_count'].mean():.2f}",
    'Avg Characters per Review': f"{df['char_count'].mean():.2f}",
    'Avg Unique Words per Review': f"{df['unique_words'].mean():.2f}",
    'Avg Lexical Diversity': f"{df['lexical_diversity'].mean():.3f}",
    'Min Word Count': df['word_count'].min(),
    'Max Word Count': df['word_count'].max(),
    'Median Word Count': df['word_count'].median(),
    'Std Dev Word Count': f"{df['word_count'].std():.2f}",
}

print("\n")
for key, value in summary.items():
    print(f"{key:<30}: {value}")

# --- Save summary to CSV ---
print("\n" + "=" * 80)
print("EXPORTING RESULTS")
print("=" * 80)

# Save summary statistics
summary_df = pd.DataFrame({
    'Metric': list(summary.keys()),
    'Value': list(summary.values())
})
summary_df.to_csv('eda_summary_statistics.csv', index=False)
print("\n✓ Saved: eda_summary_statistics.csv")

df.to_csv('eda_detailed_dataset.csv', index=False)
print("✓ Saved: eda_detailed_dataset.csv")

print("\n" + "=" * 80)
print("EDA COMPLETE!")
print("=" * 80)
print("\nGenerated Files:")
print("  Visualizations:")
print("    - eda_01_sentiment_distribution.png")
print("    - eda_02_word_count_histogram.png")
print("    - eda_03_word_count_boxplot.png")
print("    - eda_04_word_count_by_sentiment_boxplot.png")
print("    - eda_05_word_count_by_sentiment_violin.png")
print("    - eda_06_qq_plot.png")
print("    - eda_07_char_count_histogram.png")
print("    - eda_08_unique_words_analysis.png")
print("    - eda_09_correlation_heatmap.png")
print("    - eda_10_comparative_metrics.png")
print("  Data Files:")
print("    - eda_summary_statistics.csv")
print("    - eda_detailed_dataset.csv")
print("\n" + "=" * 80)
