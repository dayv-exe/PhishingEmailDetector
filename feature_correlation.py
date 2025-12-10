import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

# Load the feature-engineered dataset
try:
    df = pd.read_csv('cleaned_datasets/FeatureEngineered_PhishingEmailData.csv')
    print("✓ Feature-engineered dataset loaded successfully")
    print(f"Records: {len(df)}")
except FileNotFoundError:
    print("Error: FeatureEngineered_PhishingEmailData.csv not found!")
    print("Please run featureengineering.py first to generate the dataset.")
    exit(1)

# ==================== URL COUNT vs PHISHING CORRELATION ANALYSIS ====================
print("\n" + "=" * 60)
print("URL COUNT vs PHISHING CORRELATION ANALYSIS")
print("=" * 60)

if 'urls' in df.columns and 'label' in df.columns:
    # Calculate correlation between urls count and label
    correlation = df['urls'].corr(df['label'])
    print(f"\n✓ Correlation between URL count and label (phishing): {correlation:.4f}")

    # Basic statistics
    print("\n✓ URL Count Statistics:")
    print(f"  • Overall mean URL count: {df['urls'].mean():.2f}")
    print(f"  • Overall median URL count: {df['urls'].median():.2f}")
    print(f"  • Maximum URLs in a single email: {df['urls'].max()}")

    # Statistics by label
    legit_urls = df[df['label'] == 0]['urls']
    phish_urls = df[df['label'] == 1]['urls']

    print(f"\n✓ Legitimate Emails:")
    print(f"  • Mean URL count: {legit_urls.mean():.2f}")
    print(f"  • Median URL count: {legit_urls.median():.2f}")
    print(f"  • Max URL count: {legit_urls.max()}")
    print(f"  • Std deviation: {legit_urls.std():.2f}")

    print(f"\n✓ Phishing Emails:")
    print(f"  • Mean URL count: {phish_urls.mean():.2f}")
    print(f"  • Median URL count: {phish_urls.median():.2f}")
    print(f"  • Max URL count: {phish_urls.max()}")
    print(f"  • Std deviation: {phish_urls.std():.2f}")

    # Distribution by URL count
    print("\n✓ Distribution by URL Count:")
    url_stats = df.groupby(['urls', 'label']).size().unstack(fill_value=0)
    url_stats['Total'] = url_stats.sum(axis=1)
    url_stats['Phishing_Rate'] = (url_stats[1] / url_stats['Total'] * 100)

    # Show stats for URL counts up to 10 (or max if less)
    max_display = min(11, df['urls'].max() + 1)
    for url_count in range(max_display):
        if url_count in url_stats.index:
            legit = url_stats.loc[url_count, 0]
            phish = url_stats.loc[url_count, 1]
            total = url_stats.loc[url_count, 'Total']
            rate = url_stats.loc[url_count, 'Phishing_Rate']
            print(f"  • {url_count} URLs: {total} emails ({legit} legitimate, {phish} phishing) - {rate:.2f}% phishing")

    if df['urls'].max() >= max_display:
        print(f"  • {max_display}+ URLs: (aggregated in visualizations)")

    # Calculate emails with/without URLs
    emails_with_urls = df[df['urls'] > 0].shape[0]
    emails_without_urls = df[df['urls'] == 0].shape[0]

    phishing_with_urls = df[(df['label'] == 1) & (df['urls'] > 0)].shape[0]
    phishing_without_urls = df[(df['label'] == 1) & (df['urls'] == 0)].shape[0]
    total_phishing = df[df['label'] == 1].shape[0]

    legit_with_urls = df[(df['label'] == 0) & (df['urls'] > 0)].shape[0]
    legit_without_urls = df[(df['label'] == 0) & (df['urls'] == 0)].shape[0]
    total_legit = df[df['label'] == 0].shape[0]

    print(f"\n✓ Emails with/without URLs:")
    print(
        f"  • Phishing emails with URLs: {phishing_with_urls}/{total_phishing} ({phishing_with_urls / total_phishing * 100:.2f}%)")
    print(
        f"  • Phishing emails without URLs: {phishing_without_urls}/{total_phishing} ({phishing_without_urls / total_phishing * 100:.2f}%)")
    print(
        f"  • Legitimate emails with URLs: {legit_with_urls}/{total_legit} ({legit_with_urls / total_legit * 100:.2f}%)")
    print(
        f"  • Legitimate emails without URLs: {legit_without_urls}/{total_legit} ({legit_without_urls / total_legit * 100:.2f}%)")

    # Visualization 1: Box Plot Comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    data_to_plot = [legit_urls, phish_urls]
    box = ax.boxplot(data_to_plot, tick_labels=['Legitimate', 'Phishing'],
                     patch_artist=True, showmeans=True)

    # Color the boxes
    colors = ['#2ecc71', '#e74c3c']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_title("URL Count Distribution: Legitimate vs Phishing Emails",
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel("Number of URLs", fontsize=11)
    ax.set_xlabel("Email Type", fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Visualization 2: Bar Chart - Distribution by URL Count (grouped)
    fig, ax = plt.subplots(figsize=(12, 6))

    # Group URL counts for better visualization
    max_individual = 10
    url_groups = []

    for i in range(max_individual + 1):
        if i in url_stats.index:
            url_groups.append({
                'count': str(i),
                'legitimate': url_stats.loc[i, 0],
                'phishing': url_stats.loc[i, 1]
            })

    # Add 10+ group if there are higher counts
    if df['urls'].max() > max_individual:
        over_10_legit = df[(df['label'] == 0) & (df['urls'] > max_individual)].shape[0]
        over_10_phish = df[(df['label'] == 1) & (df['urls'] > max_individual)].shape[0]
        url_groups.append({
            'count': f'{max_individual}+',
            'legitimate': over_10_legit,
            'phishing': over_10_phish
        })

    url_df = pd.DataFrame(url_groups)
    x = range(len(url_df))
    width = 0.35

    bars1 = ax.bar([i - width / 2 for i in x], url_df['legitimate'], width,
                   label='Legitimate', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar([i + width / 2 for i in x], url_df['phishing'], width,
                   label='Phishing', color='#e74c3c', alpha=0.8)

    ax.set_title("Email Distribution by URL Count", fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel("Number of Emails", fontsize=11)
    ax.set_xlabel("Number of URLs", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(url_df['count'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Visualization 3: Phishing Rate by URL Count
    fig, ax = plt.subplots(figsize=(12, 6))

    # Use URL counts that have enough data (at least 10 emails)
    significant_counts = url_stats[url_stats['Total'] >= 10].copy()

    if len(significant_counts) > 0:
        x = range(len(significant_counts))
        counts = significant_counts.index.tolist()
        rates = significant_counts['Phishing_Rate'].tolist()

        bars = ax.bar(x, rates, color='#e74c3c', alpha=0.8, width=0.6)

        # Color bars based on phishing rate
        mean_rate = sum(rates) / len(rates)
        for bar, rate in zip(bars, rates):
            if rate > mean_rate * 1.5:
                bar.set_color('#c0392b')  # Dark red for very high risk
            elif rate > mean_rate:
                bar.set_color('#e74c3c')  # Red for above average
            else:
                bar.set_color('#3498db')  # Blue for below average

        ax.set_title("Phishing Rate by URL Count (min 10 emails per group)",
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel("Phishing Rate (%)", fontsize=11)
        ax.set_xlabel("Number of URLs", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(counts)
        ax.axhline(y=mean_rate, color='black', linestyle='--', linewidth=1, alpha=0.5,
                   label=f'Average ({mean_rate:.1f}%)')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.show()
    else:
        print("\n⚠ Not enough data to create phishing rate by URL count chart")

    # Visualization 4: Stacked Bar - With/Without URLs
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ['Legitimate', 'Phishing']
    with_urls = [legit_with_urls, phishing_with_urls]
    without_urls = [legit_without_urls, phishing_without_urls]

    x = range(len(categories))
    width = 0.6

    bars1 = ax.bar(x, without_urls, width, label='Without URLs (0)', color='#95a5a6', alpha=0.8)
    bars2 = ax.bar(x, with_urls, width, bottom=without_urls, label='With URLs (1+)',
                   color='#3498db', alpha=0.8)

    ax.set_title("URL Presence in Legitimate vs Phishing Emails",
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel("Number of Emails", fontsize=11)
    ax.set_xlabel("Email Type", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add percentage labels
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        total = height1 + height2

        if height1 > 0:
            pct1 = (height1 / total * 100)
            ax.text(bar1.get_x() + bar1.get_width() / 2., height1 / 2,
                    f'{pct1:.1f}%', ha='center', va='center', fontweight='bold', color='white')

        if height2 > 0:
            pct2 = (height2 / total * 100)
            ax.text(bar2.get_x() + bar2.get_width() / 2., height1 + height2 / 2,
                    f'{pct2:.1f}%', ha='center', va='center', fontweight='bold', color='white')

    plt.tight_layout()
    plt.show()

    # Visualization 5: Histogram Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Limit to reasonable range for visualization
    max_hist = min(20, df['urls'].max())

    ax1.hist(legit_urls, bins=range(0, max_hist + 2), color='#2ecc71', alpha=0.7, edgecolor='black')
    ax1.set_title("Legitimate Emails: URL Count Distribution", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Number of URLs", fontsize=10)
    ax1.set_ylabel("Frequency", fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.axvline(legit_urls.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {legit_urls.mean():.2f}')
    ax1.legend()

    ax2.hist(phish_urls, bins=range(0, max_hist + 2), color='#e74c3c', alpha=0.7, edgecolor='black')
    ax2.set_title("Phishing Emails: URL Count Distribution", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Number of URLs", fontsize=10)
    ax2.set_ylabel("Frequency", fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    ax2.axvline(phish_urls.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {phish_urls.mean():.2f}')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # Statistical tests
    from scipy.stats import mannwhitneyu, ttest_ind

    # T-test
    t_stat, t_pvalue = ttest_ind(legit_urls, phish_urls)
    print(f"\n✓ Independent T-Test Results:")
    print(f"  • T-statistic: {t_stat:.4f}")
    print(f"  • P-value: {t_pvalue:.10e}")

    # Mann-Whitney U test (non-parametric alternative)
    u_stat, u_pvalue = mannwhitneyu(legit_urls, phish_urls, alternative='two-sided')
    print(f"\n✓ Mann-Whitney U Test Results:")
    print(f"  • U-statistic: {u_stat:.4f}")
    print(f"  • P-value: {u_pvalue:.10e}")

    if t_pvalue < 0.05:
        print(f"\n  • Result: Statistically significant difference (p < 0.05)")
        print(f"    URL count differs significantly between legitimate and phishing emails.")
        if phish_urls.mean() > legit_urls.mean():
            print(f"    Phishing emails tend to have MORE URLs on average.")
        else:
            print(f"    Phishing emails tend to have FEWER URLs on average.")
    else:
        print(f"\n  • Result: No statistically significant difference (p >= 0.05)")

    # Risk analysis
    print(f"\n✓ Risk Analysis by URL Count:")
    high_risk = url_stats[url_stats['Phishing_Rate'] > 50].copy()
    if len(high_risk) > 0:
        print(f"  • High risk URL counts (>50% phishing):")
        for count, row in high_risk.iterrows():
            print(f"    - {count} URLs: {row['Phishing_Rate']:.2f}% phishing ({row[1]}/{row['Total']} emails)")
    else:
        print(f"  • No URL counts with >50% phishing rate")

    low_risk = url_stats[url_stats['Phishing_Rate'] < 25].copy()
    if len(low_risk) > 0:
        print(f"  • Low risk URL counts (<25% phishing):")
        for count, row in low_risk.iterrows():
            print(f"    - {count} URLs: {row['Phishing_Rate']:.2f}% phishing ({row[1]}/{row['Total']} emails)")

else:
    print("\n⚠ Warning: 'urls' or 'label' column not found in dataset")
    print("Please ensure both columns exist to perform correlation analysis.")

print("\n" + "=" * 60)
print("URL COUNT ANALYSIS COMPLETE")
print("=" * 60)

# ==================== TOP LEVEL DOMAIN vs PHISHING CORRELATION ANALYSIS ====================
print("\n" + "=" * 60)
print("TOP LEVEL DOMAIN vs PHISHING CORRELATION ANALYSIS")
print("=" * 60)

if 'top_level_domain' in df.columns and 'label' in df.columns:

    # Get TLD distribution by label
    print("\n✓ Top 15 TLDs in Phishing Emails:")
    phishing_tlds = df[df['label'] == 1]['top_level_domain'].value_counts().head(15)
    for tld, count in phishing_tlds.items():
        pct = (count / df[df['label'] == 1].shape[0]) * 100
        print(f"  • .{tld}: {count} ({pct:.2f}%)")

    print("\n✓ Top 15 TLDs in Legitimate Emails:")
    legit_tlds = df[df['label'] == 0]['top_level_domain'].value_counts().head(15)
    for tld, count in legit_tlds.items():
        pct = (count / df[df['label'] == 0].shape[0]) * 100
        print(f"  • .{tld}: {count} ({pct:.2f}%)")

    # Calculate phishing rate by TLD (for TLDs with at least 50 emails)
    tld_stats = df.groupby('top_level_domain').agg({
        'label': ['sum', 'count', 'mean']
    }).reset_index()
    tld_stats.columns = ['tld', 'phishing_count', 'total_count', 'phishing_rate']
    tld_stats = tld_stats.sort_values('phishing_rate', ascending=False)

    print(f"\n✓ Top 15 Most Suspicious TLDs (Highest Phishing Rate, min 50 emails):")
    for idx, row in tld_stats.head(15).iterrows():
        print(
            f"  • .{row['tld']}: {row['phishing_rate'] * 100:.2f}% phishing ({row['phishing_count']:.0f}/{row['total_count']:.0f})")

    print(f"\n✓ Top 15 Most Trustworthy TLDs (Lowest Phishing Rate, min 50 emails):")
    for idx, row in tld_stats.tail(15).iterrows():
        print(
            f"  • .{row['tld']}: {row['phishing_rate'] * 100:.2f}% phishing ({row['phishing_count']:.0f}/{row['total_count']:.0f})")

    # Visualization 1: Top 10 TLDs by Volume (Stacked Bar)
    fig, ax = plt.subplots(figsize=(12, 6))

    top_tlds = df['top_level_domain'].value_counts().head(10).index
    tld_data = []
    for tld in top_tlds:
        legit = df[(df['top_level_domain'] == tld) & (df['label'] == 0)].shape[0]
        phish = df[(df['top_level_domain'] == tld) & (df['label'] == 1)].shape[0]
        tld_data.append({'tld': tld, 'legitimate': legit, 'phishing': phish})

    tld_df = pd.DataFrame(tld_data)
    x = range(len(tld_df))
    width = 0.6

    bars1 = ax.bar(x, tld_df['legitimate'], width, label='Legitimate', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x, tld_df['phishing'], width, bottom=tld_df['legitimate'],
                   label='Phishing', color='#e74c3c', alpha=0.8)

    ax.set_title("Top 10 Most Common TLDs: Legitimate vs Phishing", fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel("Number of Emails", fontsize=11)
    ax.set_xlabel("Top-Level Domain", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f".{tld}" for tld in tld_df['tld']], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Visualization 2: Phishing Rate by Top TLDs
    fig, ax = plt.subplots(figsize=(12, 6))

    top_15_suspicious = tld_stats.head(15)
    x = range(len(top_15_suspicious))

    bars = ax.bar(x, top_15_suspicious['phishing_rate'] * 100, color='#e74c3c', alpha=0.8)

    # Color bars based on phishing rate
    for i, (bar, rate) in enumerate(zip(bars, top_15_suspicious['phishing_rate'])):
        if rate > 0.5:
            bar.set_color('#c0392b')  # Dark red for very suspicious
        elif rate > 0.3:
            bar.set_color('#e74c3c')  # Red for suspicious
        else:
            bar.set_color('#f39c12')  # Orange for moderate

    ax.set_title("Top 15 Most Suspicious TLDs by Phishing Rate", fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel("Phishing Rate (%)", fontsize=11)
    ax.set_xlabel("Top-Level Domain", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f".{tld}" for tld in top_15_suspicious['tld']], rotation=45, ha='right')
    ax.axhline(y=50, color='black', linestyle='--', linewidth=1, alpha=0.5, label='50% threshold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (bar, row) in enumerate(zip(bars, top_15_suspicious.itertuples())):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()

    # Visualization 3: Heatmap of TLD distribution
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get top 20 TLDs and create pivot table
    top_20_tlds = df['top_level_domain'].value_counts().head(20).index
    pivot_data = df[df['top_level_domain'].isin(top_20_tlds)].groupby(
        ['top_level_domain', 'label']
    ).size().unstack(fill_value=0)

    # Normalize by row to show percentage
    pivot_normalized = pivot_data.div(pivot_data.sum(axis=1), axis=0) * 100

    sns.heatmap(pivot_normalized, annot=True, fmt='.1f', cmap='RdYlGn_r',
                cbar_kws={'label': 'Percentage (%)'}, ax=ax, linewidths=1, linecolor='black')

    ax.set_title("Top 20 TLDs: Distribution by Email Type", fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel("Top-Level Domain", fontsize=11)
    ax.set_xlabel("Label", fontsize=11)
    ax.set_xticklabels(['Legitimate', 'Phishing'], rotation=0)
    ax.set_yticklabels([f".{tld}" for tld in pivot_normalized.index], rotation=0)

    plt.tight_layout()
    plt.show()

    # Chi-square test for TLD independence
    ct = pd.crosstab(df['top_level_domain'], df['label'])
    chi2, p_value, dof, expected = chi2_contingency(ct)

    print(f"\n✓ Chi-square Test Results (TLD vs Phishing):")
    print(f"  • Chi-square statistic: {chi2:.4f}")
    print(f"  • P-value: {p_value:.10e}")
    print(f"  • Degrees of freedom: {dof}")

    if p_value < 0.05:
        print(f"  • Result: Statistically significant relationship (p < 0.05)")
        print(f"    Top-level domain is significantly related to phishing emails.")
    else:
        print(f"  • Result: No statistically significant relationship (p >= 0.05)")

else:
    print("\n⚠ Warning: 'top_level_domain' or 'label' column not found in dataset")

# ==================== DAY OF WEEK vs PHISHING CORRELATION ANALYSIS ====================
print("\n" + "=" * 60)
print("DAY OF WEEK vs PHISHING CORRELATION ANALYSIS")
print("=" * 60)

if 'day_of_week' in df.columns and 'day_name' in df.columns and 'label' in df.columns:

    # Get distribution by day and label, sorted by day_of_week number
    print("\n✓ Email Distribution by Day of Week:")
    day_stats = df.groupby(['day_of_week', 'day_name', 'label']).size().unstack(fill_value=0)
    day_stats = day_stats.sort_index(level='day_of_week')
    day_stats['Total'] = day_stats.sum(axis=1)
    day_stats['Phishing_Rate'] = (day_stats[1] / day_stats['Total'] * 100)

    for (day_num, day_name), row in day_stats.iterrows():
        legit = row[0]
        phish = row[1]
        total = row['Total']
        rate = row['Phishing_Rate']
        print(f"  • {day_name}: {total} emails ({legit} legitimate, {phish} phishing) - {rate:.2f}% phishing")

    # Calculate correlation using the numeric day_of_week column
    correlation = df['day_of_week'].corr(df['label'])
    print(f"\n✓ Correlation between day_of_week (numeric) and label: {correlation:.4f}")

    # Visualization 1: Stacked Bar Chart - Email Volume by Day
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create dataframe sorted by day_of_week number
    day_data = []
    for (day_num, day_name), row in day_stats.iterrows():
        day_data.append({
            'day_num': day_num,
            'day_name': day_name,
            'legitimate': row[0],
            'phishing': row[1]
        })

    day_df = pd.DataFrame(day_data)
    x = range(len(day_df))
    width = 0.6

    bars1 = ax.bar(x, day_df['legitimate'], width, label='Legitimate', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x, day_df['phishing'], width, bottom=day_df['legitimate'],
                   label='Phishing', color='#e74c3c', alpha=0.8)

    ax.set_title("Email Distribution by Day of Week: Legitimate vs Phishing",
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel("Number of Emails", fontsize=11)
    ax.set_xlabel("Day of Week", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(day_df['day_name'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add percentage labels
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        total = height1 + height2

        if total > 0:
            pct_phishing = (height2 / total * 100)
            # Add label at top of stacked bar
            ax.text(bar2.get_x() + bar2.get_width() / 2., height1 + height2 + 5,
                    f'{pct_phishing:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.show()

    # Visualization 2: Phishing Rate by Day of Week
    fig, ax = plt.subplots(figsize=(12, 6))

    x = range(len(day_df))
    phishing_rates = [(row['phishing'] / (row['legitimate'] + row['phishing']) * 100)
                      for _, row in day_df.iterrows()]

    bars = ax.bar(x, phishing_rates, color='#e74c3c', alpha=0.8, width=0.6)

    # Color bars based on phishing rate
    mean_rate = sum(phishing_rates) / len(phishing_rates)
    for bar, rate in zip(bars, phishing_rates):
        if rate > mean_rate:
            bar.set_color('#c0392b')  # Dark red for above average
        else:
            bar.set_color('#3498db')  # Blue for below average

    ax.set_title("Phishing Rate by Day of Week", fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel("Phishing Rate (%)", fontsize=11)
    ax.set_xlabel("Day of Week", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(day_df['day_name'], rotation=45, ha='right')
    ax.axhline(y=mean_rate, color='black', linestyle='--', linewidth=1, alpha=0.5,
               label=f'Average ({mean_rate:.1f}%)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, rate in zip(bars, phishing_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.show()

    # Visualization 3: Side-by-Side Comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    x = range(len(day_df))
    width = 0.35

    bars1 = ax.bar([i - width / 2 for i in x], day_df['legitimate'], width,
                   label='Legitimate', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar([i + width / 2 for i in x], day_df['phishing'], width,
                   label='Phishing', color='#e74c3c', alpha=0.8)

    ax.set_title("Legitimate vs Phishing Emails by Day of Week (Side-by-Side)",
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel("Number of Emails", fontsize=11)
    ax.set_xlabel("Day of Week", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(day_df['day_name'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 5,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()

    # Visualization 4: Heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create pivot table and normalize, using day_name for display
    pivot_data = df.groupby(['day_of_week', 'day_name', 'label']).size().unstack(fill_value=0)
    pivot_data = pivot_data.sort_index(level='day_of_week')
    # Reset index to use day_name for the heatmap
    pivot_data.index = pivot_data.index.droplevel('day_of_week')
    pivot_normalized = pivot_data.div(pivot_data.sum(axis=1), axis=0) * 100

    sns.heatmap(pivot_normalized, annot=True, fmt='.1f', cmap='RdYlGn_r',
                cbar_kws={'label': 'Percentage (%)'}, ax=ax, linewidths=1, linecolor='black')

    ax.set_title("Email Type Distribution by Day of Week (Percentage Heatmap)",
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel("Day of Week", fontsize=11)
    ax.set_xlabel("Label", fontsize=11)
    ax.set_xticklabels(['Legitimate', 'Phishing'], rotation=0)
    ax.set_yticklabels(pivot_normalized.index, rotation=0)

    plt.tight_layout()
    plt.show()

    # Chi-square test for independence
    ct = pd.crosstab(df['day_of_week'], df['label'])
    chi2, p_value, dof, expected = chi2_contingency(ct)

    print(f"\n✓ Chi-square Test Results (Day of Week vs Phishing):")
    print(f"  • Chi-square statistic: {chi2:.4f}")
    print(f"  • P-value: {p_value:.10e}")
    print(f"  • Degrees of freedom: {dof}")

    if p_value < 0.05:
        print(f"  • Result: Statistically significant relationship (p < 0.05)")
        print(f"    Day of week is significantly related to phishing emails.")
    else:
        print(f"  • Result: No statistically significant relationship (p >= 0.05)")

    # Find highest and lowest risk days
    max_idx = day_stats['Phishing_Rate'].idxmax()
    min_idx = day_stats['Phishing_Rate'].idxmin()
    max_day_name = max_idx[1]  # Get day_name from tuple index
    min_day_name = min_idx[1]
    print(f"\n✓ Risk Assessment:")
    print(f"  • Highest risk day: {max_day_name} ({day_stats.loc[max_idx, 'Phishing_Rate']:.2f}% phishing)")
    print(f"  • Lowest risk day: {min_day_name} ({day_stats.loc[min_idx, 'Phishing_Rate']:.2f}% phishing)")

else:
    print("\n⚠ Warning: 'day_of_week', 'day_name', or 'label' column not found in dataset")
    print("Please ensure all columns exist to perform correlation analysis.")

print("\n" + "=" * 60)
print("DAY OF WEEK ANALYSIS COMPLETE")
print("=" * 60)

# ==================== HOUR OF DAY vs PHISHING CORRELATION ANALYSIS ====================
print("\n" + "=" * 60)
print("HOUR OF DAY vs PHISHING CORRELATION ANALYSIS")
print("=" * 60)

if 'hour' in df.columns and 'label' in df.columns:

    # Get distribution by hour and label
    print("\n✓ Email Distribution by Hour of Day:")
    hour_stats = df.groupby(['hour', 'label']).size().unstack(fill_value=0)
    hour_stats = hour_stats.sort_index()
    hour_stats['Total'] = hour_stats.sum(axis=1)
    hour_stats['Phishing_Rate'] = (hour_stats[1] / hour_stats['Total'] * 100)

    # Print statistics for each hour
    for hour in range(24):
        if hour in hour_stats.index:
            legit = hour_stats.loc[hour, 0]
            phish = hour_stats.loc[hour, 1]
            total = hour_stats.loc[hour, 'Total']
            rate = hour_stats.loc[hour, 'Phishing_Rate']
            print(f"  • {hour:02d}:00 - {legit} legitimate, {phish} phishing ({rate:.2f}% phishing, {total} total)")

    # Calculate correlation
    correlation = df['hour'].corr(df['label'])
    print(f"\n✓ Correlation between hour and label: {correlation:.4f}")

    # Time period breakdown
    print("\n✓ Email Distribution by Time Period:")


    def get_time_period(hour):
        if 0 <= hour < 6:
            return 'Night (00:00-05:59)'
        elif 6 <= hour < 12:
            return 'Morning (06:00-11:59)'
        elif 12 <= hour < 18:
            return 'Afternoon (12:00-17:59)'
        else:
            return 'Evening (18:00-23:59)'


    df_temp = df.copy()
    df_temp['time_period'] = df_temp['hour'].apply(get_time_period)

    period_order = ['Night (00:00-05:59)', 'Morning (06:00-11:59)',
                    'Afternoon (12:00-17:59)', 'Evening (18:00-23:59)']

    for period in period_order:
        period_data = df_temp[df_temp['time_period'] == period]
        total = len(period_data)
        legit = len(period_data[period_data['label'] == 0])
        phish = len(period_data[period_data['label'] == 1])
        rate = (phish / total * 100) if total > 0 else 0
        print(f"  • {period}: {total} emails ({legit} legitimate, {phish} phishing) - {rate:.2f}% phishing")

    # Visualization 1: Stacked Bar Chart - Email Volume by Hour
    fig, ax = plt.subplots(figsize=(14, 6))

    hours = sorted(hour_stats.index)
    legitimate = [hour_stats.loc[h, 0] for h in hours]
    phishing = [hour_stats.loc[h, 1] for h in hours]

    x = range(len(hours))
    width = 0.8

    bars1 = ax.bar(x, legitimate, width, label='Legitimate', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x, phishing, width, bottom=legitimate,
                   label='Phishing', color='#e74c3c', alpha=0.8)

    ax.set_title("Email Distribution by Hour of Day: Legitimate vs Phishing",
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel("Number of Emails", fontsize=11)
    ax.set_xlabel("Hour of Day", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h:02d}:00" for h in hours], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add shaded regions for time periods
    ax.axvspan(-0.5, 5.5, alpha=0.1, color='blue', label='Night')
    ax.axvspan(5.5, 11.5, alpha=0.1, color='yellow', label='Morning')
    ax.axvspan(11.5, 17.5, alpha=0.1, color='orange', label='Afternoon')
    ax.axvspan(17.5, 23.5, alpha=0.1, color='purple', label='Evening')

    plt.tight_layout()
    plt.show()

    # Visualization 2: Phishing Rate by Hour
    fig, ax = plt.subplots(figsize=(14, 6))

    phishing_rates = [hour_stats.loc[h, 'Phishing_Rate'] for h in hours]

    bars = ax.bar(x, phishing_rates, color='#e74c3c', alpha=0.8, width=0.8)

    # Color bars based on phishing rate
    mean_rate = sum(phishing_rates) / len(phishing_rates)
    for bar, rate in zip(bars, phishing_rates):
        if rate > mean_rate * 1.2:
            bar.set_color('#c0392b')  # Dark red for high risk
        elif rate > mean_rate:
            bar.set_color('#e74c3c')  # Red for above average
        else:
            bar.set_color('#3498db')  # Blue for below average

    ax.set_title("Phishing Rate by Hour of Day", fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel("Phishing Rate (%)", fontsize=11)
    ax.set_xlabel("Hour of Day", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h:02d}:00" for h in hours], rotation=45, ha='right')
    ax.axhline(y=mean_rate, color='black', linestyle='--', linewidth=1, alpha=0.5,
               label=f'Average ({mean_rate:.1f}%)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add shaded regions for time periods
    ax.axvspan(-0.5, 5.5, alpha=0.1, color='blue')
    ax.axvspan(5.5, 11.5, alpha=0.1, color='yellow')
    ax.axvspan(11.5, 17.5, alpha=0.1, color='orange')
    ax.axvspan(17.5, 23.5, alpha=0.1, color='purple')

    plt.tight_layout()
    plt.show()

    # Visualization 3: Line Chart - Phishing Rate Trend
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(hours, phishing_rates, marker='o', linewidth=2, markersize=6,
            color='#e74c3c', label='Phishing Rate')
    ax.fill_between(hours, phishing_rates, alpha=0.3, color='#e74c3c')

    ax.set_title("Phishing Rate Trend Throughout the Day", fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel("Phishing Rate (%)", fontsize=11)
    ax.set_xlabel("Hour of Day", fontsize=11)
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{h:02d}:00" for h in hours], rotation=45, ha='right')
    ax.axhline(y=mean_rate, color='black', linestyle='--', linewidth=1, alpha=0.5,
               label=f'Average ({mean_rate:.1f}%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add shaded regions for time periods
    ax.axvspan(-0.5, 5.5, alpha=0.1, color='blue')
    ax.axvspan(5.5, 11.5, alpha=0.1, color='yellow')
    ax.axvspan(11.5, 17.5, alpha=0.1, color='orange')
    ax.axvspan(17.5, 23.5, alpha=0.1, color='purple')

    plt.tight_layout()
    plt.show()

    # Visualization 4: Time Period Comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    period_stats = []
    for period in period_order:
        period_data = df_temp[df_temp['time_period'] == period]
        legit = len(period_data[period_data['label'] == 0])
        phish = len(period_data[period_data['label'] == 1])
        period_stats.append({'period': period, 'legitimate': legit, 'phishing': phish})

    period_df = pd.DataFrame(period_stats)
    x = range(len(period_df))
    width = 0.6

    bars1 = ax.bar(x, period_df['legitimate'], width, label='Legitimate',
                   color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x, period_df['phishing'], width, bottom=period_df['legitimate'],
                   label='Phishing', color='#e74c3c', alpha=0.8)

    ax.set_title("Email Distribution by Time Period", fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel("Number of Emails", fontsize=11)
    ax.set_xlabel("Time Period", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([p.split(' ')[0] for p in period_df['period']], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add percentage labels
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        total = height1 + height2

        if total > 0:
            pct_phishing = (height2 / total * 100)
            ax.text(bar2.get_x() + bar2.get_width() / 2., height1 + height2 + 50,
                    f'{pct_phishing:.1f}%', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.show()

    # Visualization 5: Heatmap - Hour Distribution
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create pivot table and normalize
    pivot_data = df.groupby(['hour', 'label']).size().unstack(fill_value=0)
    pivot_data = pivot_data.sort_index()
    pivot_normalized = pivot_data.div(pivot_data.sum(axis=1), axis=0) * 100

    sns.heatmap(pivot_normalized, annot=True, fmt='.1f', cmap='RdYlGn_r',
                cbar_kws={'label': 'Percentage (%)'}, ax=ax, linewidths=0.5, linecolor='gray')

    ax.set_title("Email Type Distribution by Hour (Percentage Heatmap)",
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel("Hour of Day", fontsize=11)
    ax.set_xlabel("Label", fontsize=11)
    ax.set_xticklabels(['Legitimate', 'Phishing'], rotation=0)
    ax.set_yticklabels([f"{h:02d}:00" for h in pivot_normalized.index], rotation=0)

    plt.tight_layout()
    plt.show()

    # Chi-square test for independence
    ct = pd.crosstab(df['hour'], df['label'])
    chi2, p_value, dof, expected = chi2_contingency(ct)

    print(f"\n✓ Chi-square Test Results (Hour vs Phishing):")
    print(f"  • Chi-square statistic: {chi2:.4f}")
    print(f"  • P-value: {p_value:.10e}")
    print(f"  • Degrees of freedom: {dof}")

    if p_value < 0.05:
        print(f"  • Result: Statistically significant relationship (p < 0.05)")
        print(f"    Hour of day is significantly related to phishing emails.")
    else:
        print(f"  • Result: No statistically significant relationship (p >= 0.05)")

    # Find highest and lowest risk hours
    max_hour = hour_stats['Phishing_Rate'].idxmax()
    min_hour = hour_stats['Phishing_Rate'].idxmin()
    print(f"\n✓ Risk Assessment:")
    print(f"  • Highest risk hour: {max_hour:02d}:00 ({hour_stats.loc[max_hour, 'Phishing_Rate']:.2f}% phishing)")
    print(f"  • Lowest risk hour: {min_hour:02d}:00 ({hour_stats.loc[min_hour, 'Phishing_Rate']:.2f}% phishing)")

    # Top 5 riskiest hours
    top_risk_hours = hour_stats.nlargest(5, 'Phishing_Rate')
    print(f"\n✓ Top 5 Riskiest Hours:")
    for hour, row in top_risk_hours.iterrows():
        print(f"  • {hour:02d}:00 - {row['Phishing_Rate']:.2f}% phishing ({row[1]}/{row['Total']} emails)")

    # Top 5 safest hours
    safest_hours = hour_stats.nsmallest(5, 'Phishing_Rate')
    print(f"\n✓ Top 5 Safest Hours:")
    for hour, row in safest_hours.iterrows():
        print(f"  • {hour:02d}:00 - {row['Phishing_Rate']:.2f}% phishing ({row[1]}/{row['Total']} emails)")

else:
    print("\n⚠ Warning: 'hour' or 'label' column not found in dataset")
    print("Please ensure both columns exist to perform correlation analysis.")

print("\n" + "=" * 60)
print("HOUR OF DAY ANALYSIS COMPLETE")
print("=" * 60)