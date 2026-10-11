"""
Seaborn Core Plot Types Applied to Titanic Dataset (train.csv)
Based on: "Seaborn's Core Plot Types" by Techie Ashu (Medium)
https://medium.com/@btech.07/seaborns-core-plot-types-7fefba8a9724

Covers the 5 core plot families:
1. Relational Plots: scatterplot, lineplot, relplot
2. Categorical Plots: countplot, barplot, boxplot, violinplot, swarmplot, pointplot, catplot
3. Distribution Plots: histplot, kdeplot, rugplot, displot (replaces deprecated distplot)
4. Regression Plots: regplot, lmplot
5. Matrix Plots: heatmap, clustermap
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_data(filepath='train.csv'):
    """Load and prepare train.csv dataset."""
    if not os.path.exists(filepath):
        # Fallback to eda/train.csv if running from root
        filepath = os.path.join(os.path.dirname(__file__), 'train.csv')
    df = pd.read_csv(filepath)
    return df


def plot_relational(df, output_dir=None):
    """1. Relational Plots: visualize relationships between numeric variables."""
    print("--> Generating Relational Plots...")
    
    # 1.1 Scatterplot: Age vs Fare, colored by survival and sized by Pclass
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x='Age',
        y='Fare',
        hue='Survived',
        size='Pclass',
        palette='viridis',
        alpha=0.7
    )
    plt.title('Relational: scatterplot (Age vs Fare by Survival & Class)')
    if output_dir:
        plt.savefig(os.path.join(output_dir, '1_1_scatterplot.png'), bbox_inches='tight')
    plt.close()

    # 1.2 Lineplot: Age vs Fare trend across Pclass
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x='Age', y='Fare', hue='Pclass', palette='Set1', errorbar=None)
    plt.title('Relational: lineplot (Average Fare Trend across Ages by Pclass)')
    if output_dir:
        plt.savefig(os.path.join(output_dir, '1_2_lineplot.png'), bbox_inches='tight')
    plt.close()

    # 1.3 Relplot (Figure-level): Faceted scatterplot across classes
    g = sns.relplot(
        data=df,
        x='Age',
        y='Fare',
        hue='Survived',
        col='Pclass',
        kind='scatter',
        palette='coolwarm',
        height=4,
        aspect=0.9
    )
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle('Relational: relplot (Faceted by Pclass)')
    if output_dir:
        g.savefig(os.path.join(output_dir, '1_3_relplot.png'), bbox_inches='tight')
    plt.close()


def plot_categorical(df, output_dir=None):
    """2. Categorical Plots: relationships involving categorical data."""
    print("--> Generating Categorical Plots...")

    # 2.1 Countplot: Passenger counts by Pclass and Survived
    plt.figure(figsize=(7, 4))
    sns.countplot(data=df, x='Pclass', hue='Survived', palette='Set2')
    plt.title('Categorical: countplot (Passenger Counts by Pclass & Survival)')
    if output_dir:
        plt.savefig(os.path.join(output_dir, '2_1_countplot.png'), bbox_inches='tight')
    plt.close()

    # 2.2 Barplot: Mean survival rate by Pclass and Sex
    plt.figure(figsize=(7, 4))
    sns.barplot(data=df, x='Pclass', y='Survived', hue='Sex', palette='pastel')
    plt.title('Categorical: barplot (Survival Rate by Pclass & Sex)')
    if output_dir:
        plt.savefig(os.path.join(output_dir, '2_2_barplot.png'), bbox_inches='tight')
    plt.close()

    # 2.3 Boxplot: Age distribution across Pclass and Survival
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='Pclass', y='Age', hue='Survived', palette='coolwarm')
    plt.title('Categorical: boxplot (Age Distribution by Pclass & Survival)')
    if output_dir:
        plt.savefig(os.path.join(output_dir, '2_3_boxplot.png'), bbox_inches='tight')
    plt.close()

    # 2.4 Violinplot: Age distribution split by Survival
    plt.figure(figsize=(8, 5))
    sns.violinplot(data=df, x='Pclass', y='Age', hue='Survived', split=True, palette='muted')
    plt.title('Categorical: violinplot (Age Distribution by Pclass & Survival)')
    if output_dir:
        plt.savefig(os.path.join(output_dir, '2_4_violinplot.png'), bbox_inches='tight')
    plt.close()

    # 2.5 Swarmplot: Sampled individual points of Age by Pclass
    plt.figure(figsize=(8, 5))
    sampled_df = df.dropna(subset=['Age']).sample(n=200, random_state=42)
    sns.swarmplot(data=sampled_df, x='Pclass', y='Age', hue='Survived', palette='deep', size=5)
    plt.title('Categorical: swarmplot (Sampled Age Points by Pclass & Survival)')
    if output_dir:
        plt.savefig(os.path.join(output_dir, '2_5_swarmplot.png'), bbox_inches='tight')
    plt.close()

    # 2.6 Pointplot: Survival rate comparison
    plt.figure(figsize=(7, 4))
    sns.pointplot(data=df, x='Pclass', y='Survived', hue='Sex', palette='dark', markers=['o', 's'], linestyles=['-', '--'])
    plt.title('Categorical: pointplot (Survival Rate Points and Intervals)')
    if output_dir:
        plt.savefig(os.path.join(output_dir, '2_6_pointplot.png'), bbox_inches='tight')
    plt.close()

    # 2.7 Catplot (Figure-level): Faceted barplot across Embarked locations
    g = sns.catplot(
        data=df.dropna(subset=['Embarked']),
        x='Pclass',
        y='Survived',
        hue='Sex',
        col='Embarked',
        kind='bar',
        palette='Set1',
        height=4,
        aspect=0.85
    )
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle('Categorical: catplot (Survival Rate by Embarked Port)')
    if output_dir:
        g.savefig(os.path.join(output_dir, '2_7_catplot.png'), bbox_inches='tight')
    plt.close()


def plot_distributions(df, output_dir=None):
    """3. Distribution Plots: understanding variable distributions."""
    print("--> Generating Distribution Plots...")

    # 3.1 Histplot: Age distribution with KDE (modern replacement for distplot)
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='Age', kde=True, hue='Survived', bins=25, palette='tab10')
    plt.title('Distribution: histplot with KDE (Age Distribution by Survival)')
    if output_dir:
        plt.savefig(os.path.join(output_dir, '3_1_histplot.png'), bbox_inches='tight')
    plt.close()

    # 3.2 Kdeplot: Fare density comparison
    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=df, x='Fare', hue='Survived', fill=True, common_norm=False, palette='crest')
    plt.xlim(0, 200)  # Zoom in on main density
    plt.title('Distribution: kdeplot (Fare Density by Survival)')
    if output_dir:
        plt.savefig(os.path.join(output_dir, '3_2_kdeplot.png'), bbox_inches='tight')
    plt.close()

    # 3.3 Rugplot: Marginal observations along axis
    plt.figure(figsize=(8, 3))
    sns.rugplot(data=df, x='Age', hue='Survived', height=0.5, palette='Set2')
    plt.title('Distribution: rugplot (Age Marginal Data Points)')
    if output_dir:
        plt.savefig(os.path.join(output_dir, '3_3_rugplot.png'), bbox_inches='tight')
    plt.close()

    # 3.4 Displot (Figure-level): Faceted KDE distribution
    g = sns.displot(data=df, x='Age', hue='Survived', kind='kde', fill=True, col='Sex', height=4, aspect=1.1)
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle('Distribution: displot (Age Density Faceted by Sex)')
    if output_dir:
        g.savefig(os.path.join(output_dir, '3_4_displot.png'), bbox_inches='tight')
    plt.close()


def plot_regression(df, output_dir=None):
    """4. Regression Plots: linear relationships and trends."""
    print("--> Generating Regression Plots...")

    # 4.1 Regplot: Age vs Fare with regression line
    plt.figure(figsize=(8, 5))
    clean_df = df.dropna(subset=['Age', 'Fare'])
    sns.regplot(
        data=clean_df,
        x='Age',
        y='Fare',
        scatter_kws={'alpha': 0.4, 'color': 'teal'},
        line_kws={'color': 'darkred', 'linewidth': 2}
    )
    plt.title('Regression: regplot (Age vs Fare Linear Regression)')
    if output_dir:
        plt.savefig(os.path.join(output_dir, '4_1_regplot.png'), bbox_inches='tight')
    plt.close()

    # 4.2 Lmplot (Figure-level): Faceted regression across classes
    g = sns.lmplot(
        data=clean_df,
        x='Age',
        y='Fare',
        hue='Survived',
        col='Pclass',
        palette='Set1',
        scatter_kws={'alpha': 0.4},
        height=4,
        aspect=0.9
    )
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle('Regression: lmplot (Age vs Fare Faceted by Pclass & Survival)')
    if output_dir:
        g.savefig(os.path.join(output_dir, '4_2_lmplot.png'), bbox_inches='tight')
    plt.close()


def plot_matrix(df, output_dir=None):
    """5. Matrix Plots: correlation heatmaps and clustering."""
    print("--> Generating Matrix Plots...")

    # Numerical correlation matrix
    num_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    corr = df[num_cols].corr()

    # 5.1 Heatmap: Correlation heatmap
    plt.figure(figsize=(7, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, cbar=True)
    plt.title('Matrix: heatmap (Correlation Matrix of Numeric Features)')
    if output_dir:
        plt.savefig(os.path.join(output_dir, '5_1_heatmap.png'), bbox_inches='tight')
    plt.close()

    # 5.2 Clustermap: Hierarchically clustered correlation
    cm = sns.clustermap(corr, annot=True, cmap='vlag', fmt='.2f', figsize=(7, 7), linewidths=0.5)
    cm.fig.subplots_adjust(top=0.9)
    cm.fig.suptitle('Matrix: clustermap (Hierarchical Correlation Clustering)')
    if output_dir:
        cm.savefig(os.path.join(output_dir, '5_2_clustermap.png'), bbox_inches='tight')
    plt.close()


def run_all(save_plots=True):
    """Execute and test all Seaborn plot types on train.csv."""
    df = load_data()
    print(f"Loaded train.csv successfully. Shape: {df.shape}")

    output_dir = None
    if save_plots:
        output_dir = os.path.join(os.path.dirname(__file__), 'plots')
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving generated plots to: {output_dir}")

    plot_relational(df, output_dir)
    plot_categorical(df, output_dir)
    plot_distributions(df, output_dir)
    plot_regression(df, output_dir)
    plot_matrix(df, output_dir)

    print("\n[SUCCESS] All 5 core Seaborn plot categories tested and executed successfully!")


if __name__ == '__main__':
    run_all(save_plots=True)
