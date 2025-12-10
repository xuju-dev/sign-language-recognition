import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests
import pingouin as pg

seeds = [42, 15, 88, 7, 66, 17, 66, 123, 456, 839]  # 10 runs
fried_diffs = []
wilc_diffs = []

p_scipy_all = []
p_pingouin_all = []

for s in seeds:
    np.random.seed(s)
    v1 = np.random.rand(25)
    v2 = np.random.rand(25)
    v3 = np.random.rand(25)

    print("Seed: 42")
    print(f"Vector 1:\n{v1}\n")
    print(f"Vector 2:\n{v2}\n")
    print(f"Vector 3:\n{v3}\n")

    # TODO: Load in evaluation results from training

    # Statistical Analysis
    print("Statistical Analysis:\n")
    # === scipy + statsmodels ===
    # Friedman test
    stat_fried, p_fried = friedmanchisquare(v1, v2, v3)
    print(f"(Scipy) Friedman test statistic: {stat_fried}, p-value: {p_fried}")

    # Wilcox`on paired test
    stat_will12, p_wil12 = wilcoxon(v1, v2)
    stat_will13, p_wil13 = wilcoxon(v1, v3)
    stat_will23, p_wil23 = wilcoxon(v2, v3)
    p_wilcoxon_scipy = [p_wil12, p_wil13, p_wil23]
    # print(f"(Scipy) Wilcoxon test statistic [1-2, 1-3, 2-3]: {[stat_will12, stat_will13, stat_will23]}")
    print(f"(Scipy) Wilcoxon p-value [1-2, 1-3, 2-3]: {p_wilcoxon_scipy}")

    # Hommel post hoc
    post_p = multipletests(
        [p_wil12, p_wil13, p_wil23], 
        alpha=0.05, 
        method='hommel', 
        maxiter=1, 
        is_sorted=False, 
        returnsorted=False
    )

    print(f"(statsmodels) Post-hoc Hommel corrected p-values (from scipy): {post_p[1]}\n")

    # === pingouin ===
    df = pd.DataFrame({
        'v1': {i: v for i, v in enumerate(v1)},
        'v2': {i: v for i, v in enumerate(v2)},
        'v3': {i: v for i, v in enumerate(v3)},
    })

    # Friedman test
    p_fried_ping = pg.friedman(data=df, method='f')
    print(f"(pingouin) Friedman test p-value: {p_fried_ping['p-unc'].values[0]}")

    # Wilcoxon paired test
    p_wil12_ping = pg.wilcoxon(v1, v2)['p-val'].values[0]
    p_wil13_ping = pg.wilcoxon(v1, v3)['p-val'].values[0]
    p_wil23_ping = pg.wilcoxon(v2, v3)['p-val'].values[0]
    p_wilcoxon_ping = [p_wil12_ping, p_wil13_ping, p_wil23_ping]
    print(f"(pingouin) Wilcoxon p-value [1-2, 1-3, 2-3]: {p_wilcoxon_ping}")

    # Hommel post hoc
    post_p = multipletests(
        [p_wil12, p_wil13, p_wil23], 
        alpha=0.05, 
        method='hommel', 
        maxiter=1, 
        is_sorted=False, 
        returnsorted=False
    )

    print(f"(statsmodels) Post-hoc Hommel corrected p-values (from pingouin): {post_p[1]}")

    # === Power analysis ===
    # print("\nPower Analysis:")
    # # Effect size calculation
    # ef12 = pg.compute_effsize(v1, v2, paired=True, eftype='eta-square')
    # ef13 = pg.compute_effsize(v1, v3, paired=True, eftype='eta-square')
    # ef23 = pg.compute_effsize(v2, v3, paired=True, eftype='eta-square')
    # print(f"(pingouin) Effect size (eta-square) between v1 and v2: {ef12}")
    # print(f"(pingouin) Effect size (eta-square) between v1 and v3: {ef13}")
    # print(f"(pingouin) Effect size (eta-square) between v2 and v3: {ef23}")

    # Visualization of p-values
    # Example p-values (replace with your own)
    comparisons = ["A vs B", "A vs C", "B vs C"]

    p_scipy = p_fried
    p_pingouin = p_fried_ping['p-unc'].values[0]

    fried_diff = np.abs(p_scipy - p_pingouin)
    print(fried_diff)

    wilc_diff = np.abs(np.array(p_wilcoxon_scipy) - np.array(p_wilcoxon_ping))
    print(wilc_diff)

    # x = np.arange(len(comparisons))
    # width = 0.3                     # width of the bars

    # fig, ax = plt.subplots(figsize=(8, 5))

    # rects1 = ax.bar(x - width/2, p_scipy, width, label='SciPy')
    # rects2 = ax.bar(x + width/2, p_pingouin, width, label='Pingouin')

    # ax.text(0.5, 0.65, f'p-value difference: {diff:.4f}', fontsize=10, color='black')

    # # Labels and formatting
    # ax.set_ylabel('p-value')
    # ax.set_title('Comparison of Friedman-ANOVA p-values: SciPy vs. Pingouin')
    # ax.set_xticks(x)
    # ax.set_xticklabels(comparisons)
    # ax.axhline(0.05, color='red', linestyle='--', linewidth=1, label='α = 0.05')
    # plt.ylim(0.3, 0.75) 
    # ax.legend()

    # plt.tight_layout()
    # plt.show()
    fried_diffs.append(fried_diff)
    wilc_diffs.append(wilc_diff)

    p_scipy_all.append(p_scipy)
    p_pingouin_all.append(p_pingouin)

print("\nSummary of p-value differences across seeds:")
print(f"Friedman-ANOVA p-value differences: {fried_diffs}")
print(f"Wilcoxon p-value differences: {wilc_diffs}")

plt.figure(figsize=(10, 5))
# Create evenly spaced positions: 0..9
x = range(len(seeds))

plt.figure(figsize=(10,5))
plt.plot(x, fried_diffs, marker="o", linestyle="-")

plt.text(
    0.95, 0.95, 
    f'Mean p-value difference: {np.mean(fried_diffs):.4f}', 
    horizontalalignment='right', 
    verticalalignment='top'
)

# Set custom tick labels (your seed IDs)
plt.xticks(x)
plt.legend()
plt.ylabel('Absolute p-value Difference')
plt.title('Friedman-ANOVA p-value Differences between SciPy and Pingouin')
plt.grid()
plt.tight_layout()
plt.savefig('friedman_pvalue_differences.png', dpi=200)
plt.close()

x = np.arange(len(seeds))
width = 0.2

fig, ax = plt.subplots(figsize=(8, 5))

rects1 = ax.bar(x - width/2, p_scipy_all, width, label='SciPy', color='#ADD8E6')
rects2 = ax.bar(x + width/2, p_pingouin_all, width, label='Pingouin', color='#D8BFD8')

def add_labels(rects, top: float = 0.0):
    for rect in rects:
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width()/2,
            (height/2)+top,                      # middle of bar; use height*0.9 for top
            f"{height:.3f}",              # value with 3 decimals
            ha='center', va='center',
            fontsize=8, color='black'
        )

add_labels(rects1, 0.05)
add_labels(rects2, -0.05)

# Labels and formatting
ax.set_ylabel('p-value')
ax.set_title('Comparison of Friedman-ANOVA p-values: SciPy vs. Pingouin')
ax.set_xticks(x)
ax.set_xticklabels(x)
ax.legend()

plt.tight_layout()
plt.savefig('bar_friedman_pvalue_differences.png', dpi=200)