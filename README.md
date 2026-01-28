# Statistical Analysis of Sign Language Recognition with Simple CNN Models
[Statistical Analysis of Sign Language Recognition with Simple CNN Models](docs/ATAI_P_Report_XU.pdf)
A university project in Worclaw in in cooperation with the lecture "Advanced Topics of AI".

![BaselineCNN Architecture (without activation and regularization layers)](report_visualizations/model_architecture_white_bg.png)

## Table of Contents
- [Statistical Analysis of Sign Language Recognition with Simple CNN Models](#statistical-analysis-of-sign-language-recognition-with-simple-cnn-models)
  - [Table of Contents](#table-of-contents)
  - [Objective](#objective)
    - [Research Steps](#research-steps)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
    - [Running the Project](#running-the-project)
  - [Results](#results)
    - [Performance Metrics](#performance-metrics)
    - [Statistical Analysis](#statistical-analysis)
      - [Friedman ANOVA Test](#friedman-anova-test)
      - [Wilcoxon Signed-Rank Test (Accuracy)](#wilcoxon-signed-rank-test-accuracy)
      - [Wilcoxon Signed-Rank Test (F1-Score)](#wilcoxon-signed-rank-test-f1-score)
      - [Notes](#notes)
  - [Visualizing the model architecture](#visualizing-the-model-architecture)
  - [Mentionable References](#mentionable-references)

## Objective
Conduct a statistical analysis on multiple experiment groups to analyze whether they are statistically different.

### Research Steps
1. Define experiment groups (here: 3 CNN models varying in layer count and regularization factor)
2. Train on a given dataset (here: https://www.kaggle.com/datasets/grassknoted/asl-alphabet).
3. Evaluate results based on accuracy and F1-score.
4. Do statistical tests:
   - Friedman ANOVA
   - Wilcoxon paired signed-rank
   - Post-hoc Hommel correction
5. Compute effect size:
   - Friedman ANOVA effect size: Kendall's W
   - Wilcoxon effect size: r

_Disclaimer: It is not the main objective to train the best model for this task (SLR) but rather to learn to conduct and properly document a statistical analysis._

For more detailed information, see the [final report](docs/ATAI_P_Report_XU.pdf).

## Project Structure

```
sign-language-recognition/
├── data/                              # Dataset directory
│   └── asl_alphabet_dataset/          # ASL Alphabet dataset
│       ├── train/                     # Training data
│       ├── test/                      # Test data
│       └── asl_alphabet_preprocessed/ # Preprocessed versions
├── src/                               # Source code
│   ├── models/                        # Model definitions
│   │   ├── simple_cnn.py              # BaselineCNN, DeeperCNN, RegularizedCNN
│   │   └── mobilenetv3.py             # MobileNetV3 model variants
│   ├── dataloader/                    # Data loading utilities
│   └── utils.py                       # Utility functions
├── configs/                           # Configuration files
├── docs/                              # Final report
├── output/                            # Experimental results for CNN variants
│   ├── BaselineCNN/                   # BaselineCNN results
│   ├── DeeperCNN/                     # DeeperCNN results
│   ├── RegularizedCNN/                # RegularizedCNN results
│   └── simple_experimental_results_0.1-5.csv  # Aggregated results
├── report_visualizations/             # Model architecture visualizations
├── train_simple.py                    # Training script for CNN models
├── train.py                           # Training script for MobileNetV3
├── preprocess.py                      # Data preprocessing script
├── statistical_test.ipynb             # Statistical analysis notebook
└── requirements.txt                   # Python dependencies
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/xuju-dev/sign-language-recognition.git
cd sign-language-recognition
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset from [Kaggle ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) and place it in the `data/asl_alphabet_dataset/` directory.

4. (Optional) Set up PYTHONPATH to project directory on macOS for model visualization:
```bash
export PYTHONPATH=$(pwd)
```

### Running the Project

To preprocess the data:
```bash
python preprocess.py
```

To train CNN models:
```bash
python train_simple.py
```

To train mobilenetv3_small model (https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k):
```bash
python train.py
```

To run statistical analysis:
```bash
jupyter notebook statistical_test.ipynb
```

## Results

The experiments evaluate three CNN model variants:
- **BaselineCNN**: Baseline CNN architecture (2-layered CNN with 1 max pooling layer, dropout factor 0.3)
- **DeeperCNN**: Extended CNN with more layers (3-layered CNN with 2 max pooling layers, dropout factor 0.3)
- **RegularizedCNN**: CNN with regularization techniques (3-layered CNN with 2 max pooling layers, dropout factor 0.5)

### Performance Metrics
Models were trained with ** 5 times repeated 5-fold cross-validation** (n=25 runs) on the ASL Alphabet dataset resulting in total 75 total runs (25 runs per model variant).

Metrics reported include **Accuracy**, **Macro F1-Score**, and **Training Time** (minutes).  

Values are reported as **mean ± standard deviation** (rounded to four decimal places).

| Metric | BaselineCNN | DeeperCNN | RegularizedCNN |
|--------|------------|-----------|----------------|
| Accuracy | 0.4214 ± 0.0073 | 0.4602 ± 0.0290 | 0.4015 ± 0.0319 |
| F1-Score (Macro) | 0.4036 ± 0.0073 | 0.4422 ± 0.0309 | 0.3820 ± 0.0330 |
| Training Time (minutes) | 11.09 ± 0.48 | 14.61 ± 0.48 | 14.44 ± 0.40 |

> **Note:** Accuracy and F1-score values are fractions (0–1). Training Time is per model run.

### Statistical Analysis

Comprehensive statistical testing was performed on 25 repeated measurements (5 repeats × 5-fold CV) per model:

#### Friedman ANOVA Test

P-values from Friedman ANOVA for model performance metrics. Significance is evaluated at α = 0.05.

| Metric | p-value | Significant (α=0.05) |
|--------|---------|---------------------|
| Accuracy | 6.96e-8 | ✓ |
| F1-Score (Macro) | 1.17e-7 | ✓ |


**Effect Size (Kendall's W):**
- Accuracy: W = 0.66 (large effect)
- F1-Score: W = 0.64 (large effect)

---

#### Wilcoxon Signed-Rank Test (Accuracy)

Pairwise comparisons of models using Wilcoxon signed-rank test with **Hommel correction**:

| Comparison | Wilcoxon p-value | Hommel-corrected p | Significant (α=0.05) |
|------------|----------------|------------------|---------------------|
| BaselineCNN vs DeeperCNN | 3.19e-5 | 6.39e-5 | ✓ |
| BaselineCNN vs RegularizedCNN | 3.78e-3 | 3.78e-3 | ✓ |
| DeeperCNN vs RegularizedCNN | 1.19e-7 | 3.58e-7 | ✓ |

**Effect Size r for Wilcoxon tests (Accuracy):**
- BaselineCNN vs. DeeperCNN: r = 0.76 (large effect)
- BaselineCNN vs. RegularizedCNN: r = 0.56 (large effect)
- DeeperCNN vs. RegularizedCNN: r = 0.87 (large effect)

#### Wilcoxon Signed-Rank Test (F1-Score)

| Comparison | Wilcoxon p-value | Hommel-corrected p | Significant (α=0.05) |
|------------|----------------|------------------|---------------------|
| BaselineCNN vs DeeperCNN | 4.54e-5 | 9.08e-5 | ✓ |
| BaselineCNN vs RegularizedCNN | 1.63e-3 | 1.63e-3 | ✓ |
| DeeperCNN vs RegularizedCNN | 1.19e-7 | 3.58e-7 | ✓ |

**Effect Size r for Wilcoxon tests (F1-Score):**
- BaselineCNN vs. DeeperCNN: r = 0.75 (large effect)
- BaselineCNN vs. RegularizedCNN: r = 0.61 (large effect)
- DeeperCNN vs. RegularizedCNN: r = 0.87 (large effect)

---

#### Notes

- **Best performing model** for Accuracy and F1-Score: **DeeperCNN**  
- Performance and training time boxplots are available in `test_visualization/model_performance_comparison.png`  
- Effect sizes can be added to highlight practical significance.
  
**Conclusion:** All three model variants show statistically significant differences. The DeeperCNN model outperforms both BaselineCNN and RegularizedCNN and shows the highest statistically significant difference when compaired with the other two models. With its largest effect size against RegularizedCNN.

Results and detailed statistical comparisons can be run and found in `output/simple_experimental_results_0.1-5.csv` and `statistical_test.ipynb`.

## Visualizing the model architecture
The model architecture is visualized using `nnviz`.
```
pip install nnviz
```

On MacOS I had to set the environment variable before building the graphics.
```
export PYTHONPATH=$(pwd)
```

Then visualization can be run like this:
```
nnviz src.models.simple_cnn:BaselineCNN --style show_specs=False --style show_node_arguments=False --style show_node_params=False --style show_node_source=False --out report_visualizations/baseline_architecture.png
```
All flags here are optional.
`--style/-S` flag allows for customization. 
`--out/-o` flag is optional and states the path to save the visualization. Standard output is a PDF.
For more detailed information, see https://nnviz.readthedocs.io/en/latest/cli/customization.html.

To make the plot horizontal I make a `.dot` file and then convert it to `.svg` through `Graphviz`.
```
nnviz src.models.simple_cnn:BaselineCNN -S show_specs=False -S show_node_name=False -S show_node_params=False -S show_node_arguments=False -S show_node_source=False -S show_clusters=False -o report_visualizations/baseline_architecture.dot
```
```
dot -Grankdir=LR -Tsvg report_visualizations/baseline_architecture.dot -o report_visualizations/baseline_architecture.svg
```

## Mentionable References
Demšar, J. (2006). Statistical Comparisons of Classifiers over Multiple Data Sets. Journal of Machine Learning Research, 7(1), 1–30. Retrieved from http://jmlr.org/papers/v7/demsar06a.html

Howard, A., Sandler, M., Chu, G., Chen, L.-C., Chen, B., Tan, M., Wang, W., Zhu, Y., Pang, R., Vasudevan, V., Le, Q. V., & Adam, H. (2019). Searching for MobileNetV3 (Version 5). arXiv. https://doi.org/10.48550/ARXIV.1905.02244


For complete list please see [final report](docs/ATAI_P_Report_XU.pdf).
