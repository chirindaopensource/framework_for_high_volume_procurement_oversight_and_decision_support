# **`README.md`**

# The Payment Heterogeneity Index: An Integrated Unsupervised Framework

<!-- PROJECT SHIELDS -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2605.12547-b31b1b.svg)](https://arxiv.org/abs/2605.12547)
[![Journal](https://img.shields.io/badge/Journal-ArXiv%20Preprint-003366)](https://arxiv.org/abs/2605.12547)
[![Year](https://img.shields.io/badge/Year-2026-purple)](https://github.com/chirindaopensource/framework_for_high_volume_procurement_oversight_and_decision_support)
[![Discipline: Forensic Data Science](https://img.shields.io/badge/Discipline-Forensic%20Data%20Science-00529B)](https://github.com/chirindaopensource/framework_for_high_volume_procurement_oversight_and_decision_support)
[![Discipline: Econometrics](https://img.shields.io/badge/Discipline-Computational%20Econometrics-00529B)](https://github.com/chirindaopensource/framework_for_high_volume_procurement_oversight_and_decision_support)
[![Discipline: Machine Learning](https://img.shields.io/badge/Discipline-Unsupervised%20Learning-00529B)](https://github.com/chirindaopensource/framework_for_high_volume_procurement_oversight_and_decision_support)
[![Data Sources](https://img.shields.io/badge/Data-City%20of%20York%20Council%20Transparency-lightgrey)](https://data.yorkopendata.org/dataset/all-payments-to-suppliers/resource/87260305-e43a-40a4-866e-ef5e050a3426)
[![Core Method: GMM](https://img.shields.io/badge/Method-Gaussian%20Mixture%20Models-orange)](https://github.com/chirindaopensource/framework_for_high_volume_procurement_oversight_and_decision_support)
[![Core Method: TF-IDF](https://img.shields.io/badge/Method-TF--IDF%20%26%20Jaccard-orange)](https://github.com/chirindaopensource/framework_for_high_volume_procurement_oversight_and_decision_support)
[![Core Method: Permutation](https://img.shields.io/badge/Method-Monte%20Carlo%20Permutation-orange)](https://github.com/chirindaopensource/framework_for_high_volume_procurement_oversight_and_decision_support)
[![Core Method: K-S Test](https://img.shields.io/badge/Method-Kolmogorov--Smirnov%20Test-orange)](https://github.com/chirindaopensource/framework_for_high_volume_procurement_oversight_and_decision_support)
[![Optimization](https://img.shields.io/badge/Optimization-BIC%20Model%20Selection-red)](https://github.com/chirindaopensource/framework_for_high_volume_procurement_oversight_and_decision_support)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type Checking: mypy](https://img.shields.io/badge/type%20checking-mypy-blue)](http://mypy-lang.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NetworkX](https://img.shields.io/badge/NetworkX-%2300529B.svg?style=flat)](https://networkx.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=flat&logo=scipy&logoColor=white)](https://scipy.org/)
[![YAML](https://img.shields.io/badge/YAML-%23CB171E.svg?style=flat&logo=yaml&logoColor=white)](https://yaml.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=flat&logo=Jupyter&logoColor=white)](https://jupyter.org/)
[![Open Source](https://img.shields.io/badge/Open%20Source-%E2%9D%A4-brightgreen)](https://github.com/chirindaopensource/framework_for_high_volume_procurement_oversight_and_decision_support)

**Repository:** `https://github.com/chirindaopensource/framework_for_high_volume_procurement_oversight_and_decision_support`

**Owner:** 2026 Craig Chirinda (Open Source Projects)

This repository contains an **independent**, professional-grade Python implementation of the research methodology from the 2026 paper entitled **"The Payment Heterogeneity Index: An Integrated Unsupervised Framework for High-Volume Procurement Oversight and Decision Support"** by:

*   **Kyriakos Christodoulides**

The project provides a complete, end-to-end computational framework for replicating the paper's findings. It delivers a modular, highly optimized pipeline that executes the entire research workflow: from the rigorous lexical harmonisation of fragmented supplier identities, to the robust non-parametric scaling of financial flows, and the probabilistic extraction of latent payment regimes via Gaussian Mixture Models (GMMs). The pipeline culminates in the synthesis of the Payment Heterogeneity Index (PHI), converting massive transactional noise into a prioritized, auditable shortlist for human-in-the-loop institutional oversight.

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Features](#features)
- [Methodology Implemented](#methodology-implemented)
- [Core Components (Notebook Structure)](#core-components-notebook-structure)
- [Key Callable: `execute_full_phi_framework`](#key-callable-execute_full_phi_framework)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Input Data Structure](#input-data-structure)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Contributing](#contributing)
- [Recommended Extensions](#recommended-extensions)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Introduction

This project provides a Python implementation of the analytical framework presented in Christodoulides (2026). The core of this repository is the iPython Notebook `framework_for_high_volume_procurement_oversight_and_decision_support_draft.ipynb`, which contains a comprehensive suite of orchestrated tasks to replicate the paper's findings.

The pipeline addresses a foundational vulnerability in public procurement and forensic accounting: the reliance on simple univariate metrics (like the Coefficient of Variation) or restrictive digit-based tests (like Benford's Law), which fail to capture the complex, multi-modal structures inherent in modern institutional billing.

The codebase operationalizes the proposed solution—an **interpretable decision-support engine for institutional integrity**:
-   **Anchors** entity resolution in a multi-metric fuzzy logic ensemble (TF-IDF, Jaccard, Token Set Ratio) and Graph Theory (Connected Components) to eliminate identity fragmentation.
-   **Enforces** robust, outlier-resistant standardization using the Global Median and Interquartile Range (IQR).
-   **Propagates** the analysis to the structural level by fitting spherical Gaussian Mixture Models (GMMs), optimized via the Bayesian Information Criterion (BIC), to extract latent payment regimes.
-   **Evaluates** the topological structure through a multiplicative composite index (PHI) that synthesizes Modality, Asymmetry, Tail Behavior, and Structural Dispersion, providing exact log-additive attribution for Explainable AI (XAI).

## Theoretical Background

The implemented methods combine techniques from Computational Linguistics, Robust Statistics, and Information Theory.

**1. Robust Standardization:**
To map heterogeneous suppliers onto a common monetary reference frame without outlier distortion:
$$ \tilde{a}_i = \frac{a_i - \tilde{A}}{IQR} $$

**2. Probabilistic Regime Extraction (GMM):**
Latent payment regimes are extracted using EM optimization, constrained by a sample-size heuristic to prevent overfitting:
$$ k_{max} = \min\left(4, \left\lfloor \frac{n}{25} \right\rfloor\right) $$

**3. The Structural Heterogeneity Index (SHI) Components:**
The framework quantifies four distinct morphological dimensions:
*   **Modality ($M$):** $M = k$ (post-pruning component count).
*   **Asymmetry ($A$):** $A = 1 + |a_q|$, where $a_q$ is Bowley Skewness.
*   **Tail Behavior ($T$):** $T = 1 + |\ln(t_q)|$, where $t_q = \frac{(Q_{0.95} - Q_{0.05}) + \epsilon}{(Q_{0.75} - Q_{0.25}) + \epsilon}$.
*   **Structural Dispersion ($D$):** $D = 1 + \pi_{i^*} s_{i^*} + \sum_{i \neq i^*} \pi_i s_i \ln(1 + d_i)$.

**4. Joint Amplification and XAI Attribution:**
The Payment Heterogeneity Index (PHI) is the multiplicative synthesis of these components, decomposable via logarithms for audit transparency:
$$ PHI = M \times A \times T \times D $$
$$ Contribution_X = \left( \frac{\ln(X)}{\ln(PHI)} \right) \times 100\% $$

Below is a diagram which summarizes the proposed approach:

<div align="center">
  <img src="https://github.com/chirindaopensource/framework_for_high_volume_procurement_oversight_and_decision_support/blob/main/framework_for_high_volume_procurement_oversight_and_decision_support_ipo_main_four.png" alt="PHI System Architecture" width="100%">
</div>

## Features

The provided iPython Notebook implements the full research pipeline, including:

-   **Strict Topological Invariants:** Enforces $E \subseteq V \times V$ during Match Graph construction, mathematically guaranteeing that isolated suppliers (singletons) are preserved during entity resolution.
-   **IEEE 754 Floating-Point Safeguards:** Implements strict numerical clipping (`np.clip`) on GMM variances to prevent `NaN` propagation caused by variance collapse in highly clustered regimes.
-   **Stochastic Reproducibility:** Employs cryptographic hashing (SHA-256) to generate deterministic, immutable artifacts at every stage of the pipeline, ensuring perfect auditability.
-   **Exact Inferential Statistics:** Replaces heuristic thresholding with vectorized Monte Carlo permutation testing ($N=5000$) and Two-Sample Kolmogorov-Smirnov divergence testing to validate threshold anchoring.
-   **Configuration-Driven Design:** All study parameters, discretization thresholds, and topological policies are managed in an external `config.yaml` file, ensuring strict methodological reproducibility.

## Methodology Implemented

The core analytical steps directly implement the methodology from the paper:

1.  **Data Preprocessing (Tasks 1-4):** Ingests raw administrative data. Enforces strict temporal alignment, type coercion, and positive-flow isolation to remove accounting reversals.
2.  **Identity Harmonisation (Tasks 5-7):** Normalizes strings and computes TF-IDF, Jaccard, and Token Set Ratios. Constructs a Match Graph and extracts canonical suppliers via Connected Components.
3.  **Robust Scaling & Regime Extraction (Tasks 8-10):** Computes global Median/IQR, standardizes the high-volume cohort ($n \ge 50$), fits spherical GMMs via BIC, and prunes negligible weights ($\pi < 0.05$).
4.  **PHI Synthesis & Triage (Tasks 11-12):** Computes empirical quantiles, synthesizes $M, A, T, D$, assigns risk tiers, and computes exact log-additive XAI attributions.
5.  **Baseline Comparison (Task 13):** Computes the Coefficient of Variation (CV) and executes Spearman rank correlation to demonstrate PHI's superiority over univariate metrics.
6.  **Validation & Audit (Tasks 14-17):** Extracts endogenous peaks via Gaussian KDE, runs permutation tests for threshold anchoring, aggregates sectoral risk, and generates a JSON-serializable Reproduction Manifest.

## Core Components (Notebook Structure)

The notebook is structured as a logical pipeline with modular orchestrator functions for each of the 17 major tasks. All functions are self-contained, fully documented with strict type hints and comprehensive docstrings, and designed for professional-grade execution.

## Key Callable: `execute_full_phi_framework`

The project is designed around a single, top-level user-facing interface function:

-   **`execute_full_phi_framework`:** This apex orchestrator function runs the entire automated research pipeline from end-to-end. A single call to this function reproduces the entire computational portion of the project, managing data validation, entity resolution, GMM fitting, PHI scoring, inferential testing, and the final reproduction audit, while returning a comprehensive dictionary of immutable artifacts.

## Prerequisites

-   Python 3.10+
-   Core Python dependencies: `numpy`, `pandas`, `scipy`, `scikit-learn`, `networkx`, `rapidfuzz`, `pyyaml`, `Faker`.

## Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/chirindaopensource/framework_for_high_volume_procurement_oversight_and_decision_support.git
    cd framework_for_high_volume_procurement_oversight_and_decision_support
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    ```sh
    pip install numpy pandas scipy scikit-learn networkx rapidfuzz pyyaml Faker
    ```

## Input Data Structure

The pipeline requires a strictly formatted `pandas.DataFrame` (`df_raw`) containing 14 columns, including:
-   `record_id`: `int64` (Synthetic primary key).
-   `Creditor Name`: `string` (Raw supplier string for identity harmonisation).
-   `Payment Date`: `datetime64[ns]` (Transaction timestamp).
-   `Net Amount`: `float64` (Primary unit of analysis).
-   `Directorate` / `Subjective Detail`: `string` (Categorical metadata for forensic drill-down).

## Usage

The notebook provides a complete, step-by-step guide. The primary workflow is to execute the final cell, which demonstrates how to synthetically generate the required administrative data, load the configuration, and use the top-level orchestrator to execute the pipeline:

```python
import pandas as pd
import numpy as np
import yaml
from faker import Faker
from typing import Dict, Any

# ==============================================================================
# Step 1: Synthetic Data Generation (df_raw)
# ==============================================================================
# Initialize Faker and Numpy RNG for perfect reproducibility
fake = Faker('en_GB')
Faker.seed(42)
np.random.seed(42)

def generate_synthetic_procurement_data(
    num_records: int = 5000,
    num_base_suppliers: int = 20
) -> pd.DataFrame:
    """
    Generates a synthetic, high-fidelity administrative procurement DataFrame.
    Mimics identity fragmentation, heavy-tailed financial noise, and schema constraints.
    """
    if num_records <= 0:
        raise ValueError(f"num_records must be strictly positive, got {num_records}.")

    base_suppliers = [fake.company().upper() for _ in range(num_base_suppliers)]
    suffixes = ["", " LTD", " LIMITED", " PLC", " UK"]
    available_directorates = [
        "Adult Social Care and Integration", "Transport Environment and Planning", 
        "Finance", "Children and Education"
    ]

    data = {
        "record_id": [], "Organisation Name": [], "Directorate": [], "Department": [],
        "Service Plan": [], "Creditor Name": [], "Payment Date": [], "Transaction No": [],
        "Card Transaction": [], "Net Amount": [], "Irrecoverable VAT": [],
        "Subjective Group": [], "Subjective Subgroup": [], "Subjective Detail": []
    }

    start_ts = pd.Timestamp("2025-05-01")

    for i in range(1, num_records + 1):
        data["record_id"].append(i)
        data["Organisation Name"].append("City of York Council")
        
        directorate = np.random.choice(available_directorates)
        data["Directorate"].append(directorate)
        data["Department"].append(f"Dept of {directorate.split()[0]}")
        data["Service Plan"].append("Standard Service Plan")
        
        base_supplier = np.random.choice(base_suppliers)
        suffix = np.random.choice(suffixes)
        data["Creditor Name"].append(f"{base_supplier}{suffix}".strip())
        
        random_offset = pd.Timedelta(days=np.random.randint(0, 184))
        data["Payment Date"].append(start_ts + random_offset)
        
        data["Transaction No"].append(f"TXN-{i:07d}")
        data["Card Transaction"].append(np.random.rand() < 0.10)
        
        # Simulate heavy-tailed financial amounts (LogNormal)
        amount = np.random.lognormal(mean=6.0, sigma=1.5)
        # Inject 2% negative values to test positive-flow isolation logic
        if np.random.rand() < 0.02:
            amount = -amount
            
        amount = round(amount, 2)
        data["Net Amount"].append(amount)
        data["Irrecoverable VAT"].append(round(amount * 0.20, 2))
        
        data["Subjective Group"].append("Supplies and Services")
        data["Subjective Subgroup"].append("Professional Fees")
        data["Subjective Detail"].append("Consultancy")

    df_raw = pd.DataFrame(data)

    # Enforce strict schema types
    df_raw["record_id"] = df_raw["record_id"].astype("int64")
    string_cols = ["Organisation Name", "Directorate", "Department", "Service Plan", 
                   "Creditor Name", "Transaction No", "Subjective Group", 
                   "Subjective Subgroup", "Subjective Detail"]
    for col in string_cols:
        df_raw[col] = df_raw[col].astype("string")
        
    df_raw["Payment Date"] = df_raw["Payment Date"].astype("datetime64[ns]")
    df_raw["Card Transaction"] = df_raw["Card Transaction"].astype("bool")
    df_raw["Net Amount"] = df_raw["Net Amount"].astype("float64")
    df_raw["Irrecoverable VAT"] = df_raw["Irrecoverable VAT"].astype("float64")

    return df_raw

df_raw = generate_synthetic_procurement_data()

# ==============================================================================
# Step 2: Loading the Configuration (config.yaml)
# ==============================================================================
def load_study_configuration(filepath: str = "config.yaml") -> Dict[str, Any]:
    """Loads the deterministic hyperparameters and methodological constraints."""
    if not isinstance(filepath, str):
        raise TypeError(f"filepath must be a string, got {type(filepath)}.")
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        print(f"Successfully loaded configuration from: {filepath}")
        return config
    except FileNotFoundError:
        print(f"Fatal Error: Configuration file '{filepath}' not found.")
        raise

# Note: Ensure 'config.yaml' is saved in your working directory.
config = load_study_configuration("config.yaml")

# ==============================================================================
# Step 3: Executing the Pipeline
# ==============================================================================
if __name__ == "__main__":
    if df_raw is not None and not df_raw.empty and config is not None:
        print("\n" + "="*80)
        print("INITIATING PHI FRAMEWORK ORCHESTRATOR")
        print("="*80)
        
        # Execute the top-level orchestrator function
        # Note: execute_full_phi_framework is defined in the Jupyter Notebook
        final_artifacts = execute_full_phi_framework(
            df_raw=df_raw,
            config=config
        )
        
        print("\n" + "="*80)
        print("STUDY EXECUTION COMPLETE. EXTRACTING ARTIFACTS.")
        print("="*80)
        
        # 1. Accessing the Final Prioritized Shortlist
        bundle = final_artifacts.get("reproducibility_bundle")
        if bundle and bundle.ranked_phi_artifact:
            df_ranked = bundle.ranked_phi_artifact.data
            print("\n[Top 5 High-Priority Suppliers for Audit Triage]")
            print(df_ranked[['pseudonymous_id', 'PHI', 'risk_tier', 'M', 'A', 'T', 'D']].head(5).to_string(index=False))
            
        # 2. Accessing Sectoral Risk Distribution
        sectoral_artifact = final_artifacts.get("sectoral_risk_artifact")
        if sectoral_artifact:
            df_sectoral = sectoral_artifact.directorate_summary
            print("\n[Sectoral Risk Distribution (Management Triage)]")
            print(df_sectoral[['Directorate', 'Total', 'High', 'Prevalence_High']].to_string(index=False))

        # 3. Accessing the Reproduction Manifest
        manifest = final_artifacts.get("reproduction_manifest")
        if manifest:
            print("\n" + "="*80)
            print("FINAL REPRODUCTION MANIFEST (AUDIT TRAIL)")
            print("="*80)
            print(manifest.to_json(indent=2))
    else:
        print("Fatal Error: Missing data or configuration. Cannot proceed.")
```

## Output Structure

The pipeline returns a comprehensive dictionary containing four key artifacts:
-   **`reproducibility_bundle`**: The core state object containing all intermediate DataFrames, scaling parameters, and GMM models.
-   **`threshold_anchoring_artifact`**: The inferential statistics, endogenous peaks, and permutation test results.
-   **`sectoral_risk_artifact`**: The directorate-level aggregation for management triage.
-   **`reproduction_manifest`**: A JSON-serializable audit trail verifying the execution against the manuscript's diagnostic checkpoints.

## Project Structure

```
framework_for_high_volume_procurement_oversight_and_decision_support/
│
├── framework_for_high_volume_procurement_oversight_and_decision_support_draft.ipynb  # Main implementation notebook
├── config.yaml                                                                       # Master configuration file
├── requirements.txt                                                                  # Python package dependencies
│
├── LICENSE                                                                           # MIT Project License File
└── README.md                                                                         # This file
```

## Customization

The pipeline is highly customizable via the `config.yaml` file. Users can modify study parameters such as:
-   **Identity Harmonisation:** Adjust the `tf_idf_cosine_similarity` or `token_set_ratio` thresholds to tighten or loosen entity resolution.
-   **GMM Hyperparameters:** Modify the `max_components_cap` or the `regularizer_lambda` to alter the sensitivity of the regime extraction.
-   **Risk Tiering:** Adjust the `high_phi_percentile_ge` to expand or contract the targeted audit shortlist.
-   **Inferential Rigor:** Increase the `permutation_iterations` for higher statistical confidence in the threshold anchoring analysis.

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request with a clear description of your changes. Adherence to PEP 8, strict type hinting, and the 1:1 inline comment-to-code-line ratio is required.

## Recommended Extensions

Future extensions, as suggested by the author, could include:
-   **Longitudinal Stability Analysis:** Investigating the stability of PHI signatures over longer fiscal cycles (e.g., multi-year panels).
-   **Semi-Supervised Learning:** Integrating investigator feedback on High-PHI alerts to calibrate the weighting of the $M, A, T, D$ components against confirmed institutional outcomes.
-   **Cross-Institutional Calibration:** Applying the framework to diverse municipal or federal datasets to establish generalized baseline thresholds.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this code or the methodology in your research, please cite the original paper:

```bibtex
@article{christodoulides2026payment,
  title={The Payment Heterogeneity Index: An Integrated Unsupervised Framework for High-Volume Procurement Oversight and Decision Support},
  author={Christodoulides, Kyriakos},
  journal={arXiv preprint arXiv:2605.12547},
  year={2026}
}
```

For the implementation itself, you may cite this repository:
```
Chirinda, C. (2026). The Payment Heterogeneity Index: An Integrated Unsupervised Framework.
GitHub repository: https://github.com/chirindaopensource/framework_for_high_volume_procurement_oversight_and_decision_support
```

## Acknowledgments

-   Credit to **Kyriakos Christodoulides** for the foundational research that forms the entire basis for this computational replication.
-   This project is built upon the exceptional tools provided by the open-source community. Sincere thanks to the developers of the scientific Python ecosystem, particularly the **NumPy**, **Pandas**, **Scikit-Learn**, and **NetworkX** contributors.

--

*This README was generated based on the structure and content of the `framework_for_high_volume_procurement_oversight_and_decision_support_draft.ipynb` notebook and follows best practices for research software documentation.*
