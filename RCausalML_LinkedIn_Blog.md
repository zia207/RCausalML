# Introducing RCausalML: An R Package for Machine Learning-Based Causal Inference

*By Zia Ahmed, Upatta Analytics*

---

I'm thrilled to announce the release of **RCausalML** — an open-source R package that bridges the gap between machine learning's predictive power and causal inference's ability to answer the most important question in data science: **not just *what* will happen, but *why*, and *what would change* if we intervened?**

This is my first R package on machine learning-based causal inference and I'd love to share what it does and why I built it.

---

## The Problem: Correlation Is Not Enough

Modern machine learning is extraordinarily good at finding patterns in data. But patterns alone can't tell us whether a marketing campaign actually *caused* more sales, whether a drug *caused* patients to recover, or whether a policy *caused* economic growth. For that, we need causal inference — and that's exactly where RCausalML comes in.

---

## What Is RCausalML?

**RCausalML** is an R package for estimating causal effects from experimental or observational data. Inspired by Python's [CausalML](https://github.com/uber/causalml) and Microsoft's [EconML](https://github.com/py-why/EconML), it brings state-of-the-art Conditional Average Treatment Effect (CATE) estimation to the R ecosystem, with a clean and unified API.

**Package Website:** [https://zia207.github.io/RCausalML/](https://zia207.github.io/RCausalML/)

**GitHub Repository:** [https://github.com/zia207/RCausalML](https://github.com/zia207/RCausalML)

**Installation:**
```r
remotes::install_github("zia207/RCausalML")
```

---

## Key Capabilities

### 1. Meta-Learners

The package implements five powerful meta-learner algorithms for estimating heterogeneous treatment effects:

- **S-Learner** — single-model approach
- **T-Learner** — treatment-specific models
- **X-Learner** — cross-fitting for improved accuracy in imbalanced settings
- **R-Learner** — residual-based approach for partially linear models
- **DR-Learner** — doubly robust estimation for extra protection against model misspecification

All meta-learners support confidence intervals and bootstrap estimation with optional parallel processing.

### 2. Double Machine Learning (DML)

For high-dimensional settings with confounders, RCausalML provides a comprehensive suite of DML estimators including LinearDML, SparseLinearDML, KernelDML, NonParamDML, and CausalForestDML — as well as instrumental variable variants (OrthoIVLearner, DMLIVLearner). Integration with the R `DoubleML` package enables Difference-in-Differences, Partially Linear Regression, and Partially Linear IV models.

### 3. Tree-Based Methods

- Uplift Random Forests (KL, ED, Chi-square, and contextual treatment selection)
- Causal Trees and Interaction Trees
- Causal Survival Forest
- Causal XGBoost
- Multi-Arm Causal Boosting for K ≥ 2 treatment arms

### 4. Policy Learning & Interpretation

Go beyond *estimating* treatment effects to *acting* on them:

- **Policy Learner** — determines optimal treatment assignment using doubly robust weighted classification
- **DR Policy Tree / Forest** — EconML-style cross-fitted outcome models with policy optimization
- **CATE Interpreter** — single-tree summaries of treatment effect heterogeneity
- **Policy Interpreter** — identifies subgroups most eligible for treatment

### 5. Deep Causal Machine Learning

For researchers pushing the frontier, RCausalML integrates neural methods via the `torch` package:

- **Treatment effect estimators:** TARNet, CFRNet, DragonNet, GANITE
- **Generative models:** CEVAE, CausalGAN, Deep Structural Causal Models
- **Time-series causal discovery:** Neural Granger Causality, RNNs, LSTMs, Graph Neural Networks, Transformer-based models
- **Counterfactual reasoning:** DeepSynth, CRN, G-Net

### 6. Causal Structure Learning

Discover the underlying causal graph from data with:

- **NOTEARS** (linear and nonlinear)
- **DAG-GNN** (graph neural network approach)
- **GraN-DAG** (gradient-based neural DAG learner)
- **CASTLE**

All accessible through a single unified `causalStructureML()` function.

---

## SHAP Integration & Interpretability

Understanding *why* a treatment effect is heterogeneous is as important as estimating it. RCausalML integrates seamlessly with the `kernelshap` and `shapviz` packages, enabling full SHAP-based explanations: feature importance plots, dependence plots, waterfall charts, and force plots — all computed from any fitted CATE model.

---

## AutoML for Nuisance Models

The `EconAutoMLConfig` and `AutomatedMLModel` system automatically selects the best nuisance models (outcome model, treatment model) from a candidate library — `lm`, `glmnet`, `ranger`, `rpart` — so you can focus on causal questions rather than model tuning.

---

## Validated Performance

Benchmark results on synthetic data show strong accuracy:

- **LinearDML** achieves **0.98 correlation** with the true CATE on Nie & Wager (2018) style benchmarks
- **Policy Learners** achieve **100% agreement** with oracle policies on simulated scenarios

---

## Rich Tutorial Library

The package includes **25+ Quarto notebooks** covering:

- Getting started with meta-learners
- Double ML from basics to advanced IV
- Uplift modeling for marketing and clinical applications
- Deep causal ML with neural networks
- Causal structure learning
- Policy optimization and counterfactual reasoning

Explore the full documentation at: [https://zia207.github.io/RCausalML/](https://zia207.github.io/RCausalML/)

---

## Who Is This For?

RCausalML is designed for:

- **Data scientists** who want to move beyond predictive modeling to answer causal questions
- **Econometricians and statisticians** who want to leverage modern ML methods
- **Researchers in health, economics, and social sciences** doing treatment effect estimation
- **Industry practitioners** in marketing, pricing, and policy optimization who need personalized intervention strategies

---

## What's Next?

Version 0.3.0 is now available on GitHub. Upcoming versions will expand support for:

- Panel data and time-varying treatments
- Federated causal inference
- Integration with additional AutoML backends
- Extended causal graph visualization tools

---

## Get Involved

I'd love to hear your feedback, use cases, and contributions!

- **Website:** [https://zia207.github.io/RCausalML/](https://zia207.github.io/RCausalML/)
- **GitHub:** [https://github.com/zia207/RCausalML](https://github.com/zia207/RCausalML)
- **Install:** `remotes::install_github("zia207/RCausalML")`

If you find this package useful, please give it a ⭐ on GitHub and share it with your network. Causal inference is the next frontier in responsible, actionable data science — and I hope RCausalML makes it more accessible to the R community.

---

*#RStats #CausalInference #MachineLearning #DataScience #OpenSource #EconML #CausalML #TreatmentEffect #Statistics #R #AI*

---

*Zia Ahmed | Upatta Analytics*
