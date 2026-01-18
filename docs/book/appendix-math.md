---
title: "Appendix A: Mathematical Foundations"
description: "Comprehensive reference for retrieval metrics, statistical testing, loss functions, and optimization techniques used throughout the book."
authors:
  - Jason Liu
date: 2025-01-17
tags:
  - reference
  - mathematics
  - metrics
  - statistics
  - loss functions
---

# Appendix A: Mathematical Foundations

This appendix provides complete mathematical definitions, derivations, and examples for all quantitative concepts used throughout the book. Use this as a reference when you need precise formulas or want to understand the theory behind the metrics.

---

## Retrieval Metrics

### Precision

**Definition**: Of the documents retrieved, what fraction were relevant?

$$\text{Precision} = \frac{|\text{Relevant} \cap \text{Retrieved}|}{|\text{Retrieved}|} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$$

**Precision@K**: Precision calculated for the top K results only.

$$\text{Precision@}K = \frac{|\text{Relevant} \cap \text{Retrieved}_K|}{K}$$

**Example**: You retrieve 10 documents, 4 are relevant.

$$\text{Precision@10} = \frac{4}{10} = 0.40 = 40\%$$

**Interpretation**: Higher precision means fewer irrelevant results. Important when users have limited time to review results or when irrelevant results cause confusion.

---

### Recall

**Definition**: Of all relevant documents that exist, what fraction did we retrieve?

$$\text{Recall} = \frac{|\text{Relevant} \cap \text{Retrieved}|}{|\text{Relevant}|} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$$

**Recall@K**: Recall calculated when retrieving K documents.

$$\text{Recall@}K = \frac{|\text{Relevant} \cap \text{Retrieved}_K|}{|\text{Relevant}|}$$

**Example**: There are 8 relevant documents total, you retrieve 10 documents, 4 of which are relevant.

$$\text{Recall@10} = \frac{4}{8} = 0.50 = 50\%$$

**Interpretation**: Higher recall means fewer missed relevant documents. Critical for safety-critical applications (medical, legal) where missing information has serious consequences.

---

### F1 Score

**Definition**: The harmonic mean of precision and recall, providing a single metric that balances both.

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot \text{TP}}{2 \cdot \text{TP} + \text{FP} + \text{FN}}$$

**Why harmonic mean?** The harmonic mean penalizes extreme imbalances. If precision is 100% but recall is 10%, arithmetic mean gives 55%, but harmonic mean gives 18%—better reflecting the poor overall performance.

**Derivation**:

Starting from the definition of harmonic mean of two numbers $a$ and $b$:

$$H = \frac{2ab}{a + b}$$

Substituting $a = \text{Precision}$ and $b = \text{Recall}$:

$$F_1 = \frac{2 \cdot P \cdot R}{P + R}$$

**Example**: Precision = 0.40, Recall = 0.50

$$F_1 = \frac{2 \cdot 0.40 \cdot 0.50}{0.40 + 0.50} = \frac{0.40}{0.90} = 0.444 = 44.4\%$$

---

### F-beta Score

**Definition**: A generalization of F1 that allows weighting precision vs recall.

$$F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}$$

**Common values**:

| Beta | Emphasis | Use Case |
|------|----------|----------|
| $\beta = 0.5$ | Precision 2x more important | Customer-facing search (clean results) |
| $\beta = 1$ | Equal weight (F1) | General purpose |
| $\beta = 2$ | Recall 2x more important | Legal/medical (cannot miss relevant) |

**Example**: $\beta = 2$, Precision = 0.40, Recall = 0.50

$$F_2 = (1 + 4) \cdot \frac{0.40 \cdot 0.50}{4 \cdot 0.40 + 0.50} = 5 \cdot \frac{0.20}{2.10} = 0.476 = 47.6\%$$

---

### Mean Reciprocal Rank (MRR)

**Definition**: The average of reciprocal ranks of the first relevant result across queries.

$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

Where $\text{rank}_i$ is the position of the first relevant document for query $i$.

**Example**: Three queries with first relevant document at positions 1, 3, and 2.

$$\text{MRR} = \frac{1}{3} \left( \frac{1}{1} + \frac{1}{3} + \frac{1}{2} \right) = \frac{1}{3} \cdot 1.833 = 0.611$$

**Interpretation**: MRR measures how quickly users find something useful. An MRR of 0.5 means on average the first relevant result appears around position 2.

**Limitations**: MRR only considers the first relevant result. If multiple relevant documents matter, use NDCG or MAP instead.

---

### Normalized Discounted Cumulative Gain (NDCG)

**Definition**: A ranking metric that accounts for graded relevance and position.

**Step 1: Cumulative Gain (CG)**

$$\text{CG}_K = \sum_{i=1}^{K} \text{rel}_i$$

Where $\text{rel}_i$ is the relevance score of the document at position $i$.

**Step 2: Discounted Cumulative Gain (DCG)**

$$\text{DCG}_K = \sum_{i=1}^{K} \frac{\text{rel}_i}{\log_2(i + 1)}$$

The logarithmic discount penalizes relevant documents appearing lower in the ranking.

**Step 3: Ideal DCG (IDCG)**

$$\text{IDCG}_K = \sum_{i=1}^{K} \frac{\text{rel}_i^*}{\log_2(i + 1)}$$

Where $\text{rel}_i^*$ is the relevance in the ideal (perfectly sorted) ranking.

**Step 4: Normalized DCG**

$$\text{NDCG}_K = \frac{\text{DCG}_K}{\text{IDCG}_K}$$

**Example**: Retrieved documents with relevance scores [3, 2, 0, 1, 2] (scale 0-3).

$$\text{DCG}_5 = \frac{3}{\log_2(2)} + \frac{2}{\log_2(3)} + \frac{0}{\log_2(4)} + \frac{1}{\log_2(5)} + \frac{2}{\log_2(6)}$$

$$= \frac{3}{1} + \frac{2}{1.585} + \frac{0}{2} + \frac{1}{2.322} + \frac{2}{2.585}$$

$$= 3 + 1.262 + 0 + 0.431 + 0.774 = 5.467$$

Ideal ranking: [3, 2, 2, 1, 0]

$$\text{IDCG}_5 = \frac{3}{1} + \frac{2}{1.585} + \frac{2}{2} + \frac{1}{2.322} + \frac{0}{2.585}$$

$$= 3 + 1.262 + 1 + 0.431 + 0 = 5.693$$

$$\text{NDCG}_5 = \frac{5.467}{5.693} = 0.960$$

---

### Mean Average Precision (MAP)

**Definition**: The mean of Average Precision (AP) scores across all queries.

**Average Precision for a single query**:

$$\text{AP} = \frac{1}{|\text{Relevant}|} \sum_{k=1}^{K} \text{Precision@}k \cdot \text{rel}(k)$$

Where $\text{rel}(k) = 1$ if document at position $k$ is relevant, 0 otherwise.

**Mean Average Precision**:

$$\text{MAP} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \text{AP}_q$$

**Example**: Query with relevant documents at positions 1, 3, 5 out of 10 retrieved.

$$\text{AP} = \frac{1}{3} \left( \frac{1}{1} + \frac{2}{3} + \frac{3}{5} \right) = \frac{1}{3} (1 + 0.667 + 0.6) = 0.756$$

**Interpretation**: MAP rewards systems that rank relevant documents higher. Unlike MRR, it considers all relevant documents, not just the first.

---

## Statistical Testing

### Chi-Square Test for Independence

**Use case**: Test whether two categorical variables are independent (e.g., query type vs success/failure).

**Formula**:

$$\chi^2 = \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$

Where:
- $O_{ij}$ = observed frequency in cell $(i,j)$
- $E_{ij}$ = expected frequency = $\frac{\text{row}_i \text{ total} \times \text{column}_j \text{ total}}{\text{grand total}}$

**Degrees of freedom**: $(r - 1)(c - 1)$ where $r$ = rows, $c$ = columns.

**Example**: Testing if query type affects retrieval success.

|  | Success | Failure | Total |
|--|---------|---------|-------|
| Factual | 80 | 20 | 100 |
| Procedural | 60 | 40 | 100 |
| Total | 140 | 60 | 200 |

Expected values:
- $E_{11} = \frac{100 \times 140}{200} = 70$
- $E_{12} = \frac{100 \times 60}{200} = 30$
- $E_{21} = \frac{100 \times 140}{200} = 70$
- $E_{22} = \frac{100 \times 60}{200} = 30$

$$\chi^2 = \frac{(80-70)^2}{70} + \frac{(20-30)^2}{30} + \frac{(60-70)^2}{70} + \frac{(40-30)^2}{30}$$

$$= \frac{100}{70} + \frac{100}{30} + \frac{100}{70} + \frac{100}{30} = 1.43 + 3.33 + 1.43 + 3.33 = 9.52$$

With df = 1 and $\alpha = 0.05$, critical value = 3.84. Since 9.52 > 3.84, we reject the null hypothesis—query type affects success rate.

---

### Two-Sample t-Test

**Use case**: Compare means of two groups (e.g., recall before vs after a change).

**Formula** (assuming unequal variances, Welch's t-test):

$$t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$$

**Degrees of freedom** (Welch-Satterthwaite approximation):

$$\text{df} = \frac{\left(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}\right)^2}{\frac{(s_1^2/n_1)^2}{n_1-1} + \frac{(s_2^2/n_2)^2}{n_2-1}}$$

**Example**: Comparing recall before (n=50, mean=0.65, std=0.15) and after (n=50, mean=0.72, std=0.12).

$$t = \frac{0.72 - 0.65}{\sqrt{\frac{0.15^2}{50} + \frac{0.12^2}{50}}} = \frac{0.07}{\sqrt{0.00045 + 0.000288}} = \frac{0.07}{0.0272} = 2.57$$

With df approximately 92 and $\alpha = 0.05$, critical value is approximately 1.99. Since 2.57 > 1.99, the improvement is statistically significant.

---

### Confidence Intervals

**Definition**: A range of values that contains the true parameter with a specified probability.

**For a proportion** (e.g., success rate):

$$\hat{p} \pm z_{\alpha/2} \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$$

Where:
- $\hat{p}$ = sample proportion
- $z_{\alpha/2}$ = z-score for confidence level (1.96 for 95%)
- $n$ = sample size

**Example**: 75 successes out of 100 queries.

$$\hat{p} = 0.75$$

$$\text{95\% CI} = 0.75 \pm 1.96 \sqrt{\frac{0.75 \times 0.25}{100}} = 0.75 \pm 1.96 \times 0.0433 = 0.75 \pm 0.085$$

$$\text{95\% CI} = [0.665, 0.835]$$

**For a mean**:

$$\bar{x} \pm t_{\alpha/2, n-1} \frac{s}{\sqrt{n}}$$

**Example**: Mean recall of 0.72 with std 0.12 from 50 samples.

$$\text{95\% CI} = 0.72 \pm 2.01 \times \frac{0.12}{\sqrt{50}} = 0.72 \pm 0.034 = [0.686, 0.754]$$

---

### Sample Size Calculations

**For comparing two proportions** (power analysis):

$$n = \frac{(z_{\alpha/2} + z_\beta)^2 [p_1(1-p_1) + p_2(1-p_2)]}{(p_1 - p_2)^2}$$

Where:
- $z_{\alpha/2}$ = z-score for significance level (1.96 for $\alpha = 0.05$)
- $z_\beta$ = z-score for power (0.84 for 80% power)
- $p_1, p_2$ = expected proportions in each group

**Example**: Detecting improvement from 65% to 75% recall with 80% power.

$$n = \frac{(1.96 + 0.84)^2 [0.65 \times 0.35 + 0.75 \times 0.25]}{(0.75 - 0.65)^2}$$

$$= \frac{7.84 \times [0.2275 + 0.1875]}{0.01} = \frac{7.84 \times 0.415}{0.01} = 325.4$$

You need approximately 326 samples per group (652 total) to detect this difference.

**Rule of thumb for quick estimation**:

| Effect Size | Samples Needed (per group) |
|-------------|---------------------------|
| Large (15%+ difference) | 100-200 |
| Medium (5-15% difference) | 200-500 |
| Small (2-5% difference) | 500-2000 |

---

### Effect Size (Cohen's d)

**Definition**: A standardized measure of the magnitude of difference between two groups.

$$d = \frac{\bar{X}_1 - \bar{X}_2}{s_{\text{pooled}}}$$

Where:

$$s_{\text{pooled}} = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}}$$

**Interpretation**:

| Cohen's d | Effect Size |
|-----------|-------------|
| 0.2 | Small |
| 0.5 | Medium |
| 0.8 | Large |

**Example**: Before (mean=0.65, std=0.15, n=50), After (mean=0.72, std=0.12, n=50).

$$s_{\text{pooled}} = \sqrt{\frac{49 \times 0.0225 + 49 \times 0.0144}{98}} = \sqrt{\frac{1.1025 + 0.7056}{98}} = \sqrt{0.0184} = 0.136$$

$$d = \frac{0.72 - 0.65}{0.136} = 0.515$$

This is a medium effect size—the improvement is meaningful in practical terms.

---

## Loss Functions

### Triplet Loss

**Definition**: Ensures the anchor is closer to the positive than to the negative by at least a margin.

$$\mathcal{L}_{\text{triplet}} = \max(0, \|f(a) - f(p)\|^2 - \|f(a) - f(n)\|^2 + \alpha)$$

Where:
- $f(a)$ = anchor embedding (query)
- $f(p)$ = positive embedding (relevant document)
- $f(n)$ = negative embedding (irrelevant document)
- $\alpha$ = margin (typically 0.5-1.0)

**Intuition**: The loss is zero when the positive is closer than the negative by at least the margin. Otherwise, the loss pushes the model to move the positive closer and the negative farther.

**Derivation**:

We want: $\|f(a) - f(p)\|^2 + \alpha < \|f(a) - f(n)\|^2$

Rearranging: $\|f(a) - f(p)\|^2 - \|f(a) - f(n)\|^2 + \alpha < 0$

The hinge function $\max(0, x)$ ensures we only penalize violations:

$$\mathcal{L} = \max(0, d_{ap} - d_{an} + \alpha)$$

**Example**: $d_{ap} = 0.3$, $d_{an} = 0.5$, $\alpha = 0.5$

$$\mathcal{L} = \max(0, 0.3 - 0.5 + 0.5) = \max(0, 0.3) = 0.3$$

The positive needs to be 0.3 units closer to satisfy the margin constraint.

---

### InfoNCE Loss (Contrastive Loss)

**Definition**: Treats retrieval as a classification problem—identify the positive among many negatives.

$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(a, p) / \tau)}{\sum_{i=0}^{N} \exp(\text{sim}(a, x_i) / \tau)}$$

Where:
- $\text{sim}(a, p)$ = similarity between anchor and positive (typically dot product or cosine)
- $\tau$ = temperature parameter (typically 0.05-0.1)
- $N$ = number of negatives
- $x_0 = p$ (positive), $x_1, ..., x_N$ are negatives

**Derivation**:

InfoNCE is derived from noise contrastive estimation. The goal is to maximize the probability that the positive is selected from a set of candidates:

$$P(\text{positive} | a) = \frac{\exp(\text{sim}(a, p) / \tau)}{\sum_{i} \exp(\text{sim}(a, x_i) / \tau)}$$

Taking the negative log gives us the cross-entropy loss.

**Temperature effect**:
- Lower $\tau$ (e.g., 0.05): Sharper distribution, harder to learn, more discriminative
- Higher $\tau$ (e.g., 0.5): Softer distribution, easier to learn, less discriminative

**Example**: $\text{sim}(a, p) = 0.8$, $\text{sim}(a, n_1) = 0.3$, $\text{sim}(a, n_2) = 0.2$, $\tau = 0.1$

$$\mathcal{L} = -\log \frac{\exp(0.8/0.1)}{\exp(0.8/0.1) + \exp(0.3/0.1) + \exp(0.2/0.1)}$$

$$= -\log \frac{\exp(8)}{\exp(8) + \exp(3) + \exp(2)}$$

$$= -\log \frac{2981}{2981 + 20.1 + 7.4} = -\log \frac{2981}{3008.5} = -\log(0.991) = 0.009$$

The loss is low because the positive has much higher similarity than the negatives.

---

### Multiple Negatives Ranking Loss

**Definition**: A variant of InfoNCE that uses other examples in the batch as negatives.

For a batch of $(q_i, d_i)$ pairs:

$$\mathcal{L} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp(\text{sim}(q_i, d_i) / \tau)}{\sum_{j=1}^{B} \exp(\text{sim}(q_i, d_j) / \tau)}$$

**Advantage**: No need to explicitly mine negatives—the batch provides them automatically. With batch size $B$, you get $B-1$ negatives per example.

**Effective negatives**: With batch size 32, each query sees 31 negatives. Larger batches provide harder negatives on average.

---

### Cross-Entropy Loss for Re-Rankers

**Definition**: Standard classification loss for binary relevance.

$$\mathcal{L}_{\text{CE}} = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$$

Where:
- $y$ = true label (1 for relevant, 0 for not relevant)
- $\hat{y}$ = predicted probability

**For graded relevance** (multi-class):

$$\mathcal{L}_{\text{CE}} = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)$$

Where $C$ is the number of relevance grades.

---

## Optimization

### Learning Rate Schedules

**Linear Warmup**:

$$\text{lr}(t) = \text{lr}_{\text{base}} \cdot \frac{t}{T_{\text{warmup}}}$$

For $t < T_{\text{warmup}}$, then constant or decay.

**Cosine Annealing**:

$$\text{lr}(t) = \text{lr}_{\text{min}} + \frac{1}{2}(\text{lr}_{\text{max}} - \text{lr}_{\text{min}})\left(1 + \cos\left(\frac{t \cdot \pi}{T}\right)\right)$$

**Warmup + Cosine Decay** (commonly used):

1. Warmup phase: Linear increase from $0.1 \times \text{lr}_{\text{base}}$ to $\text{lr}_{\text{base}}$
2. Decay phase: Cosine decay from $\text{lr}_{\text{base}}$ to near zero

**Typical values for embedding fine-tuning**:
- Base learning rate: $2 \times 10^{-5}$
- Warmup steps: 10% of total steps
- Final learning rate: $1 \times 10^{-6}$

---

### Gradient Accumulation

**Purpose**: Simulate larger batch sizes when GPU memory is limited.

**Effective batch size**:

$$B_{\text{effective}} = B_{\text{actual}} \times A$$

Where $A$ is the number of accumulation steps.

**Update rule**:

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{1}{A} \sum_{i=1}^{A} \nabla_\theta \mathcal{L}_i$$

**Example**: With batch size 8 and accumulation steps 4, effective batch size is 32.

**Trade-off**: Gradient accumulation increases training time linearly with accumulation steps but allows training with larger effective batches.

---

### Adam Optimizer

**Definition**: Adaptive learning rate optimizer combining momentum and RMSprop.

**Update equations**:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$
$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Where:
- $g_t$ = gradient at step $t$
- $m_t$ = first moment estimate (momentum)
- $v_t$ = second moment estimate (adaptive learning rate)
- $\beta_1$ = 0.9 (momentum decay)
- $\beta_2$ = 0.999 (RMSprop decay)
- $\epsilon$ = $10^{-8}$ (numerical stability)

**AdamW** (weight decay variant):

$$\theta_{t+1} = \theta_t - \eta \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t\right)$$

Where $\lambda$ is the weight decay coefficient (typically 0.01).

---

## Similarity Metrics

### Cosine Similarity

**Definition**: Measures the cosine of the angle between two vectors.

$$\text{cos\_sim}(a, b) = \frac{a \cdot b}{\|a\| \|b\|} = \frac{\sum_{i=1}^{n} a_i b_i}{\sqrt{\sum_{i=1}^{n} a_i^2} \sqrt{\sum_{i=1}^{n} b_i^2}}$$

**Range**: [-1, 1] where 1 = identical direction, 0 = orthogonal, -1 = opposite direction.

**For normalized vectors** ($\|a\| = \|b\| = 1$):

$$\text{cos\_sim}(a, b) = a \cdot b$$

**Example**: $a = [0.6, 0.8]$, $b = [0.8, 0.6]$

$$\text{cos\_sim} = \frac{0.6 \times 0.8 + 0.8 \times 0.6}{\sqrt{0.36 + 0.64} \sqrt{0.64 + 0.36}} = \frac{0.96}{1 \times 1} = 0.96$$

---

### Euclidean Distance

**Definition**: The straight-line distance between two points.

$$d(a, b) = \|a - b\|_2 = \sqrt{\sum_{i=1}^{n} (a_i - b_i)^2}$$

**Relationship to cosine similarity** (for normalized vectors):

$$\|a - b\|^2 = 2(1 - \text{cos\_sim}(a, b))$$

**When to use**:
- Cosine similarity: When magnitude does not matter (most embedding use cases)
- Euclidean distance: When magnitude matters or vectors are already normalized

---

### Dot Product

**Definition**: Sum of element-wise products.

$$a \cdot b = \sum_{i=1}^{n} a_i b_i$$

**Relationship to cosine similarity**:

$$a \cdot b = \|a\| \|b\| \cos(\theta)$$

For normalized vectors, dot product equals cosine similarity.

**Advantage**: Faster to compute than cosine similarity (no normalization step).

---

## Reciprocal Rank Fusion (RRF)

**Definition**: Combines rankings from multiple retrieval systems.

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + r(d)}$$

Where:
- $R$ = set of rankings
- $r(d)$ = rank of document $d$ in ranking $r$
- $k$ = constant (typically 60)

**Example**: Document appears at rank 3 in semantic search and rank 7 in lexical search.

$$\text{RRF} = \frac{1}{60 + 3} + \frac{1}{60 + 7} = \frac{1}{63} + \frac{1}{67} = 0.0159 + 0.0149 = 0.0308$$

**Why k=60?** This value was empirically determined to work well across many datasets. It dampens the impact of very high ranks while still giving credit to top results.

**Weighted RRF**:

$$\text{RRF}(d) = \sum_{r \in R} w_r \cdot \frac{1}{k + r(d)}$$

Where $w_r$ is the weight for ranking system $r$.

---

## Quick Reference Tables

### Retrieval Metrics Summary

| Metric | Formula | Range | Higher is Better |
|--------|---------|-------|------------------|
| Precision@K | $\frac{\text{Relevant} \cap \text{Top-K}}{K}$ | [0, 1] | Yes |
| Recall@K | $\frac{\text{Relevant} \cap \text{Top-K}}{\text{Total Relevant}}$ | [0, 1] | Yes |
| F1 | $\frac{2 \cdot P \cdot R}{P + R}$ | [0, 1] | Yes |
| MRR | $\frac{1}{|Q|} \sum \frac{1}{\text{rank}_i}$ | [0, 1] | Yes |
| NDCG@K | $\frac{\text{DCG}_K}{\text{IDCG}_K}$ | [0, 1] | Yes |
| MAP | $\frac{1}{|Q|} \sum \text{AP}_q$ | [0, 1] | Yes |

### Statistical Tests Summary

| Test | Use Case | Assumptions |
|------|----------|-------------|
| Chi-square | Categorical vs categorical | Expected counts > 5 |
| t-test | Compare two means | Approximately normal, n > 30 |
| Paired t-test | Before/after comparison | Paired observations |
| Mann-Whitney U | Compare two groups | Non-parametric alternative to t-test |

### Loss Functions Summary

| Loss | Data Format | Best For |
|------|-------------|----------|
| Triplet | (anchor, positive, negative) | Explicit hard negatives |
| InfoNCE | (anchor, positive, negatives[]) | Large negative sets |
| Multiple Negatives | (query, document) pairs | Efficient batch training |
| Cross-Entropy | (query, document, label) | Re-ranker training |

---

## Implementation Reference

### Python Functions for Key Metrics

```python
import numpy as np
from typing import List, Set

def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Calculate Precision@K."""
    retrieved_k = retrieved[:k]
    relevant_retrieved = sum(1 for doc in retrieved_k if doc in relevant)
    return relevant_retrieved / k if k > 0 else 0.0

def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Calculate Recall@K."""
    retrieved_k = set(retrieved[:k])
    relevant_retrieved = len(retrieved_k & relevant)
    return relevant_retrieved / len(relevant) if relevant else 0.0

def f1_score(precision: float, recall: float) -> float:
    """Calculate F1 score."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def mrr(rankings: List[List[str]], relevant_sets: List[Set[str]]) -> float:
    """Calculate Mean Reciprocal Rank."""
    reciprocal_ranks = []
    for ranking, relevant in zip(rankings, relevant_sets):
        for i, doc in enumerate(ranking, 1):
            if doc in relevant:
                reciprocal_ranks.append(1.0 / i)
                break
        else:
            reciprocal_ranks.append(0.0)
    return np.mean(reciprocal_ranks)

def dcg_at_k(relevances: List[float], k: int) -> float:
    """Calculate Discounted Cumulative Gain@K."""
    relevances_k = relevances[:k]
    gains = [rel / np.log2(i + 2) for i, rel in enumerate(relevances_k)]
    return sum(gains)

def ndcg_at_k(relevances: List[float], k: int) -> float:
    """Calculate Normalized DCG@K."""
    dcg = dcg_at_k(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)
    return dcg / idcg if idcg > 0 else 0.0
```

### Statistical Testing Functions

```python
import numpy as np
from typing import List
from scipy import stats

def welch_ttest(group1: List[float], group2: List[float]) -> dict:
    """Perform Welch's t-test for two independent samples."""
    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant_at_05': p_value < 0.05,
        'significant_at_01': p_value < 0.01
    }

def confidence_interval_proportion(
    successes: int, 
    total: int, 
    confidence: float = 0.95
) -> tuple:
    """Calculate confidence interval for a proportion."""
    p = successes / total
    z = stats.norm.ppf((1 + confidence) / 2)
    margin = z * np.sqrt(p * (1 - p) / total)
    return (p - margin, p + margin)

def sample_size_two_proportions(
    p1: float, 
    p2: float, 
    alpha: float = 0.05, 
    power: float = 0.80
) -> int:
    """Calculate required sample size per group."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    
    numerator = (z_alpha + z_beta) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2))
    denominator = (p1 - p2) ** 2
    
    return int(np.ceil(numerator / denominator))
```

---

## Navigation

- **Previous**: [Chapter 9: Context Window Management](chapter9.md)
- **Next**: [Appendix B: Algorithms Reference](appendix-algorithms.md)
- **Reference**: [Glossary](glossary.md) | [Quick Reference](quick-reference.md)
- **Book Index**: [Book Overview](index.md)
