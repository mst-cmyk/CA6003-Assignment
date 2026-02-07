# Executive Summary | 执行摘要

## Titanic Survival Analysis: Data Governance, Bias, and Preparation Impact
## 泰坦尼克号生存分析：数据治理、偏差与准备影响

---

### Team | 团队
**[Names here]**

**Course | 课程:** CA6003 - Best Practices in Data Governance, Preparation and Analytics

---

## Research Question | 研究问题

**EN:** How do data preparation choices affect the reliability and fairness of Titanic survival predictions?

**中文:** 不同的数据准备策略如何影响泰坦尼克号生存预测的可靠性与公平性？

---

## Dataset | 数据集

**EN:** Titanic passenger data (891 passengers, 12 features) with target `Survived` (0 = died, 1 = survived).  
Key missingness: **Age 19.9%**, **Cabin 77.1%**, Embarked 0.2%.

**中文:** 泰坦尼克号乘客数据（891人，12特征），目标变量为 `Survived`（0=死亡，1=生存）。  
关键缺失：**Age 19.9%**，**Cabin 77.1%**，Embarked 0.2%。

---

## Method Overview | 方法概览

### 1. Data Governance & Profiling | 数据治理与画像
**EN:** Checked data types, missingness patterns, imbalance (61.6% died vs 38.4% survived), and outliers (Fare right-skewed).  
**中文:** 检查数据类型、缺失模式、类别不平衡（死亡61.6% vs 生存38.4%）与异常值（Fare右偏）。

### 2. Bias Analysis (6 Types) | 偏差分析（6类）
**EN:** Gender bias, class bias, age bias, MNAR missingness, survivorship bias, Simpson’s paradox interaction.  
**中文:** 性别偏差、阶层偏差、年龄偏差、MNAR缺失偏差、幸存者偏差、辛普森悖论交互偏差。

### 3. Preparation Scenarios | 准备方案对比
**Scenario A – Minimal:** drop missing rows, label encode, no feature engineering (only 183 rows remain).  
**Scenario B – Full:** class-stratified age imputation, cabin deck extraction, title & family features, one-hot encoding, log Fare (all 891 rows kept).  

**方案A（最小处理）**：直接删除缺失，标签编码，无特征工程（只剩183行）。  
**方案B（完整处理）**：按舱位分组填补Age、提取Cabin楼层、Title与家庭特征、独热编码、Fare对数（保留全部891行）。

### 4. Modeling | 建模
**EN:** Logistic Regression + Decision Tree (simple, interpretable).  
**中文:** 逻辑回归 + 决策树（简单、可解释）。

### 5. Evaluation | 评估
**EN:** Accuracy, Precision, Recall, F1, ROC-AUC; fairness by gender and class.  
**中文:** 准确率、精确率、召回率、F1、ROC-AUC；并按性别与舱位评估公平性。

---

## Key Findings | 关键发现

### 1. Bias is structural, not random | 偏差是结构性的
**EN:** “Women and children first” makes **Sex** the strongest predictor. Class location drives survival inequality.  
**中文:** “妇孺优先”使 **性别** 成为最强预测因子；舱位位置导致生存率不平等。

### 2. MNAR missingness is critical | MNAR缺失至关重要
**EN:** Cabin missing is 77.1% overall but **97.6% in 3rd class**, showing records were not random.  
**中文:** Cabin总缺失77.1%，但**三等舱缺失高达97.6%**，说明缺失并非随机。

### Bias Mitigation Summary

During preprocessing, several data-centric mitigation strategies were applied:

- MNAR处理：使用舱位分层统计填补缺失，而非直接删除。
- 年龄偏差：应用年龄分组减少极端变异和偏态。
- 类别不平衡：保留全部数据避免因激进删除导致的幸存者偏差。

这些步骤旨在产生更代表性和稳定性的模型行为，跨不同子群体。

### 3. Minimal processing creates selection bias | 最小处理导致选择偏差
**EN:** Dropping missing rows keeps only 183 passengers (20.5%), mostly wealthier classes → biased model.  
**中文:** 删除缺失仅保留183人（20.5%），主要是一等舱 → 模型偏向富裕阶层。

### 4. Full prep improves representativeness | 完整处理提高代表性
**EN:** Keeping all 891 passengers preserves class and gender diversity, improving fairness comparisons.  
**中文:** 保留全部891人维持阶层与性别多样性，更利于公平性分析。

---

## Expected Model Insights | 预期模型洞察

**EN:**
- `Sex` and `Pclass` dominate predictive power.
- Feature engineering (Title, FamilySize) improves interpretability.
- Full prep should improve accuracy for 3rd class most.

**中文:**
- `Sex` 与 `Pclass` 是最强预测变量。
- 特征工程（Title、FamilySize）增强可解释性。
- 完整处理对三等舱提升最大。

---

## Model Results | 模型结果

**Run date | 运行日期:** February 7, 2026  
**Source | 来源:** `outputs/model_metrics.csv`

| Model | Accuracy | F1 | ROC-AUC |
|------|----------|----|---------|
| Minimal + Logistic | 0.757 | 0.830 | 0.860 |
| Full + Logistic | **0.832** | 0.776 | **0.867** |
| Minimal + Tree | 0.730 | 0.808 | 0.817 |
| Full + Tree | **0.816** | 0.740 | **0.840** |

**EN:** Full preparation improves overall accuracy for both models and provides more reliable class comparisons.  
**中文:** 完整处理在两个模型上都提高总体准确率，并使分舱位比较更可靠。

---

## Model Feature Impact | 模型特征影响结论

**Logistic Regression (Full Prep) | 逻辑回归（完整处理）**

**Top Positive Drivers | 正向影响最大的特征（提高生存）**
1. Title_Master (儿童/男童身份)
2. Sex_female (女性)
3. Deck_E / Deck_D（高层甲板）
4. Title_Mrs（已婚女性）
5. Fare_Log（更高票价）
6. AgeGroup_Child（儿童）
7. Has_Cabin（有舱位记录）

**Top Negative Drivers | 负向影响最大的特征（降低生存）**
1. Title_Mr（成年男性）
2. Sex_male（男性）
3. Pclass（舱位等级越高数值越大 → 生存率越低）
4. Deck_G / Deck_T（低层/少见甲板）
5. SibSp（同伴/配偶人数更多）

**Decision Tree Importance | 决策树重要性（Top）**
1. Title_Mr
2. Pclass
3. FamilySize
4. Has_Cabin
5. Age

**EN Summary:** Survival is strongly driven by **gender, social title, class, and cabin indicators**.  
**中文总结:** 生存率主要由 **性别、社会称谓、舱位等级与舱位记录** 决定。

---

## Visual Evidence | 可视化证据

### 1. Survival Distribution | 生存分布
![Survival Distribution](figures/survival_distribution.png)

**EN:** Class imbalance visible (more deaths than survivors).  
**中文:** 显示类别不平衡（死亡人数多于生存人数）。

### 2. Survival by Sex | 按性别生存
![Survival by Sex](figures/survival_by_sex.png)

**EN:** Female survival rate is much higher than male.  
**中文:** 女性生存率明显高于男性。

### 3. Survival by Class | 按舱位生存
![Survival by Class](figures/survival_by_class.png)

**EN:** First class survival is highest; third class lowest.  
**中文:** 一等舱生存率最高，三等舱最低。

### 4. Age Distribution | 年龄分布
![Age Distribution](figures/age_distribution.png)

**EN:** Age is right-skewed with many young adults.  
**中文:** 年龄分布右偏，年轻成人占多数。

### 5. Fare Distribution | 票价分布
![Fare Distribution](figures/fare_distribution.png)

**EN:** Fare is highly right-skewed with extreme outliers.  
**中文:** 票价明显右偏，存在极端高值。

### 6. Correlation Heatmap | 相关性热力图
![Correlation Heatmap](figures/correlation_heatmap.png)

**EN:** Pclass and Fare correlate with survival; Age shows weaker linear correlation.  
**中文:** 舱位与票价与生存率相关，年龄线性相关性较弱。

---

## Governance Lessons | 数据治理经验

1. **Profile before modeling** | **建模前必须画像**
2. **Missing data can be MNAR** | **缺失值可能非随机**
3. **Preparation affects fairness** | **准备决定公平性**
4. **Simple models reveal evidence** | **简单模型足够揭示证据**

---

## Conclusion | 结论

**EN:** Titanic survival prediction is less about model complexity and more about data governance. Full preparation preserves evidence, reduces bias, and improves accuracy, leading to more reliable and fair conclusions.

**中文:** 泰坦尼克号生存预测的关键不在复杂模型，而在数据治理。完整准备保留证据、减少偏差并提升准确率，使结论更可靠、更公平。


