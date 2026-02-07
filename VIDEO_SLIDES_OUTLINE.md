# Video Presentation Outline (12 Minutes)
# 视频演示大纲（12分钟）

---

## Team | 团队
**[Names here]**

### Script | 演讲稿
**EN:** "Our team members are [Names], and each of us will cover a specific part of the analysis. We will introduce the research question, data governance, bias analysis, modeling, and conclusions."

**中文:** "我们的团队成员是[Names]，每位成员负责不同部分，包括研究问题、数据治理、偏差分析、建模和结论。"

---

## Timing Guide | 时间分配

| Section 部分 | Speaker 演讲者 | Time 时间 |
|--------------|----------------|-----------|
| Slides 1-2: Intro 介绍 | Member 1 | 1:30 |
| Slides 3-4: Governance 数据治理 | Member 2 | 3:00 |
| Slides 5-8: Bias & EDA 偏差与EDA | Member 3 | 4:00 |
| Slides 9-12: Modeling & Conclusion 建模与结论 | Member 4 | 3:30 |

### Script | 演讲稿
**EN:** "We will follow a strict 12-minute structure: intro and question first, then data governance and missingness, followed by bias analysis, and finally modeling results and conclusions."

**中文:** "我们将严格按照12分钟结构进行：先介绍与研究问题，再讲数据治理与缺失处理，随后偏差分析，最后模型结果与结论。"

---

# SLIDES | 幻灯片

---

## SLIDE 1: Title (0:00-0:30)
## 幻灯片1：标题（0:00-0:30）

**Speaker | 演讲者:** Member 1

### Title | 标题
**EN:** Titanic Survival Analysis: Data Governance and Bias
**中文:** 泰坦尼克号生存分析：数据治理与偏差

### Content | 内容
- Team members | 团队成员
- Course | 课程: CA6003
- Institution | 机构

### Script | 演讲稿
**EN (Verbatim, ~25–30s):**  
"Hello everyone, welcome to our presentation. We are Group [X], and our topic is Titanic Survival Analysis: Data Governance and Bias.  
Our goal is not just to predict survival, but to show how governance choices—like handling missing data and designing features—change the conclusions.  
We will keep the models simple and focus on clear, evidence-based interpretation."

**中文（逐字稿，约25–30秒）：**  
"大家好，欢迎观看我们的演示。我们是第[X]组，主题是泰坦尼克号生存分析：数据治理与偏差。  
我们的目标不只是预测生存，而是展示数据治理选择——比如如何处理缺失值、如何构造特征——会如何改变结论。  
我们会使用简单模型，重点放在清晰、基于证据的解释上。"

---

## SLIDE 2: Research Question (0:30-1:30)
## 幻灯片2：研究问题（0:30-1:30）

**Speaker | 演讲者:** Member 1

### Title | 标题
**EN:** Research Question & Dataset
**中文:** 研究问题与数据集

### Key Points | 要点
- Question: How do prep choices affect reliability and fairness?
- Dataset: Titanic passengers (891 rows, 12 features)
- Target: Survived (0/1)

### Script | 演讲稿
**EN (Verbatim, ~55–60s):**  
"Our research question is: how do data preparation choices affect the reliability and fairness of survival prediction?  
The Titanic dataset is ideal because it contains clear historical bias—women and children first—and also severe missing data, especially for cabin information.  
This means governance decisions are not hidden. They directly change who remains in the dataset and which patterns the model learns.  
So we compare a minimal preparation pipeline with a full, governance-driven pipeline, and then interpret the feature impacts."

**中文（逐字稿，约55–60秒）：**  
"我们的研究问题是：数据准备策略如何影响生存预测的可靠性与公平性？  
泰坦尼克数据集非常适合这个问题，因为它既有明显的历史偏差——“妇孺优先”，又存在严重缺失，尤其是Cabin信息。  
这意味着数据治理决策不会被隐藏，而是会直接改变数据集中保留了哪些人，模型学习到哪些规律。  
因此我们对比最小处理与完整处理，并解释特征影响。"

---

## SLIDE 3: Data Profiling (1:30-2:30)
## 幻灯片3：数据画像（1:30-2:30）

**Speaker | 演讲者:** Member 2

### Title | 标题
**EN:** Data Profiling Results
**中文:** 数据画像结果

### Key Points | 要点
- Missing: Age 19.9%, Cabin 77.1%, Embarked 0.2%
- Class imbalance: 61.6% died vs 38.4% survived
- Fare is highly skewed

### Script | 演讲稿
**EN (Verbatim, ~55–60s):**  
"We start with profiling. The dataset has 891 passengers and 12 features.  
Missingness is severe: Age is missing about 19.9%, Cabin about 77.1%, and Embarked about 0.2%.  
The target is imbalanced: 61.6% died and 38.4% survived.  
Fare is highly right-skewed, which signals outliers and socioeconomic effects.  
These profiling results already show that data governance is not optional—it determines whether analysis is valid."

**中文（逐字稿，约55–60秒）：**  
"我们从数据画像开始。数据集共有891名乘客、12个特征。  
缺失非常严重：Age缺失约19.9%，Cabin缺失约77.1%，Embarked缺失约0.2%。  
目标变量不平衡：死亡61.6%，生存38.4%。  
票价高度右偏，说明存在异常值及阶层效应。  
这些画像结果说明数据治理不是可选项，而是分析有效性的前提。"

---

## SLIDE 4: MNAR Missingness (2:30-4:00)
## 幻灯片4：MNAR缺失（2:30-4:00）

**Speaker | 演讲者:** Member 2

### Title | 标题
**EN:** Missing Not At Random (MNAR)
**中文:** 非随机缺失（MNAR）

### Evidence | 证据
- Cabin missing overall 77.1%
- 1st class missing 18.5% vs 3rd class missing 97.6%
- Dropping missing rows leaves only 183 passengers

### Script | 演讲稿
**EN (Verbatim, ~85–90s):**  
"Missingness here is not random. Cabin is missing for 77.1% overall, but the pattern is highly unequal by class:  
only 18.5% missing in 1st class, but 97.6% missing in 3rd class.  
This is classic MNAR—Missing Not At Random.  
If we simply delete missing rows, the dataset collapses from 891 to 183 passengers.  
That means we remove almost all 3rd-class passengers and create a selection bias toward wealthy groups.  
To avoid this, we keep all passengers, impute Age by class, and create indicators like Has_Cabin and Deck.  
This preserves evidence and reduces bias before modeling."

**中文（逐字稿，约85–90秒）：**  
"这里的缺失不是随机的。Cabin总体缺失77.1%，但按舱位差异巨大：  
一等舱只缺失18.5%，三等舱却高达97.6%。  
这是典型的MNAR——非随机缺失。  
如果直接删除缺失行，样本从891降到183人。  
这几乎删除了所有三等舱乘客，产生严重的选择偏差。  
因此我们保留所有乘客，按舱位分组填补Age，并构造Has_Cabin和Deck等特征，  
以保留证据并在建模前降低偏差。"

---

## SLIDE 5: Bias #1 Gender (4:00-4:45)
## 幻灯片5：偏差1 性别（4:00-4:45）

**Speaker | 演讲者:** Member 3

### Title | 标题
**EN:** Gender Bias
**中文:** 性别偏差

### Key Stats | 关键数据
- Female survival: 74.2%
- Male survival: 18.9%

### Script | 演讲稿
**EN (Verbatim, ~40–45s):**  
"The first bias is gender. The ‘women and children first’ policy created a massive gap.  
Female survival is about 74.2%, while male survival is only about 18.9%.  
This makes Sex the strongest predictor.  
We do not remove this bias, because it is historical reality, but we must acknowledge it clearly in interpretation."

**中文（逐字稿，约40–45秒）：**  
"第一个偏差是性别。“妇孺优先”导致巨大差距：  
女性生存率约74.2%，男性只有约18.9%。  
因此性别是最强预测因子。  
我们不会去“消除”这个偏差，因为它是历史现实，但必须在解释中清楚说明。"

---

## SLIDE 6: Bias #2 Class (4:45-5:30)
## 幻灯片6：偏差2 阶层（4:45-5:30）

**Speaker | 演讲者:** Member 3

### Title | 标题
**EN:** Socioeconomic Bias (Pclass)
**中文:** 阶层偏差（舱位等级）

### Key Stats | 关键数据
- 1st class survival: 63.0%
- 3rd class survival: 24.2%

### Script | 演讲稿
**EN (Verbatim, ~40–45s):**  
"The second bias is socioeconomic class.  
First-class survival is about 63%, while third-class survival is about 24.2%.  
Class location and access to lifeboats created structural disadvantage for lower classes.  
This means Pclass is a strong predictor and a key fairness dimension."

**中文（逐字稿，约40–45秒）：**  
"第二个偏差是阶层。一等舱生存率约63%，三等舱只有24.2%。  
舱位位置与救生艇获取造成对低阶层的结构性不利。  
这使舱位等级成为强预测因子，也是公平性的重要维度。"

---

## SLIDE 7: Bias #3 Age (5:30-6:15)
## 幻灯片7：偏差3 年龄（5:30-6:15）

**Speaker | 演讲者:** Member 3

### Title | 标题
**EN:** Age Bias
**中文:** 年龄偏差

### Key Stats | 关键数据
- Children (0-12) survival: 57.9%
- Seniors (60+) survival: 22.7%

### Script | 演讲稿
**EN (Verbatim, ~40–45s):**  
"The third bias is age. Children aged 0–12 had about 57.9% survival, while seniors over 60 had only about 22.7%.  
Age effects are non-linear.  
This is why we include age groups and a decision tree model, which can capture these group differences better than a purely linear model."

**中文（逐字稿，约40–45秒）：**  
"第三个偏差是年龄。0–12岁儿童生存率约57.9%，60岁以上老人只有约22.7%。  
年龄效应是非线性的。  
因此我们引入年龄分组，并使用决策树来捕捉这种分组差异，而不仅是线性趋势。"

---

## SLIDE 8: Simpson’s Paradox (6:15-7:30)
## 幻灯片8：辛普森悖论（6:15-7:30）

**Speaker | 演讲者:** Member 3

### Title | 标题
**EN:** Simpson’s Paradox (Sex × Class)
**中文:** 辛普森悖论（性别×舱位）

### Key Insight | 关键洞察
- Gender gap varies across classes
- 2nd class gap is largest, 3rd class smallest

### Script | 演讲稿
**EN (Verbatim, ~70–75s):**  
"We also observe Simpson’s paradox. The gender advantage is not constant across classes.  
For example, the gender gap is largest in 2nd class and smallest in 3rd class.  
If we only look at overall averages, we would miss these interaction effects.  
That is why we analyze gender and class together, and in modeling we consider interaction-aware interpretation."

**中文（逐字稿，约70–75秒）：**  
"我们还观察到辛普森悖论。性别优势在不同舱位并不一致。  
例如二等舱的性别差距最大，而三等舱最小。  
如果只看总体平均值，就会忽略这种交互效应。  
因此我们在分析中同时考虑性别和舱位，并在建模解释中强调交互。"

---

## SLIDE 9: Prep Comparison (7:30-8:30)
## 幻灯片9：准备方案对比（7:30-8:30）

**Speaker | 演讲者:** Member 4

### Title | 标题
**EN:** Minimal vs Full Preparation
**中文:** 最小处理 vs 完整处理

### Key Points | 要点
- Minimal: 183 rows, selection bias, label encoding
- Full: 891 rows, imputation + feature engineering, one-hot encoding
- Full prep improves fairness for 3rd class

### Script | 演讲稿
**EN (Verbatim, ~55–60s):**  
"We compare two preparation strategies.  
Minimal prep deletes all missing rows, label-encodes categories, and uses no feature engineering. This leaves only 183 passengers—about 20.5% of the data.  
Full prep keeps all 891 passengers, imputes Age by class, extracts Title, creates FamilySize and Has_Cabin, uses one-hot encoding, and log-transforms Fare.  
The key message: minimal prep is fast but biased; full prep is fairer and more reliable."

**中文（逐字稿，约55–60秒）：**  
"我们对比两种准备方案。  
最小处理直接删除缺失、标签编码、无特征工程，仅剩183人，约20.5%。  
完整处理保留891人，按舱位填补Age，提取Title，构造FamilySize与Has_Cabin，进行独热编码并对Fare取对数。  
核心结论是：最小处理快但偏差大，完整处理更公平、更可靠。"

---

## SLIDE 10: Modeling Setup (8:30-9:15)
## 幻灯片10：建模设置（8:30-9:15）

**Speaker | 演讲者:** Member 4

### Title | 标题
**EN:** Modeling Approach
**中文:** 建模方法

### Key Points | 要点
- Logistic Regression (interpretable)
- Decision Tree (non-linear)
- Stratified 80/20 split

### Script | 演讲稿
**EN (Verbatim, ~40–45s):**  
"We use two simple models: logistic regression for interpretability and a decision tree for non-linear effects.  
We apply a stratified 80/20 split to keep the class balance in train and test.  
The purpose is not the highest accuracy, but to use models as evidence for how preparation changes what the model learns."

**中文（逐字稿，约40–45秒）：**  
"我们使用两种简单模型：逻辑回归用于可解释性，决策树用于捕捉非线性。  
训练/测试采用分层80/20拆分以保持类别比例。  
目的不是追求最高准确率，而是用模型作为证据，展示准备方式如何改变模型学习到的规律。"

---

## SLIDE 11: Feature Impact (9:15-10:30)
## 幻灯片11：特征影响（9:15-10:30）

**Speaker | 演讲者:** Member 4

### Title | 标题
**EN:** Which Features Drive Survival?
**中文:** 哪些特征决定生存？

### Key Points | 要点
- Logistic Regression: positive drivers = female, Master/Mrs titles, higher decks, higher fare, child
- Negative drivers = Mr, male, higher Pclass (3rd), low decks, larger SibSp
- Decision Tree top importance = Title_Mr, Pclass, FamilySize, Has_Cabin, Age

### Script | 演讲稿
**EN (Verbatim, ~75–80s):**  
"Now we interpret the models.  
In logistic regression, the strongest positive drivers are female, child-related titles like Master, Mrs, higher decks, higher fare, and having cabin records.  
The strongest negative drivers are Mr, male, higher Pclass, and some low-deck indicators.  
The decision tree confirms that Title and Pclass are the top split features, followed by FamilySize, Has_Cabin, and Age.  
So the model conclusion is clear: gender, social status, and class location are the dominant drivers of survival."

**中文（逐字稿，约75–80秒）：**  
"接下来解释模型。  
逻辑回归中，最强正向因素包括女性、与儿童相关的称谓如Master、Mrs、更高甲板、更高票价以及有Cabin记录。  
最强负向因素包括Mr、男性、更高舱位等级以及低层甲板。  
决策树也验证Title与Pclass是最重要的分裂特征，其次是FamilySize、Has_Cabin与Age。  
因此模型结论很清晰：性别、社会身份与舱位等级是生存的主要驱动因素。"

---

## SLIDE 12: Conclusion (10:30-11:30)
## 幻灯片12：结论（10:30-11:30）

**Speaker | 演讲者:** All

### Title | 标题
**EN:** Conclusions & Lessons
**中文:** 结论与经验

### Key Messages | 要点
- Data governance is central to reliability
- Missing data handling changes fairness outcomes
- Full preparation improves accuracy and interpretability

### Script | 演讲稿
**EN (Verbatim, ~55–60s):**  
"To conclude, Titanic survival analysis shows that governance decisions matter more than model complexity.  
When we preserve data and handle missingness carefully, we obtain more reliable and fair results.  
Model interpretation reveals that gender, social title, and class are the strongest drivers of survival, which aligns with historical context.  
This project demonstrates that good data preparation is essential for trustworthy conclusions."

**中文（逐字稿，约55–60秒）：**  
"最后总结，泰坦尼克号分析表明数据治理决策比模型复杂度更重要。  
当我们保留数据并谨慎处理缺失值时，结果更可靠也更公平。  
模型解释显示性别、社会称谓和舱位等级是生存的最强驱动因素，与历史背景一致。  
本项目证明高质量的数据准备是可信结论的关键。"

---
