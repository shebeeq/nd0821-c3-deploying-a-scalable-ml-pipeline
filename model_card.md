# Model Card

## Model Details
* **Developer:** shebeeq
* **Model Date:** June 2026
* **Model Type:** Random Forest Classifier built using the Scikit-Learn framework.
* **Information for Users:** This model was optimized as part of the Udacity Machine Learning DevOps Engineer curriculum, utilizing an automated feature pipeline tracking system to handle catagorical vectors via `OneHotEncoder` and target parameters via `LabelBinarizer`.
* **License:** MIT License.

## Intended Use
* **Primary Intended Uses:** This machine learning system is intended to classify individuals into discrete annual income brackets based on publicly available demographic data. Specifically, it predicts whether an individual earns more than \$50,00 USD per year (`>5K`) or less than or equal to \$50,00 USD per year (`<=5K`).
* **Primary Intended Users:** The primary target audience includes data scientists, regional policy researchers, and socioeconomic analysts studying corporate demographic wage distribution trends.
* **Out-of-Scope Use Cases:** This system should not be utilized to automate real-time automated corporate hiring metrics, financial loan underwriting decisions, or individual credit rating approvals, as doing so can propagate systemic socioeconomic bias.

## Training Data
* **Data Source:** The model is trained on the classic UCI Machine Learning Repository Census Income dataset, which is locally referenceable as `census.csv` inside the project workspace directory.
* **Pre-processing Operations:** Messy trailing and leading whitespaces are dynamically stripped across text columns. Categorical profiles are transformed using an isolated `OneHotEncoder`, while target labels are securely handled via a `LabelBinarizer`. 
* **Data Splits:** The raw matrix rows are randomly split into an 80% partition used strictly for fitting model weights and a 20% independent evaluation split to calculate out-of-sample metrics.

## Evaluation Data
* **Data Source:** The evaluation dataset consists of the isolated 20% holdout split generated from the original `census.csv` distribution.
* **Pre-processing Operations:** The evaluation vectors are processed through identical transformation encoders fitted exclusively on the training subset to prevent data leakage.

## Metrics
The classification model was rigorously validated using the following three performance metrics to assess overall robustness:
* **Precision:** Measures the ratio of correctly predicted high-income earners to the total predicted high-income earners.
* **Recall:** Measures the capability of the classifier to find all actual high-income individuals in the test partition.
* **F1-Score:** The harmonic mean of Precision and Recall, which balances both indicators into a single score.

### Overall Validation Results
On the 20% out-of-sample evaluation dataset, the trained Random Forest Classifier achieved the following robust performance figures:
* **Precision:** **0.7920** (79.20%)
* **Recall:** **0.5767** (57.67%)
* **F1-Score:** **0.6674** (66.74%)

## Ethical Considerations
* **Biased Dependencies:** The baseline dataset is deeply tied to historical socioeconomic inequalities from the late 20th century. Demographic attributes such as `sex`, `race`, and `native-country` reflect systemic biases present in institutional systems. 
* **Mitigation Strategies:** Automated decisions based on these categories should be heavily monitored. Analysts should consider removing demographic columns entirely if deploying this system in contexts where equity is required.

## Caveats and Recommendations
* **Data Age:** The underlying census distributions were sampled in 1994 within the United States. Consequently, the financial attributes (such as capital gains, losses, and the \$50K threshold) do not scale to contemporary global economic metrics or modern inflation adjustments.
* **Deployment Advice:** It is highly recommended to retrain this model on updated, modern regional census registers before applying its inferences to real-world financial or demographic forecasting software.
