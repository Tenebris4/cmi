# Process Report

## Phase 1: Random Forest  

### Reason for Choosing the Model  

Our team selected the Random Forest model based on three main factors:  

1. **Simplicity and Accessibility**  
   Random Forest is a straightforward and easy-to-understand model. Since our team members had no prior experience with Machine Learning (ML) or Deep Learning (DL), Random Forest was our first thought upon analyzing the competition dataset.  

2. **Handling Nonlinear and Ordered Classification Problems**  
   This problem involves ordered classification with features related to health, behavior, etc., which are often non-linear. Random Forest is well-suited for handling such data due to its tree-based structure and reliance on Decision Trees, which effectively manage non-linear relationships.  

3. **Robustness to Missing and Imbalanced Data**  
   The dataset contains significant missing values and an imbalanced distribution of the target label `sii`:  
   - Label `0.0 (NONE)` accounts for over 50% of the data.
   - Label `1.0 (MILD)` accounts for over 26 of the data.
   - Label `2.0 (MODERATE)` accounts for over 13% of the data. 
   - Label `3.0 (SEVERE)` comprises less than 2% of the data.  
   Random Forest’s bagging ensemble mechanism helps mitigate overfitting and performs well in scenarios with imbalanced data.  

### Experimentation and Results  

#### Version 1: Initial Approach  

1. **Preprocessing**  
   - Missing values were handled using `SimpleImputer`.  
   - Categorical data was encoded using `OrdinalEncoder`.  

2. **Model Training Strategy**  
   - Given the moderate size of the dataset (3960 samples, 82 features), we applied **K-Fold Cross Validation** to maximize dataset utilization and obtain reliable performance estimates.  

3. **Hyperparameter Optimization**  
   - Hyperparameters were tuned using a simple **Grid Search**, and evaluation was based on **accuracy**.  

**Result**:  
- <span style="color:orange">**Accuracy**: **0.216**</span>

Our team recognized that the low performance was likely due to overly simple implementation methods and suspected that imputing missing values for the `sii` target label was inappropriate.  

#### Version 2: Enhanced Approach  

1. **Improved Hyperparameter Tuning and Metrics**  
   - Tried alternative hyperparameter tuning methods:  
     - **Random Search**: Achieved <span style="color:orange">**F1_weighted = 0.229**</span>  
     - **Bayesian Optimization**: Achieved <span style="color:orange">**F1_weighted = 0.240**</span>  
   - Changed the evaluation metric to **F1_weighted** instead of accuracy. F1_score provides a better balance between precision and recall, which is crucial given the class imbalance.  

2. **Alignment with Competition Metric**  
   - Used the required **Quadratic Weighted Kappa (QWK)** metric to evaluate performance and compare results.  

3. **Addressing Class Imbalance**  
   To handle the significant class imbalance in the `sii` target label, we experimented with the following techniques:  
   - **Undersampling**: Reduced the majority class sample size.  
   - **SMOTE (Synthetic Minority Oversampling Technique)**: Oversampled the minority class.  
   - **Combination of SMOTE and Undersampling**: Balanced both minority and majority classes.  

**Results**:  
- **Undersampling**: <span style="color:orange">**Submit score = 0.290**</span>  
- **SMOTE**: <span style="color:orange">**Submit score = 0.381**</span>
- **Combination of SMOTE and Undersampling**: <span style="color:orange">**Submit score = 0.377**</span>

Using **SMOTE** significantly improved the score. The improvement can be attributed to better representation of the minority class (label `3.0`), which constituted less than 2% of the dataset. By increasing the representation of this class, the model could learn its patterns more effectively.  

### Future Directions  

We aim to refine the model further by exploring additional techniques, including:  
- Advanced imputation strategies for missing values.  
- Feature engineering to capture complex relationships in the data.  
- Implementation of ensemble methods beyond Random Forest for improved performance.  
