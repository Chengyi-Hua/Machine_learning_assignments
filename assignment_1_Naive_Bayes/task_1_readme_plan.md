
---

## Tasks & Report Agenda

### **Implementation-only (code only, not in report)**
- **Task 1:** Implement `nb_train` for Naive Bayes training with symmetric Dirichlet prior.  
- **Task 2:** Implement `nb_predict` to classify and compute log-probabilities.  
- **Task 5a:** Implement `nb_generate` to generate digits from learned distributions.  

### **Report-required**
- **Task 3 – Experiments on MNIST**
  - Train Naive Bayes with α = 2.
  - Compute test accuracy.
  - Plot correctly classified digits.
  - Plot misclassified digits + confusion matrix.
  - Discuss error patterns.
- **Task 4 – Model selection (optional)**
  - Cross-validation on α values.
  - Plot accuracy vs α.
  - Discuss optimal α.
- **Task 5b – Data generation**
  - Generate digits with α = 2.
  - Compare samples with different α values.
  - Interpret sharpness vs smoothness.
- **Task 6 – Missing data**
  - Derive formulas for:
    - \(p(y|x_{1:D})\)
    - \(p(y|x_{1:D'})\)
    - \(p(x_{D'+1:D}|x_{1:D'})\)
  - Explain when/why they are useful.
  - (Optional: sample images from missing-data setup)

---

##  Experiments Workflow
1. Train model on MNIST with α = 2.  
2. Evaluate accuracy on test set.  
3. Visualize correctly classified digits.  
4. Visualize misclassified digits and confusion matrix.  
5. (Optional) Run cross-validation for α tuning.  
6. Generate digits for each class with varying α.  

---

##  Report Structure
1. **Introduction** – Goal, dataset, Naive Bayes basics.  
2. **Task 3 – Experiments** (setup, trials, results, error analysis).  
3. **Task 4 – Model selection (optional)**.  
4. **Task 5b – Data generation** (samples, influence of α).  
5. **Task 6 – Missing data** (formulas + interpretation).  
6. **Conclusion** – Summary and lessons.  
7. **References** – If additional sources were used.  

---






---

---

##  Report Content

### 1. Introduction
- Motivation: Naive Bayes as a baseline on MNIST.  
- Goal: implement, evaluate, and analyze.  

### 2. Task 3 – Experiments on MNIST
- **What was done:** trained with α = 2, verified with toy dataset.  
- **Results:**  
  - Test accuracy  
  - Correctly classified digits  
  - Misclassified digits  
  - Confusion matrix  
- **Discussion:** error patterns (e.g., 4 vs 9).  

### 3. Task 4 – Model Selection (optional)
- Cross-validation over α values.  
- Accuracy vs α plot.  
- Effect of smoothing.  

### 4. Task 5b – Data Generation
- Generate digits for each class.  
- Compare different α values.  
- Interpretation: sharpness vs smoothness.  

### 5. Task 6 – Missing Data
- Derived formulas:  
  - \(p(y|x_{1:D})\)  
  - \(p(y|x_{1:D'})\)  
  - \(p(x_{D’+1:D}|x_{1:D’})\)  
- Discussion: usefulness in incomplete inputs, imputation.  

### 6. Conclusion
- Summary of implementation & results.  
- Lessons learned: simplicity of NB, limits on complex data.  

### 7. References
- MNIST dataset (Yann LeCun)  
- Lecture slides & additional readings  

---

