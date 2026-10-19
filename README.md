# SVM on the Wisconsin Breast Cancer data

Classifies a tumour as **benign** or **malignant** from 9 cell measurements (scores 1-10) using a Support Vector Machine (scikit-learn).

Run: `pip install -r requirements.txt` then `python main.py` (about 5 seconds, results are reproducible: fixed seed).

## 1. How an SVM works

- An SVM finds the **hyperplane** that separates the two classes with the **largest margin** (the widest empty "street" between them). A large margin means good generalization.
- The training points that lie on the margin are the **support vectors**. Only they define the boundary; removing any other point does not change the model.
- Real data is rarely perfectly separable, so the **soft margin** allows some errors. The parameter **C** controls the trade-off: high C = few errors allowed, smaller margin (overfitting risk); low C = wider margin, more errors tolerated (underfitting risk).
- For non-linear problems, a **kernel** (here RBF) lets the SVM draw a curved boundary without explicitly computing a higher-dimensional space (kernel trick). **gamma** sets how far the influence of one sample reaches (high = tight boundary, low = smooth boundary).
- SVM works with distances, so features should be **scaled**. The scaler must learn its values from the training data only.

## 2. What `main.py` does

1. **Load data.** Drop the `Id` column (it says nothing about the diagnosis). `?` marks missing values. Recode the class: 2 = benign (0), 4 = malignant (1).
2. **Missing values.** Only 16 values (all in `Bare Nuclei`) are missing. They are replaced by the column **median**. This is done inside a pipeline, so the median is computed on the training data only.
3. **Scalers.** The same SVM (RBF kernel, default parameters) is tried with no scaling, `MinMaxScaler` (to [0, 1]) and `StandardScaler` (mean 0, variance 1). Each is evaluated on **30 random 80/20 train/test splits** and we report the mean and standard deviation of the accuracy.
4. **Final model.** `MinMaxScaler` + RBF SVM, trained on 80% of the data and tested on the remaining 20%. We print the accuracy and the confusion matrix.
5. **Variable importance (permutation importance).** On the test set, the values of one feature are shuffled and we measure how much the accuracy drops. A large drop = the model relies on that feature. Saved as `permutation_importance.png`.
6. **Support vectors.** The support vectors are circled on a scatter plot of the two most important features. Saved as `support_vectors.png`. (The model uses 9 features, so this 2D picture is only a projection; points are slightly jittered because the values are integers.)

## 3. Results

**Scalers (30 runs):**

| Scaler | Mean accuracy | Std |
|---|---|---|
| none | 0.9693 | 0.0100 |
| MinMaxScaler | 0.9693 | 0.0100 |
| StandardScaler | 0.9690 | 0.0105 |

The scaler makes no real difference, because all features already use the same 1-10 scale. MinMaxScaler was chosen anyway because scaling is recommended for SVMs.

**Final model on the test set (140 samples): accuracy 95.7%**

|  | predicted benign | predicted malignant |
|---|---|---|
| **true benign** | 89 | 3 |
| **true malignant** | 3 | 45 |

The most dangerous error is a malignant tumour predicted as benign (3 cases here).

**Most important features:** `Bare Nuclei` is clearly the most important (accuracy drop 0.059), followed by `Clump Thickness`, `Normal Nucleoli` and `Uniformity of Cell Shape` (about 0.015 each). The remaining features matter little, partly because many features are correlated and carry similar information.

**Support vectors:** 82 of the 559 training samples (about 15%). The other 85% could be removed without changing the model.

## 4. Notes

- Accuracy on one 140-sample test set is noisy (one error = 0.7%). The 30-run mean (about 96.9%) is the more reliable estimate.
- The default RBF parameters (`C=1`, `gamma="scale"`) already work very well, so no hyper-parameter tuning was done.
