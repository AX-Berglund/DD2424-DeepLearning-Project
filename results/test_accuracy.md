## Test Accuracy

### Supervised 
#### 100% labeled:

2025-05-20 20:17:44 [INFO] [DETAILED]   accuracy: 0.9765
2025-05-20 20:17:44 [INFO] [DETAILED]   precision: 0.9824
2025-05-20 20:17:44 [INFO] [DETAILED]   recall: 0.9755
2025-05-20 20:17:44 [INFO] [DETAILED]   f1: 0.9789
2025-05-20 20:17:44 [INFO] [DETAILED]   roc_auc: 0.9978
2025-05-20 20:17:44 [INFO] [DETAILED]   confusion_matrix:
[[219   5]
 [  7 279]]
2025-05-20 20:17:44 [INFO] [DETAILED]   y_pred: [large array omitted]
2025-05-20 20:17:44 [INFO] [DETAILED]   y_true: [large array omitted]
2025-05-20 20:17:44 [INFO] [DETAILED]   per_class_accuracy:
2025-05-20 20:17:44 [INFO] [DETAILED]     cat: 0.9777
2025-05-20 20:17:44 [INFO] [DETAILED]     dog: 0.9755
2025-05-20 20:17:44 [INFO] [DETAILED]   classification_report:
2025-05-20 20:17:44 [INFO] [DETAILED]     cat: {'precision': 0.9690265486725663, 'recall': 0.9776785714285714, 'f1-score': 0.9733333333333334, 'support': 224.0}
2025-05-20 20:17:44 [INFO] [DETAILED]     dog: {'precision': 0.9823943661971831, 'recall': 0.9755244755244755, 'f1-score': 0.9789473684210527, 'support': 286.0}
2025-05-20 20:17:44 [INFO] [DETAILED]     accuracy: 0.9765
2025-05-20 20:17:44 [INFO] [DETAILED]     macro avg: {'precision': 0.9757104574348747, 'recall': 0.9766015234765235, 'f1-score': 0.9761403508771931, 'support': 510.0}
2025-05-20 20:17:44 [INFO] [DETAILED]     weighted avg: {'precision': 0.9765230110491162, 'recall': 0.9764705882352941, 'f1-score': 0.976481596147231, 'support': 510.0}
2025-05-20 20:17:44 [WARNING] Failed to create visualizations: name 'datetime' is not defined
2025-05-20 20:17:44 [INFO] Test metrics:
2025-05-20 20:17:44 [INFO] - accuracy: 0.9765
2025-05-20 20:17:44 [INFO] - precision: 0.9824
2025-05-20 20:17:44 [INFO] - recall: 0.9755
2025-05-20 20:17:44 [INFO] - f1: 0.9789
2025-05-20 20:17:44 [INFO] - roc_auc: 0.9978
2025-05-20 20:17:44 [INFO] - confusion_matrix:
[[219   5]
 [  7 279]]
2025-05-20 20:17:44 [INFO] - class_names:
['cat', 'dog']
2025-05-20 20:17:44 [INFO] - per_class_accuracy: {'cat': np.float64(0.9776785714285714), 'dog': np.float64(0.9755244755244755)}
2025-05-20 20:17:44 [INFO] - classification_report: {'cat': {'precision': 0.9690265486725663, 'recall': 0.9776785714285714, 'f1-score': 0.9733333333333334, 'support': 224.0}, 'dog': {'precision': 0.9823943661971831, 'recall': 0.9755244755244755, 'f1-score': 0.9789473684210527, 'support': 286.0}, 'accuracy': 0.9764705882352941, 'macro avg': {'precision': 0.9757104574348747, 'recall': 0.9766015234765235, 'f1-score': 0.9761403508771931, 'support': 510.0}, 'weighted avg': {'precision': 0.9765230110491162, 'recall': 0.9764705882352941, 'f1-score': 0.976481596147231, 'support': 510.0}}



#### 50% labeled
2025-05-20 19:50:25 [INFO] [DETAILED]   accuracy: 0.9889
2025-05-20 19:50:25 [INFO] [DETAILED]   precision: 0.9853
2025-05-20 19:50:25 [INFO] [DETAILED]   recall: 0.9926
2025-05-20 19:50:25 [INFO] [DETAILED]   f1: 0.9889
2025-05-20 19:50:25 [INFO] [DETAILED]   roc_auc: 0.9982
2025-05-20 19:50:25 [INFO] [DETAILED]   confusion_matrix:
[[133   2]
 [  1 134]]
2025-05-20 19:50:25 [INFO] [DETAILED]   y_pred: [large array omitted]
2025-05-20 19:50:25 [INFO] [DETAILED]   y_true: [large array omitted]
2025-05-20 19:50:25 [INFO] [DETAILED]   per_class_accuracy:
2025-05-20 19:50:25 [INFO] [DETAILED]     cat: 0.9852
2025-05-20 19:50:25 [INFO] [DETAILED]     dog: 0.9926
2025-05-20 19:50:25 [INFO] [DETAILED]   classification_report:
2025-05-20 19:50:25 [INFO] [DETAILED]     cat: {'precision': 0.9925373134328358, 'recall': 0.9851851851851852, 'f1-score': 0.9888475836431226, 'support': 135.0}
2025-05-20 19:50:25 [INFO] [DETAILED]     dog: {'precision': 0.9852941176470589, 'recall': 0.9925925925925926, 'f1-score': 0.988929889298893, 'support': 135.0}
2025-05-20 19:50:25 [INFO] [DETAILED]     accuracy: 0.9889
2025-05-20 19:50:25 [INFO] [DETAILED]     macro avg: {'precision': 0.9889157155399473, 'recall': 0.9888888888888889, 'f1-score': 0.9888887364710078, 'support': 270.0}
2025-05-20 19:50:25 [INFO] [DETAILED]     weighted avg: {'precision': 0.9889157155399474, 'recall': 0.9888888888888889, 'f1-score': 0.988888736471008, 'support': 270.0}
2025-05-20 19:50:25 [WARNING] Failed to create visualizations: name 'datetime' is not defined
2025-05-20 19:50:25 [INFO] Test metrics:
2025-05-20 19:50:25 [INFO] - accuracy: 0.9889
2025-05-20 19:50:25 [INFO] - precision: 0.9853
2025-05-20 19:50:25 [INFO] - recall: 0.9926
2025-05-20 19:50:25 [INFO] - f1: 0.9889
2025-05-20 19:50:25 [INFO] - roc_auc: 0.9982
2025-05-20 19:50:25 [INFO] - confusion_matrix:
[[133   2]
 [  1 134]]
2025-05-20 19:50:25 [INFO] - class_names:
['cat', 'dog']
2025-05-20 19:50:25 [INFO] - per_class_accuracy: {'cat': np.float64(0.9851851851851852), 'dog': np.float64(0.9925925925925926)}
2025-05-20 19:50:25 [INFO] - classification_report: {'cat': {'precision': 0.9925373134328358, 'recall': 0.9851851851851852, 'f1-score': 0.9888475836431226, 'support': 135.0}, 'dog': {'precision': 0.9852941176470589, 'recall': 0.9925925925925926, 'f1-score': 0.988929889298893, 'support': 135.0}, 'accuracy': 0.9888888888888889, 'macro avg': {'precision': 0.9889157155399473, 'recall': 0.9888888888888889, 'f1-score': 0.9888887364710078, 'support': 270.0}, 'weighted avg': {'precision': 0.9889157155399474, 'recall': 0.9888888888888889, 'f1-score': 0.988888736471008, 'support': 270.0}}

#### 10% labeled
2025-05-20 18:44:03 [INFO] [DETAILED]   accuracy: 0.8727
2025-05-20 18:44:03 [INFO] [DETAILED]   precision: 0.8966
2025-05-20 18:44:03 [INFO] [DETAILED]   recall: 0.8667
2025-05-20 18:44:03 [INFO] [DETAILED]   f1: 0.8814
2025-05-20 18:44:03 [INFO] [DETAILED]   roc_auc: 0.9680
2025-05-20 18:44:03 [INFO] [DETAILED]   confusion_matrix:
[[22  3]
 [ 4 26]]
2025-05-20 18:44:03 [INFO] [DETAILED]   y_pred:
[0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 1 1 1 0 1 1 1 1 1 1 1
 1 1 0 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1]
2025-05-20 18:44:03 [INFO] [DETAILED]   y_true:
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
2025-05-20 18:44:03 [INFO] [DETAILED]   per_class_accuracy:
2025-05-20 18:44:03 [INFO] [DETAILED]     cat: 0.8800
2025-05-20 18:44:03 [INFO] [DETAILED]     dog: 0.8667
2025-05-20 18:44:03 [INFO] [DETAILED]   classification_report:
2025-05-20 18:44:03 [INFO] [DETAILED]     cat: {'precision': 0.8461538461538461, 'recall': 0.88, 'f1-score': 0.8627450980392157, 'support': 25.0}
2025-05-20 18:44:03 [INFO] [DETAILED]     dog: {'precision': 0.896551724137931, 'recall': 0.8666666666666667, 'f1-score': 0.8813559322033898, 'support': 30.0}
2025-05-20 18:44:03 [INFO] [DETAILED]     accuracy: 0.8727
2025-05-20 18:44:03 [INFO] [DETAILED]     macro avg: {'precision': 0.8713527851458887, 'recall': 0.8733333333333333, 'f1-score': 0.8720505151213027, 'support': 55.0}
2025-05-20 18:44:03 [INFO] [DETAILED]     weighted avg: {'precision': 0.8736435977815288, 'recall': 0.8727272727272727, 'f1-score': 0.8728964621287653, 'support': 55.0}
2025-05-20 18:44:03 [WARNING] Failed to create visualizations: name 'datetime' is not defined
2025-05-20 18:44:03 [INFO] Test metrics:
2025-05-20 18:44:03 [INFO] - accuracy: 0.8727
2025-05-20 18:44:03 [INFO] - precision: 0.8966
2025-05-20 18:44:03 [INFO] - recall: 0.8667
2025-05-20 18:44:03 [INFO] - f1: 0.8814
2025-05-20 18:44:03 [INFO] - roc_auc: 0.9680
2025-05-20 18:44:03 [INFO] - confusion_matrix:
[[22  3]
 [ 4 26]]
2025-05-20 18:44:03 [INFO] - class_names:
['cat', 'dog']
2025-05-20 18:44:03 [INFO] - per_class_accuracy: {'cat': np.float64(0.88), 'dog': np.float64(0.8666666666666667)}
2025-05-20 18:44:03 [INFO] - classification_report: {'cat': {'precision': 0.8461538461538461, 'recall': 0.88, 'f1-score': 0.8627450980392157, 'support': 25.0}, 'dog': {'precision': 0.896551724137931, 'recall': 0.8666666666666667, 'f1-score': 0.8813559322033898, 'support': 30.0}, 'accuracy': 0.8727272727272727, 'macro avg': {'precision': 0.8713527851458887, 'recall': 0.8733333333333333, 'f1-score': 0.8720505151213027, 'support': 55.0}, 'weighted avg': {'precision': 0.8736435977815288, 'recall': 0.8727272727272727, 'f1-score': 0.8728964621287653, 'support': 55.0}}

#### 1% labeled
2025-05-20 18:32:10 [INFO] [DETAILED]   accuracy: 1.0000
2025-05-20 18:32:10 [INFO] [DETAILED]   precision: 1.0000
2025-05-20 18:32:10 [INFO] [DETAILED]   recall: 1.0000
2025-05-20 18:32:10 [INFO] [DETAILED]   f1: 1.0000
2025-05-20 18:32:10 [INFO] [DETAILED]   roc_auc: 1.0000
2025-05-20 18:32:10 [INFO] [DETAILED]   confusion_matrix:
[[4 0]
 [0 4]]
2025-05-20 18:32:10 [INFO] [DETAILED]   y_pred:
[0 0 0 0 1 1 1 1]
2025-05-20 18:32:10 [INFO] [DETAILED]   y_true:
[0 0 0 0 1 1 1 1]
2025-05-20 18:32:10 [INFO] [DETAILED]   per_class_accuracy:
2025-05-20 18:32:10 [INFO] [DETAILED]     cat: 1.0000
2025-05-20 18:32:10 [INFO] [DETAILED]     dog: 1.0000
2025-05-20 18:32:10 [INFO] [DETAILED]   classification_report:
2025-05-20 18:32:10 [INFO] [DETAILED]     cat: {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 4.0}
2025-05-20 18:32:10 [INFO] [DETAILED]     dog: {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 4.0}
2025-05-20 18:32:10 [INFO] [DETAILED]     accuracy: 1.0000
2025-05-20 18:32:10 [INFO] [DETAILED]     macro avg: {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 8.0}
2025-05-20 18:32:10 [INFO] [DETAILED]     weighted avg: {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 8.0}
2025-05-20 18:32:10 [WARNING] Failed to create visualizations: name 'datetime' is not defined
2025-05-20 18:32:10 [INFO] Test metrics:
2025-05-20 18:32:10 [INFO] - accuracy: 1.0000
2025-05-20 18:32:10 [INFO] - precision: 1.0000
2025-05-20 18:32:10 [INFO] - recall: 1.0000
2025-05-20 18:32:10 [INFO] - f1: 1.0000
2025-05-20 18:32:10 [INFO] - roc_auc: 1.0000
2025-05-20 18:32:10 [INFO] - confusion_matrix:
[[4 0]
 [0 4]]
2025-05-20 18:32:10 [INFO] - y_pred:
[0 0 0 0 1 1 1 1]
2025-05-20 18:32:10 [INFO] - y_true:
[0 0 0 0 1 1 1 1]
2025-05-20 18:32:10 [INFO] - class_names:
['cat', 'dog']
2025-05-20 18:32:10 [INFO] - per_class_accuracy: {'cat': np.float64(1.0), 'dog': np.float64(1.0)}
2025-05-20 18:32:10 [INFO] - classification_report: {'cat': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 4.0}, 'dog': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 4.0}, 'accuracy': 1.0, 'macro avg': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 8.0}, 'weighted avg': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 8.0}}

### Semi-Supervised
#### 50% labeled
2025-05-20 19:29:21 [INFO] [DETAILED]   accuracy: 0.9778
2025-05-20 19:29:21 [INFO] [DETAILED]   precision: 0.9778
2025-05-20 19:29:21 [INFO] [DETAILED]   recall: 0.9778
2025-05-20 19:29:21 [INFO] [DETAILED]   f1: 0.9778
2025-05-20 19:29:21 [INFO] [DETAILED]   roc_auc: 0.9978
2025-05-20 19:29:21 [INFO] [DETAILED]   confusion_matrix:
[[132   3]
 [  3 132]]
2025-05-20 19:29:21 [INFO] [DETAILED]   y_pred: [large array omitted]
2025-05-20 19:29:21 [INFO] [DETAILED]   y_true: [large array omitted]
2025-05-20 19:29:21 [INFO] [DETAILED]   per_class_accuracy:
2025-05-20 19:29:21 [INFO] [DETAILED]     cat: 0.9778
2025-05-20 19:29:21 [INFO] [DETAILED]     dog: 0.9778
2025-05-20 19:29:21 [INFO] [DETAILED]   classification_report:
2025-05-20 19:29:21 [INFO] [DETAILED]     cat: {'precision': 0.9777777777777777, 'recall': 0.9777777777777777, 'f1-score': 0.9777777777777777, 'support': 135.0}
2025-05-20 19:29:21 [INFO] [DETAILED]     dog: {'precision': 0.9777777777777777, 'recall': 0.9777777777777777, 'f1-score': 0.9777777777777777, 'support': 135.0}
2025-05-20 19:29:21 [INFO] [DETAILED]     accuracy: 0.9778
2025-05-20 19:29:21 [INFO] [DETAILED]     macro avg: {'precision': 0.9777777777777777, 'recall': 0.9777777777777777, 'f1-score': 0.9777777777777777, 'support': 270.0}
2025-05-20 19:29:21 [INFO] [DETAILED]     weighted avg: {'precision': 0.9777777777777777, 'recall': 0.9777777777777777, 'f1-score': 0.9777777777777777, 'support': 270.0}
2025-05-20 19:29:21 [WARNING] Failed to create visualizations: name 'datetime' is not defined
2025-05-20 19:29:21 [INFO] Test metrics:
2025-05-20 19:29:21 [INFO] - accuracy: 0.9778
2025-05-20 19:29:21 [INFO] - precision: 0.9778
2025-05-20 19:29:21 [INFO] - recall: 0.9778
2025-05-20 19:29:21 [INFO] - f1: 0.9778
2025-05-20 19:29:21 [INFO] - roc_auc: 0.9978
2025-05-20 19:29:21 [INFO] - confusion_matrix:
[[132   3]
 [  3 132]]
2025-05-20 19:29:21 [INFO] - class_names:
['cat', 'dog']
2025-05-20 19:29:21 [INFO] - per_class_accuracy: {'cat': np.float64(0.9777777777777777), 'dog': np.float64(0.9777777777777777)}
2025-05-20 19:29:21 [INFO] - classification_report: {'cat': {'precision': 0.9777777777777777, 'recall': 0.9777777777777777, 'f1-score': 0.9777777777777777, 'support': 135.0}, 'dog': {'precision': 0.9777777777777777, 'recall': 0.9777777777777777, 'f1-score': 0.9777777777777777, 'support': 135.0}, 'accuracy': 0.9777777777777777, 'macro avg': {'precision': 0.9777777777777777, 'recall': 0.9777777777777777, 'f1-score': 0.9777777777777777, 'support': 270.0}, 'weighted avg': {'precision': 0.9777777777777777, 'recall': 0.9777777777777777, 'f1-score': 0.9777777777777777, 'support': 270.0}}

#### 10% labeled
2025-05-20 19:03:44 [INFO] [DETAILED]   accuracy: 0.9455
2025-05-20 19:03:44 [INFO] [DETAILED]   precision: 0.9355
2025-05-20 19:03:44 [INFO] [DETAILED]   recall: 0.9667
2025-05-20 19:03:44 [INFO] [DETAILED]   f1: 0.9508
2025-05-20 19:03:44 [INFO] [DETAILED]   roc_auc: 0.9840
2025-05-20 19:03:44 [INFO] [DETAILED]   confusion_matrix:
[[23  2]
 [ 1 29]]
2025-05-20 19:03:44 [INFO] [DETAILED]   y_pred:
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
2025-05-20 19:03:44 [INFO] [DETAILED]   y_true:
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
2025-05-20 19:03:44 [INFO] [DETAILED]   per_class_accuracy:
2025-05-20 19:03:44 [INFO] [DETAILED]     cat: 0.9200
2025-05-20 19:03:44 [INFO] [DETAILED]     dog: 0.9667
2025-05-20 19:03:44 [INFO] [DETAILED]   classification_report:
2025-05-20 19:03:44 [INFO] [DETAILED]     cat: {'precision': 0.9583333333333334, 'recall': 0.92, 'f1-score': 0.9387755102040817, 'support': 25.0}
2025-05-20 19:03:44 [INFO] [DETAILED]     dog: {'precision': 0.9354838709677419, 'recall': 0.9666666666666667, 'f1-score': 0.9508196721311475, 'support': 30.0}
2025-05-20 19:03:44 [INFO] [DETAILED]     accuracy: 0.9455
2025-05-20 19:03:44 [INFO] [DETAILED]     macro avg: {'precision': 0.9469086021505376, 'recall': 0.9433333333333334, 'f1-score': 0.9447975911676145, 'support': 55.0}
2025-05-20 19:03:44 [INFO] [DETAILED]     weighted avg: {'precision': 0.945869990224829, 'recall': 0.9454545454545454, 'f1-score': 0.9453450530733903, 'support': 55.0}
2025-05-20 19:03:44 [WARNING] Failed to create visualizations: name 'datetime' is not defined
2025-05-20 19:03:44 [INFO] Test metrics:
2025-05-20 19:03:44 [INFO] - accuracy: 0.9455
2025-05-20 19:03:44 [INFO] - precision: 0.9355
2025-05-20 19:03:44 [INFO] - recall: 0.9667
2025-05-20 19:03:44 [INFO] - f1: 0.9508
2025-05-20 19:03:44 [INFO] - roc_auc: 0.9840
2025-05-20 19:03:44 [INFO] - confusion_matrix:
[[23  2]
 [ 1 29]]
2025-05-20 19:03:44 [INFO] - class_names:
['cat', 'dog']
2025-05-20 19:03:44 [INFO] - per_class_accuracy: {'cat': np.float64(0.92), 'dog': np.float64(0.9666666666666667)}
2025-05-20 19:03:44 [INFO] - classification_report: {'cat': {'precision': 0.9583333333333334, 'recall': 0.92, 'f1-score': 0.9387755102040817, 'support': 25.0}, 'dog': {'precision': 0.9354838709677419, 'recall': 0.9666666666666667, 'f1-score': 0.9508196721311475, 'support': 30.0}, 'accuracy': 0.9454545454545454, 'macro avg': {'precision': 0.9469086021505376, 'recall': 0.9433333333333334, 'f1-score': 0.9447975911676145, 'support': 55.0}, 'weighted avg': {'precision': 0.945869990224829, 'recall': 0.9454545454545454, 'f1-score': 0.9453450530733903, 'support': 55.0}}

#### 1% labeled
2025-05-20 18:22:35 [INFO] [DETAILED]   accuracy: 1.0000
2025-05-20 18:22:35 [INFO] [DETAILED]   precision: 1.0000
2025-05-20 18:22:35 [INFO] [DETAILED]   recall: 1.0000
2025-05-20 18:22:35 [INFO] [DETAILED]   f1: 1.0000
2025-05-20 18:22:35 [INFO] [DETAILED]   roc_auc: 1.0000
2025-05-20 18:22:35 [INFO] [DETAILED]   confusion_matrix:
[[4 0]
 [0 4]]
2025-05-20 18:22:35 [INFO] [DETAILED]   y_pred:
[0 0 0 0 1 1 1 1]
2025-05-20 18:22:35 [INFO] [DETAILED]   y_true:
[0 0 0 0 1 1 1 1]
2025-05-20 18:22:35 [INFO] [DETAILED]   per_class_accuracy:
2025-05-20 18:22:35 [INFO] [DETAILED]     cat: 1.0000
2025-05-20 18:22:35 [INFO] [DETAILED]     dog: 1.0000
2025-05-20 18:22:35 [INFO] [DETAILED]   classification_report:
2025-05-20 18:22:35 [INFO] [DETAILED]     cat: {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 4.0}
2025-05-20 18:22:35 [INFO] [DETAILED]     dog: {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 4.0}
2025-05-20 18:22:35 [INFO] [DETAILED]     accuracy: 1.0000
2025-05-20 18:22:35 [INFO] [DETAILED]     macro avg: {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 8.0}
2025-05-20 18:22:35 [INFO] [DETAILED]     weighted avg: {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 8.0}
2025-05-20 18:22:35 [WARNING] Failed to create visualizations: name 'datetime' is not defined
2025-05-20 18:22:35 [INFO] Test metrics:
2025-05-20 18:22:35 [INFO] - accuracy: 1.0000
2025-05-20 18:22:35 [INFO] - precision: 1.0000
2025-05-20 18:22:35 [INFO] - recall: 1.0000
2025-05-20 18:22:35 [INFO] - f1: 1.0000
2025-05-20 18:22:35 [INFO] - roc_auc: 1.0000
2025-05-20 18:22:35 [INFO] - confusion_matrix:
[[4 0]
 [0 4]]
2025-05-20 18:22:35 [INFO] - class_names:
['cat', 'dog']
2025-05-20 18:22:35 [INFO] - per_class_accuracy: {'cat': np.float64(1.0), 'dog': np.float64(1.0)}
2025-05-20 18:22:35 [INFO] - classification_report: {'cat': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 4.0}, 'dog': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 4.0}, 'accuracy': 1.0, 'macro avg': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 8.0}, 'weighted avg': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 8.0}}
