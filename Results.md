# Model Evaluation Results

This document summarizes the evaluation results of different models on the emotion classification dataset.

## Dataset Information
- Validation set size: 800 samples
- 8 emotion classes: Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger, Contempt

## ResNet34 Results

### Classification Performance

```
              precision    recall  f1-score   support

     Neutral     0.3235    0.1100    0.1642       100
       Happy     0.6195    0.7000    0.6573       100
         Sad     0.4144    0.4600    0.4360       100
    Surprise     0.5694    0.4100    0.4767       100
        Fear     0.6300    0.6300    0.6300       100
     Disgust     0.4587    0.5000    0.4785       100
       Anger     0.3923    0.5100    0.4435       100
    Contempt     0.4122    0.5400    0.4675       100

    accuracy                         0.4825       800
   macro avg     0.4775    0.4825    0.4692       800
weighted avg     0.4775    0.4825    0.4692       800
```

### Summary Metrics
- Overall Accuracy: **0.4825**
- Macro F1-score: **0.4692**
- Valence RMSE: **0.3885**, MAE: **0.3085**
- Arousal RMSE: **0.3341**, MAE: **0.2702**

## EfficientNet-B0 Results

### Classification Performance

```
              precision    recall  f1-score   support

     Neutral     0.3627    0.3700    0.3663       100
       Happy     0.6907    0.6700    0.6802       100
         Sad     0.5000    0.4300    0.4624       100
    Surprise     0.5857    0.4100    0.4824       100
        Fear     0.5333    0.5600    0.5463       100
     Disgust     0.4409    0.5600    0.4934       100
       Anger     0.4412    0.3000    0.3571       100
    Contempt     0.4069    0.5900    0.4816       100

    accuracy                         0.4863       800
   macro avg     0.4952    0.4862    0.4837       800
weighted avg     0.4952    0.4863    0.4837       800
```

### Summary Metrics
- Overall Accuracy: **0.4863**
- Macro F1-score: **0.4837**
- Valence RMSE: **0.3962**, MAE: **0.3147**
- Arousal RMSE: **0.3487**, MAE: **0.2762**

## Models Comparison

| Model | Accuracy | F1-score | Valence RMSE | Arousal RMSE |
|-------|----------|----------|--------------|--------------|
| ResNet34 | 0.4825 | 0.4692 | 0.3885 | 0.3341 |
| EfficientNet-B0 | 0.4863 | 0.4837 | 0.3962 | 0.3487 |

EfficientNet-B0 shows a slight improvement in classification accuracy and F1-score over ResNet34, while ResNet34 performs slightly better on the regression metrics (Valence and Arousal).
