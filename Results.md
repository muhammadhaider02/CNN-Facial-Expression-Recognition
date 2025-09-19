# Model Evaluation Results

This document summarizes the evaluation results of different models on the emotion classification dataset.

## Dataset Information
- Validation set size: 800 samples
- 8 emotion classes: Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger, Contempt

## ResNet18 Results

### Classification Performance

```
              precision    recall  f1-score   support

     Neutral     0.4444    0.4000    0.4211       100
       Happy     0.6275    0.6400    0.6337       100
         Sad     0.5581    0.4800    0.5161       100
    Surprise     0.4953    0.5300    0.5121       100
        Fear     0.5268    0.5900    0.5566       100
     Disgust     0.4732    0.5300    0.5000       100
       Anger     0.4574    0.4300    0.4433       100
    Contempt     0.4227    0.4100    0.4162       100

    accuracy                         0.5012       800
   macro avg     0.5007    0.5012    0.4999       800
weighted avg     0.5007    0.5012    0.4999       800
```

### Summary Metrics
- Overall Accuracy: **0.5012**
- Macro F1-score: **0.4999**
- Cohen's Kappa: **0.4300**
- ROC-AUC (macro): **0.8229**
- PR-AUC (macro): **0.4871**
- Valence RMSE: **0.3867**, MAE: **0.2982**
- Arousal RMSE: **0.3283**, MAE: **0.2562**
- Krippendorff's Alpha: **0.4298**
- Valence CORR: **0.5805** | Arousal CORR: **0.5025**
- Valence SAGR: **0.7562** | Arousal SAGR: **0.8025**
- Valence CCC: **0.5486** | Arousal CCC: **0.4581**

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
- Krippendorff's Alpha: **0.4066**
- Valence CORR: **0.5681** | Arousal CORR: **0.4783**
- Valence SAGR: **0.7588** | Arousal SAGR: **0.7963**
- Valence CCC: **0.5193** | Arousal CCC: **0.3713**

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
- Krippendorff's Alpha: **0.4116**
- Valence CORR: **0.5693** | Arousal CORR: **0.4674**
- Valence SAGR: **0.7475** | Arousal SAGR: **0.7937**
- Valence CCC: **0.5403** | Arousal CCC: **0.4322**

## ResNet50 Results

### Classification Performance

```
              precision    recall  f1-score   support

     Neutral     0.4706    0.2400    0.3179       100
       Happy     0.7907    0.6800    0.7312       100
         Sad     0.6429    0.5400    0.5870       100
    Surprise     0.4345    0.6300    0.5143       100
        Fear     0.5410    0.6600    0.5946       100
     Disgust     0.5846    0.3800    0.4606       100
       Anger     0.4154    0.5400    0.4696       100
    Contempt     0.4786    0.5600    0.5161       100

    accuracy                         0.5288       800
   macro avg     0.5448    0.5288    0.5239       800
weighted avg     0.5448    0.5288    0.5239       800
```

### Summary Metrics
- Overall Accuracy: **0.5288**
- Macro F1-score: **0.5239**
- Cohen's Kappa: **0.4614**
- ROC-AUC (macro): **0.8822**
- PR-AUC (macro): **0.5706**
- Valence RMSE: **0.3732**, MAE: **0.2911**
- Arousal RMSE: **0.3189**, MAE: **0.2567**
- Krippendorff's Alpha: **0.4596**
- Valence CORR: **0.6362** | Arousal CORR: **0.5491**
- Valence SAGR: **0.7863** | Arousal SAGR: **0.7925**
- Valence CCC: **0.5853** | Arousal CCC: **0.4704**

## Models Comparison

| Model | Accuracy | F1-score | Valence RMSE | Arousal RMSE |
|-------|----------|----------|--------------|--------------|
| ResNet18 | 0.5012 | 0.4999 | 0.3867 | 0.3283 |
| ResNet34 | 0.4825 | 0.4692 | 0.3885 | 0.3341 |
| EfficientNet-B0 | 0.4863 | 0.4837 | 0.3962 | 0.3487 |
| ResNet50 | 0.5288 | 0.5239 | 0.3732 | 0.3189 |

ResNet50 shows the best overall classification performance with the highest accuracy (0.5288) and F1-score (0.5239). For regression, ResNet50 achieves the lowest Valence RMSE (0.3732) and the lowest Arousal RMSE (0.3189) among the compared models.