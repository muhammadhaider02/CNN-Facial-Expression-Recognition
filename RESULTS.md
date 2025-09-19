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

     Neutral     0.3514    0.3900    0.3697       100
       Happy     0.7313    0.4900    0.5868       100
         Sad     0.4508    0.5500    0.4955       100
    Surprise     0.5089    0.5700    0.5377       100
        Fear     0.5507    0.3800    0.4497       100
     Disgust     0.4821    0.5400    0.5094       100
       Anger     0.4861    0.3500    0.4070       100
    Contempt     0.4519    0.6100    0.5191       100

    accuracy                         0.4850       800
   macro avg     0.5017    0.4850    0.4844       800
weighted avg     0.5017    0.4850    0.4844       800
```

### Summary Metrics
- Overall Accuracy: **0.4850**
- Macro F1-score: **0.4844**
- Valence RMSE: **0.4604**, MAE: **0.3495**
- Arousal RMSE: **0.3767**, MAE: **0.3107**
- Krippendorff's Alpha: **0.4101**
- Valence CORR: **0.5000** | Arousal CORR: **0.4916**
- Valence SAGR: **0.7488** | Arousal SAGR: **0.7825**
- Valence CCC: **0.4059** | Arousal CCC: **0.3557**

## EfficientNet-B0 Results

### Classification Performance

```
              precision    recall  f1-score   support

     Neutral     0.3273    0.3600    0.3429       100
       Happy     0.6574    0.7100    0.6827       100
         Sad     0.5309    0.4300    0.4751       100
    Surprise     0.5510    0.5400    0.5455       100
        Fear     0.5529    0.4700    0.5081       100
     Disgust     0.4900    0.4900    0.4900       100
       Anger     0.4316    0.4100    0.4205       100
    Contempt     0.4472    0.5500    0.4933       100

    accuracy                         0.4950       800
   macro avg     0.4985    0.4950    0.4948       800
weighted avg     0.4985    0.4950    0.4948       800
```

### Summary Metrics
- Overall Accuracy: **0.4950**
- Macro F1-score: **0.4948**
- Valence RMSE: **0.3828**, MAE: **0.3009**
- Arousal RMSE: **0.3332**, MAE: **0.2683**
- Krippendorff's Alpha: **0.4225**
- Valence CORR: **0.5917** | Arousal CORR: **0.4922**
- Valence SAGR: **0.7438** | Arousal SAGR: **0.8075**
- Valence CCC: **0.5605** | Arousal CCC: **0.4558**

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

| Model | Accuracy | F1-score | Valence RMSE | Arousal RMSE | Training Time |
|-------|----------|----------|--------------|--------------|---------------|
| ResNet18 | 0.5012 | 0.4999 | 0.3867 | 0.3283 | 00:11:25 (685.9s) |
| ResNet34 | 0.4850 | 0.4844 | 0.4604 | 0.3767 | 00:08:33 (513.1s) |
| EfficientNet-B0 | 0.4950 | 0.4948 | 0.3828 | 0.3332 | 00:30:02 (1802.6s) |
| ResNet50 | 0.5288 | 0.5239 | 0.3732 | 0.3189 | 00:18:21 (1101.4s) |

ResNet50 shows the best overall classification performance with the highest accuracy (0.5288) and F1-score (0.5239). For regression, ResNet50 achieves the lowest Valence RMSE (0.3732) and the lowest Arousal RMSE (0.3189) among the compared models.