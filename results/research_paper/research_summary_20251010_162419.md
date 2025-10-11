
# Joint NIDS Model - Research Evaluation Report
Generated on: 2025-10-10 16:24:23

## Model Architecture
- **Approach**: Joint Training (Autoencoder + CapsNet)
- **Input Features**: 54 (after preprocessing)
- **Classes**: 4 (Non-Tor, NonVPN, Tor, VPN)
- **Dataset**: CIC-Darknet2020 (141,530 samples)

## Performance Metrics
- **Test Accuracy**: 0.9483 (94.83%)
- **Macro F1-Score**: 0.9113
- **Weighted F1-Score**: 0.9480
- **Micro-average AUC**: 0.9950

## Per-Class Performance

### Non-Tor
- Precision: 0.9888
- Recall: 0.9955
- F1-Score: 0.9921
- Support: 13822

### NonVPN
- Precision: 0.9041
- Recall: 0.8629
- F1-Score: 0.8830
- Support: 4762

### Tor
- Precision: 0.9466
- Recall: 0.8298
- F1-Score: 0.8844
- Support: 235

### VPN
- Precision: 0.8716
- Recall: 0.9005
- F1-Score: 0.8858
- Support: 4583

## Improvement over Baseline
- **Accuracy Improvement**: +258.8%
- **Approach**: Joint training vs. separate training
- **Key Innovation**: Shared feature learning between autoencoder and CapsNet

## Files Generated
- Metrics: results/research_paper/joint_nids_metrics_20251010_162419.csv
- Confusion Matrix: results/research_paper/confusion_matrix_20251010_162419.png
- ROC Curves: results/research_paper/roc_curves_20251010_162419.png
- Precision-Recall Curves: results/research_paper/precision_recall_curves_20251010_162419.png
- Performance Comparison: results/research_paper/performance_comparison_20251010_162419.csv
