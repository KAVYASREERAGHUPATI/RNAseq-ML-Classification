"""Generate the five original evaluation figures from saved CSVs, without retraining."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

METRICS = ['Balanced_Accuracy', 'ROC_AUC', 'Sensitivity', 'Specificity', 'F1_Score']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results'))
    parser.add_argument('--figures', type=Path, default=Path('figures'))
    args = parser.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style='whitegrid', context='notebook')
    scores = pd.read_csv(args.results / 'seven_models_comparison.csv')
    predictions = pd.read_csv(args.results / 'seven_models_patient_predictions.csv')
    folds = pd.read_csv(args.results / 'outer_fold_performance.csv')

    long = scores.melt(id_vars='Model', value_vars=METRICS, var_name='Metric', value_name='Score')
    plt.figure(figsize=(14, 7))
    sns.barplot(data=long, x='Model', y='Score', hue='Metric')
    plt.ylim(0, 1.05)
    plt.ylabel('Score')
    plt.xlabel('')
    plt.title('Performance Comparison of Seven ML Models')
    plt.xticks(rotation=35, ha='right')
    plt.legend(title='Metric', bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.savefig(args.figures / 'model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(9, 7))
    for name in scores['Model']:
        subset = predictions[predictions['Model'] == name]
        fpr, tpr, _ = roc_curve(subset['Actual'], subset['Responder_Probability'])
        auc = roc_auc_score(subset['Actual'], subset['Responder_Probability'])
        plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC={auc:.3f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Out-of-Fold ROC Curves — Seven ML Models')
    plt.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig(args.figures / 'roc_curves_all_models.png', dpi=300, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(2, 4, figsize=(17, 9))
    axes = axes.flatten()
    for i, name in enumerate(scores['Model']):
        subset = predictions[predictions['Model'] == name]
        cm = confusion_matrix(subset['Actual'], subset['Predicted'], labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['NonResponder', 'Responder'],
                    yticklabels=['NonResponder', 'Responder'], ax=axes[i])
        axes[i].set_title(name)
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    axes[-1].axis('off')
    plt.suptitle('Out-of-Fold Confusion Matrices', fontsize=15)
    plt.tight_layout()
    plt.savefig(args.figures / 'confusion_matrices_all_models.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=folds, x='Model', y='Balanced_Accuracy', color='lightblue')
    sns.stripplot(data=folds, x='Model', y='Balanced_Accuracy', color='black', size=5, alpha=0.75)
    plt.ylim(0, 1.05)
    plt.xlabel('')
    plt.ylabel('Balanced Accuracy')
    plt.title('Model Performance Across Five Outer CV Folds')
    plt.xticks(rotation=35, ha='right')
    plt.tight_layout()
    plt.savefig(args.figures / 'outer_fold_stability.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.heatmap(scores.set_index('Model')[METRICS], annot=True, fmt='.3f',
                cmap='YlGnBu', vmin=0, vmax=1, linewidths=0.5)
    plt.title('Seven-Model Performance Heatmap')
    plt.xlabel('Evaluation Metric')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(args.figures / 'model_metrics_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved five figures to {args.figures.resolve()}')


if __name__ == '__main__':
    main()
