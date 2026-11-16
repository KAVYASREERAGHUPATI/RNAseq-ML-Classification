"""Select highest outer-CV balanced-accuracy model and refit on all supplied samples."""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocessing import RNASeqPreprocessor
from train_models import SEED, load_inputs, model_definitions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--counts', type=Path, required=True)
    p.add_argument('--metadata', type=Path, required=True)
    p.add_argument('--results', type=Path, default=Path('results'))
    p.add_argument('--models', type=Path, default=Path('models'))
    p.add_argument("--quick-demo", action="store_true", help="Match reduced quick-demo training grid")
    a = p.parse_args()
    X, y = load_inputs(a.counts, a.metadata)
    summary = pd.read_csv(a.results / 'seven_models_comparison.csv')
    if summary.empty or not {'Model', 'Balanced_Accuracy'}.issubset(summary.columns):
        raise ValueError('Missing model comparison data')
    if y.value_counts().min() < 4:
        raise ValueError('At least four samples per class are required for refit tuning')
    winner = summary.sort_values(['Balanced_Accuracy', 'Model'], ascending=[False, True]).iloc[0]
    name = str(winner['Model'])
    if name not in model_definitions():
        raise ValueError(f'Unknown model in results: {name}')
    estimator, model_grid = model_definitions()[name]
    pipe = Pipeline([
        ('rna', RNASeqPreprocessor()),
        ('select', SelectKBest(score_func=f_classif)),
        ('scale', StandardScaler()),
        ('model', clone(estimator)),
    ])
    if a.quick_demo:
        if name in ("Random Forest", "XGBoost"):
            estimator.set_params(n_estimators=20)
            pipe.set_params(model=clone(estimator))
        model_grid = {k: [20] if k == "model__n_estimators" else [v[0]] for k, v in model_grid.items()}
    search = GridSearchCV(pipe, {'select__k': ([20] if a.quick_demo else [20, 50, 100]), **model_grid},
                          scoring='balanced_accuracy',
                          cv=StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED),
                          n_jobs=1, refit=True, error_score='raise')
    search.fit(X, y)
    a.models.mkdir(parents=True, exist_ok=True)
    artifact = {'pipeline': search.best_estimator_, 'gene_ids': list(X.columns),
                'model_name': name, 'classes': {'NonResponder': 0, 'Responder': 1}}
    joblib.dump(artifact, a.models / 'best_model.joblib')
    info = {'selected_model': name,
            'outer_cv_balanced_accuracy': float(winner['Balanced_Accuracy']),
            'full_data_tuning_best_parameters': search.best_params_,
            'full_data_inner_cv_balanced_accuracy': float(search.best_score_),
            'training_samples': len(X), 'input_genes': X.shape[1],
            'note': 'Refit on all available labeled samples; no independent test performance is claimed.'}
    (a.models / 'model_info.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(f"Refitted {name}; saved {a.models / 'best_model.joblib'}")


if __name__ == '__main__':
    main()
