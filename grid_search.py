import optuna
from src.tools import Dataset
from src.model import KSVM
from src.kernel import KmerKernel, KmerMismatchKernel
from functools import partial
from optuna.pruners import MedianPruner
from joblib import Parallel, delayed
import json

if __name__ == "__main__":

    cv = 3  
    best_params_per_idx = {}

    for train_idx in range(3, 4):
        print(f"Optimizing hyperparameters for train_idx={train_idx}")
        dataset = Dataset(train_idx=train_idx)

        def objective(trial):
            kmin = trial.suggest_int("kmin", 2, 30)
            kmax = trial.suggest_int("kmax", 2, 30)
            C = trial.suggest_float("C", 1e-3, 1e2, log=True)
            m = trial.suggest_int("m", 1, kmin-1)

            if kmin >= kmax:
                return 0 

            mean_accuracy = 0
            for _ in range(cv):
                Xtr, Xte, Ytr, Yte = dataset.train_test_split(test_size=0.2)
                ksvm = KSVM(partial(KmerMismatchKernel, kmin=kmin, kmax=kmax, m=m), C=C, tol=1e-5)
                alpha, beta = ksvm.fit(Xtr, Ytr)
                score = ksvm.score_recall_precision(Xte, Yte)
                mean_accuracy += score.accuracy
            
            return mean_accuracy / cv


        def optimize(n_trials, study_name):
            study = optuna.load_study(study_name=study_name, storage='sqlite:///' + str(study_name) + '.db')
            study.optimize(objective, n_trials=n_trials)
            return study

        study_name = "ksvm_tuning_" + str(train_idx)
        n_trials = 1

        pruner = MedianPruner(n_warmup_steps=5)
        study = optuna.create_study(
            direction="maximize", study_name=study_name,
            storage='sqlite:///' + str(study_name) + '.db', load_if_exists=True, pruner=pruner
        )
        r = Parallel(n_jobs=1)([delayed(optimize)(1, study_name) for _  in range(n_trials)])

        best_params_per_idx[train_idx] = {"params": study.best_params, "accuracy": study.best_value}

    with open("best_params_per_idx.json", "w") as f:
        json.dump(best_params_per_idx, f, indent=4)

