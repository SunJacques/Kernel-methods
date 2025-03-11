from src.tools import Dataset, Score
from src.model import KSVM
from src.kernel import KmerKernel
from functools import partial
import json
import numpy as np
import pandas as pd

def load_config(config_path="config.json"):
    """Load configuration from a JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)

def prepare_data(idx, to_submit):
    """Load and split the dataset based on the mode."""
    if to_submit:
        dataset_tr = Dataset(train_idx=idx)
        dataset_te = Dataset(train_idx=idx, mode="test")
        Xtr, Ytr = dataset_tr.get()
        Xte, Yte = dataset_te.get(shuffle=False)
    else:
        dataset = Dataset(train_idx=idx)
        Xtr, Xte, Ytr, Yte = dataset.train_test_split(test_size=0.2)
    return Xtr, Xte, Ytr, Yte

def train_and_evaluate(Xtr, Ytr, Xte, Yte, config, idx, to_submit):
    """Train KSVM and evaluate predictions."""
    ksvm = KSVM(
        kernel=partial(KmerKernel, kmin=config["kmin"][idx], kmax=config["kmax"][idx]),
        C=config["C"][idx],
        tol=config["tol"][idx]
    )
    ksvm.fit(Xtr, Ytr)
    predictions = ksvm.predict(Xte)

    if not to_submit:
        score = ksvm.score_recall_precision(Xte, Yte)
        print(f"Train idx {idx} Score: {score}")

    return predictions, Yte

def save_predictions(predictions, output_path):
    """Save predictions to a CSV file for submission."""
    predictions = ((predictions + 1) // 2).astype(np.int8)
    index = np.arange(len(predictions))
    df = pd.DataFrame({"Bound": predictions}, index=index)
    df.index.name = "Id"    
    df.to_csv(output_path)
    print(f"Predictions saved to {output_path}")

def main():
    conf = load_config()
    to_submit = conf["submit"]
    all_pred, all_label = [], []

    for idx in range(3):
        print(f"Processing dataset {idx}...")
        Xtr, Xte, Ytr, Yte = prepare_data(idx, to_submit)
        print(f"Train size: {len(Xtr)}, Test size: {len(Xte)}")
        pred, labels = train_and_evaluate(Xtr, Ytr, Xte, Yte, conf, idx, to_submit)
        print(f"Predictions for dataset {idx} done.")
        all_pred.append(pred)
        all_label.append(labels)

    all_pred = np.concatenate(all_pred)

    if to_submit:
        save_predictions(all_pred, "export/" + conf["output"])
    else:
        all_label = np.concatenate(all_label)
        final_score = Score(all_pred, all_label)
        print("Final Score:", final_score)

if __name__ == "__main__":
    main()