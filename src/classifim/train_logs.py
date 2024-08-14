import json
import numpy as np
import os
import pandas as pd
import re
import sys

def find_train_log_counted_key(keys):
    """
    Find the key that indicates the total time counted for the purpose of
    estimating FIM in a limited time budget.
    """
    key_candidates = ["scored", "classifim", "dtrun_only", "dt_total"]
    for key_candidate in key_candidates:
        if key_candidate in keys:
            return key_candidate
    raise ValueError(f"None of {key_candidates} in {keys}")

def read_log(filename, model_short_name=None):
    filename_stem = os.path.basename(filename)
    assert filename_stem.endswith(".log.json")
    filename_stem = filename_stem[:-9]
    with open(filename, "r") as f:
        try:
            log = json.load(f)
        except json.JSONDecodeError:
            print(f"Error reading '{file_name}'", file=sys.stderr)
            raise
        res = {"filename_stem": filename_stem}
        log_config = log["config"]
        if model_short_name is None:
            model_short_name = log_config["model_name"]
        res["name"] = model_short_name
        try:
            seed = log_config.get("seed")
            if seed is None:
                seed = log_config["suffix"]
            res["seed"] = seed
        except KeyError:
            print(f"'seed' or 'suffix' not in {list(log_config.keys())}",
                    file=sys.stderr)
            print(f"filename_stem: {filename_stem}", file=sys.stderr)
            print(f"log keys: {list(log.keys())}", file=sys.stderr)
            raise
        res["num_epochs"] = log_config["num_epochs"]
        num_for_avg = 1 + (res["num_epochs"] // 100)
        log_train = log["train"]
        if isinstance(log_train, list):
            res["model_type"] = "W"
            # W has multiple rounds of training: one for each slice.
            # Average over all rounds.
            assert all(
                len(lt["loss"]) == res["num_epochs"]
                for lt in log_train)
            res["train_ce"] = np.mean([
                lt["loss"][-num_for_avg:] for lt in log_train])
            train_n = np.sum([lt["num_points"] for lt in log_train])
            assert isinstance(train_n, np.int64)
            res["train_n"] = int(train_n)
            assert "test" not in log
            # test_ce = np.mean([lt["loss"] for lt in log_test])
            # test_n = np.sum([lt["num_points"] for lt in log_test])
            res["test_ce"] = np.nan
            res["test_n"] = -1 # has to be int
        else:
            res["model_type"] = "ClassiFIM"
            assert len(log_train["loss"]) == res["num_epochs"]
            res["train_ce"] = np.mean(log_train["loss"][-num_for_avg:])
            train_n = log_train["num_points"]
            assert isinstance(train_n, int)
            res["train_n"] = train_n
            log_test = log["test"]
            res["test_ce"] = log_test["loss"]
            res["test_n"] = log_test["num_points"]
        counted_key = find_train_log_counted_key(log["timings"].keys())
        res["time"] = log["timings"][counted_key]
    return res

def read_logs(log_dir, re_pattern=None, permissive=False):
    """
    Read logs from log_dir with filename stem matching re_pattern.

    Args:
        log_dir (str): Directory with log files.
        re_pattern (str): Regular expression pattern to match the filename stem
            (i.e. the filename without '.log.json' extension).
            None means no filtering.
        permissive (bool): If True, continue reading other logs after some fail.
    """
    train_logs = []
    if isinstance(re_pattern, str):
        re_pattern = re.compile(re_pattern)
    for filename in os.listdir(log_dir):
        if not filename.endswith(".log.json"):
            continue
        filename_stem = filename[:-9]
        if not re_pattern or re_pattern.match(filename_stem):
            try:
                train_logs.append(read_log(
                    os.path.join(log_dir, filename_stem + ".log.json")))
            except Exception as e:
                if permissive:
                    print(f"Error reading '{filename}': {e}", file=sys.stderr)
                    print(f"Continuing... (permissive={permissive})",
                          file=sys.stderr)
                else:
                    raise
    return pd.DataFrame(train_logs)

def aggregate_logs(train_logs, keys=None):
    """
    Aggregate train logs by "name" and "model_type".
    """
    columns = train_logs.columns
    if keys is None:
        keys = ["name", "model_type"]
    df = train_logs.groupby(by=keys).agg(
        **{
            column_name + ("" if fun == "mean" else f".{fun}"): (column_name, fun)
            for column_name in columns
            if (column_name not in keys + [
                "name", "model_type", "seed", "filename_stem"])
                and train_logs[column_name].dtype != object
            for fun in ("mean", "std")},
        cnt=("num_epochs", "size"))

    # Remove uninformative columns:
    for column_name in columns:
        std_name = column_name + ".std"
        if std_name not in df.columns:
            continue
        if np.all(df[std_name] == 0.0):
            del df[std_name]
    for column_name in df.columns:
        if np.all(df[column_name].isna()):
            del df[column_name]
    if "test_n" in df.columns and np.all(df["test_n"] == -1):
        del df["test_n"]

    return df
