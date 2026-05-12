import itertools
import json
import subprocess
import shutil
import argparse
from pathlib import Path


def generate_combinations(grid):
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))

def make_label(params):
    suffix = "_".join([f"{k}{v}" for k, v in params.items()])
    return f"grid_search_{suffix}"

def run_training(run_dir, label, params):
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "hyperparam.json", "w") as f:
        json.dump(params, f, indent=4)

    cmd = ["bash", "src/train.sh", "--label", label]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, default="grid_search")
    return parser.parse_args()

def main():
    args = parse_args()
    run_root = Path('data') 
    grid_search_dir = run_root / args.label

    with open(grid_search_dir / 'grid.json', 'r') as f:
        grid = json.load(f)

    best_f1 = -1
    best_result = None
    combos = list(generate_combinations(grid))
    
    print(f"========= STARTING GRID SEARCH... =========")
    for i, params in enumerate(combos):
        print(f"========= GRID SEARCH - RUN {i} OF {len(combos)} =========")
        
        label = make_label(params)
        run_dir = run_root / label

        try:
            run_training(run_dir, label, params)
            with open(run_dir / 'metrics.json', 'r') as f:
                metrics = json.load(f)
            f1 = metrics["F1"]
            result = {
                "label": label,
                "params": params,
                "metrics": metrics,
                "F1": f1,
            }
            if f1 > best_f1:
                best_f1 = f1
                best_result = result
            print(f"========= GRID SEARCH - RUN ENDED WITH F1={f1}, BEST={best_f1} ========= ")
        except subprocess.CalledProcessError:
            print(f"========= GRID SEARCH - RUN FAILED FOR {label} ========= ")
        finally:
            shutil.rmtree(run_dir)
    
    with open(grid_search_dir / 'best.json', 'w') as f:
        json.dump(best_result, f, indent=4)
    print("========= GRID SEARCH - BEST RESULTS:", best_result)


if __name__ == "__main__":
    main()