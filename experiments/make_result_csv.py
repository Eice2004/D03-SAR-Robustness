# -*- coding: utf-8 -*-
"""把九份扫描结果合并成一张画图用的 CSV。"""

import csv
import json
from pathlib import Path

BASE = Path(__file__).parent
GROUPS = ["clean", "noise-p050", "noise-p100"]
SEEDS = [12456, 12457, 12477]
OUT = BASE / "degraded" / "degraded_accuracy.csv"


def main():
    rows = []
    for group in GROUPS:
        for seed in SEEDS:
            path = BASE / "degraded" / f"{group}-seed{seed}.json"
            if not path.exists():
                print("跳过", path.name)
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            for item in data["results"]:
                rows.append([
                    group, seed, item["L"],
                    f"{item['accuracy']:.2f}", f"{item['macro_f1']:.2f}",
                    f"{item['inference_ms']:.3f}",
                ])

    rows.sort(key=lambda r: (GROUPS.index(r[0]), r[1], r[2]))

    with OUT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "seed", "L", "accuracy", "macro_f1", "inference_ms"])
        writer.writerows(rows)

    print("写入", OUT, len(rows), "行")


if __name__ == "__main__":
    main()
