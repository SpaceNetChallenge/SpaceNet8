from pathlib import Path

import fire
import torch


def main(working_dir: str, n_snapshots: int):
    snapshots = sorted(Path(working_dir).glob("model_iter_*"), key=lambda x: int(x.name.split("_")[-1]))

    new_snapshot = None
    count = 0
    for path in snapshots[-n_snapshots:]:
        snapshot = torch.load(path, map_location="cpu")
        if new_snapshot is None:
            new_snapshot = snapshot
        else:
            assert set(new_snapshot.keys()) == set(snapshot.keys())
            new_snapshot = {key: value + snapshot[key] for key, value in new_snapshot.items()}
        count += 1
        print(path)
    assert new_snapshot is not None
    new_snapshot = {key: value / count for key, value in new_snapshot.items()}
    torch.save(new_snapshot, working_dir / f"swa_{n_snapshots}.pt")


if __name__ == "__main__":
    fire.Fire(main)
