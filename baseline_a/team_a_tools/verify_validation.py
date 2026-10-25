"""Evaluate the frozen Max checkpoint on validation using the repository's own evaluator."""
import argparse
import csv
import math
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path.cwd())
    parser.add_argument('--data-root', type=Path, default=Path('data/LOL'))
    parser.add_argument('--output', type=Path, default=Path('runs/a_member_a_validation'))
    args = parser.parse_args()
    root = args.project_root.resolve()
    sys.path.insert(0, str(root / 'scripts'))
    import torch
    from torch.utils.data import DataLoader
    from common import ManifestPairs, environment, seed_runtime, sha256, write_json
    from evaluate import METRIC_PROTOCOL, validate_completed_run, validate_manifests
    from train_a import validate

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Output folder is not empty. Use a new output name.')
    seed_runtime(0, 1)
    run_dir = root / 'reports/a_full_seed42'
    model, run, verification = validate_completed_run(run_dir)
    _, manifest_hashes = validate_manifests(root / 'manifests', run, run_dir)
    dataset = ManifestPairs(root / 'manifests/validation.json', args.data_root)
    for row in dataset.rows:
        for kind in ('low', 'high'):
            path = args.data_root / row[kind]
            if sha256(path) != row[kind + '_sha256']:
                raise ValueError('Validation image hash mismatch: ' + str(path))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    aggregate, scores = validate(model, loader, torch.device('cpu'))
    sizes = {row['name']: (row['width'], row['height']) for row in dataset.rows}
    for row in scores:
        row['width'], row['height'] = sizes[row['name']]
        if not all(math.isfinite(row[key]) for key in ('psnr', 'ssim')):
            raise ValueError('Non-finite score: ' + row['name'])
    report = {
        'dataset': 'LOL-v1', 'split': 'validation', 'count': len(scores),
        'aggregate': aggregate, 'per_image': scores, 'metrics': METRIC_PROTOCOL,
        'verification': verification, 'manifests_sha256': manifest_hashes,
        'environment': environment(), 'score_source': 'validation_recheck',
        'checkpoint_path': 'reports/a_full_seed42/best.pt',
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'metrics.json', report)
    with (output / 'per_image.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['name','width','height','psnr','ssim'])
        writer.writeheader()
        writer.writerows(scores)
    print({'count': len(scores), 'selected_epoch': verification['selected_epoch'], **aggregate})


if __name__ == '__main__':
    main()
