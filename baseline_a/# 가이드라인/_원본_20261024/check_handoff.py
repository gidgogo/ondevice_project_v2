"""Read-only checks for the simplified team-A delivery (standard library only).

This validates files and recorded results. It does not perform model inference,
verify recipient dataset access, or establish the actual evaluation timestamp.
"""
import argparse
import csv
import hashlib
import json
import math
import struct
from pathlib import Path, PurePosixPath


QUALITY_FIELDS = [
    'image_id', 'split', 'model_id', 'checkpoint_id',
    'width', 'height', 'psnr_db', 'ssim',
]
REFERENCE_COMMIT = '3ead18e7b34444aa19eb39c815da86d2dc024d08'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_path(text):
    path = PurePosixPath(text)
    return bool(text) and not path.is_absolute() and '..' not in path.parts and '\\' not in text


def same_number(actual, expected, message):
    require(math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=1e-10), message)


def check(folder):
    folder = Path(folder).resolve()
    required = [
        'README.md', 'experiment_settings.md', 'split_manifest.csv',
        'baseline_quality.csv', 'baseline_validation_quality.csv',
        'quality_summary.csv', 'enhanced_manifest.csv', 'best.pt',
        'common_metrics.py', 'requirements.lock.txt', 'LICENSE',
        'evidence/test_metrics.json', 'evidence/validation_metrics.json',
        'evidence/recheck_status.json', 'team_a_tools/verify_validation.py',
        'team_a_tools/check_handoff.py',
    ]
    for name in required:
        path = (folder / name).resolve()
        require(path.is_relative_to(folder) and path.is_file(), 'Missing or invalid file: ' + name)
    status = read_json(folder / 'evidence/recheck_status.json')
    require(status['reference_commit'] == REFERENCE_COMMIT, 'Reference commit changed.')
    require(status['checkpoint_id'] == 'a_max_seed42_epoch20', 'Wrong checkpoint ID.')
    for file, key in [('best.pt', 'checkpoint_sha256'), ('common_metrics.py', 'common_metrics_sha256'),
                      ('requirements.lock.txt', 'requirements_sha256')]:
        require(sha(folder / file) == status[key], 'Hash mismatch: ' + file)

    fields, manifest = read_csv(folder / 'split_manifest.csv')
    require(fields == ['image_id', 'split', 'source_name', 'low_path', 'gt_path', 'width', 'height'],
            'Wrong manifest fields.')
    lookup = {row['image_id']: row for row in manifest}
    require(len(manifest) == len(lookup) == 500, 'Expected 500 unique image IDs.')
    counts = {s: sum(row['split'] == s for row in manifest) for s in ('train', 'validation', 'test')}
    require(counts == {'train': 435, 'validation': 50, 'test': 15}, 'Wrong split counts.')
    for row in manifest:
        require(row['image_id'] == 'LOL-v1/' + row['source_name'], 'Wrong ID: ' + row['image_id'])
        require((int(row['width']), int(row['height'])) == (600, 400), 'Wrong declared image size.')
        prefix = 'eval15' if row['split'] == 'test' else 'our485'
        for col, kind in [('low_path', 'low'), ('gt_path', 'high')]:
            require(relative_path(row[col]) and row[col] == prefix + '/' + kind + '/' + row['source_name'],
                    'Wrong dataset-relative path: ' + row[col])

    _, summary = read_csv(folder / 'quality_summary.csv')
    require(len(summary) == 2 and {r['split'] for r in summary} == {'test', 'validation'},
            'Expected exactly test and validation summary rows.')
    reports = {split: read_json(folder / ('evidence/' + split + '_metrics.json'))
               for split in ('test', 'validation')}
    protocol = reports['test']['metrics']
    require(protocol == reports['validation']['metrics'], 'Metric protocols differ.')
    for key, value in {'color_space': 'RGB', 'dtype': 'float32', 'data_range': 1.0,
                       'prediction_clamp': [0.0, 1.0], 'border_crop': 0,
                       'brightness_matching': False, 'quantize_before_scoring': False,
                       'aggregation': 'arithmetic mean of per-image scores'}.items():
        require(protocol.get(key) == value, 'Wrong metric protocol: ' + key)
    averages = {}
    for split, name, count in [('test', 'baseline_quality.csv', 15),
                               ('validation', 'baseline_validation_quality.csv', 50)]:
        fields, rows = read_csv(folder / name)
        require(fields == QUALITY_FIELDS, 'Wrong quality columns: ' + name)
        expected = {r['image_id'] for r in manifest if r['split'] == split}
        require(len(rows) == count and {r['image_id'] for r in rows} == expected,
                'Missing/duplicate/extra quality IDs: ' + split)
        report = reports[split]
        recorded = {r['name']: r for r in report['per_image']}
        require(report['count'] == len(report['per_image']) == len(recorded) == count,
                'Wrong JSON count: ' + split)
        require({'LOL-v1/' + name for name in recorded} == expected, 'JSON coverage differs: ' + split)
        require(report['environment'] == status['environment'], 'Environment records differ: ' + split)
        require(report['verification']['checkpoint_sha256']['best'] == status['checkpoint_sha256'],
                'JSON checkpoint differs: ' + split)
        require(report['verification']['selected_epoch'] == 20 and report['verification']['completed_epochs'] == 360,
                'Training selection record differs: ' + split)
        for row in rows:
            require(all(row.get(k) is not None and row[k].strip() for k in QUALITY_FIELDS),
                    'Blank quality field: ' + row['image_id'])
            require(row['split'] == split and row['model_id'] == 'UltraFastLiNET_Max'
                    and row['checkpoint_id'] == status['checkpoint_id'], 'Wrong quality identity.')
            meta = lookup[row['image_id']]
            raw = recorded[meta['source_name']]
            dims = (int(row['width']), int(row['height']))
            require(dims == (int(meta['width']), int(meta['height'])) == (raw['width'], raw['height']),
                    'Size records differ: ' + row['image_id'])
            for key, source_key in [('psnr_db', 'psnr'), ('ssim', 'ssim')]:
                score = float(row[key])
                require(math.isfinite(score), 'Non-finite score: ' + row['image_id'])
                require(score == float(raw[source_key]), 'CSV/JSON score mismatch: ' + row['image_id'])
        avg = {k: math.fsum(float(row[k]) for row in rows) / count for k in ('psnr_db', 'ssim')}
        aggregate = next(row for row in summary if row['split'] == split)
        require(int(aggregate['count']) == count, 'Summary count differs: ' + split)
        for key, report_key in [('psnr_db', 'psnr'), ('ssim', 'ssim')]:
            same_number(avg[key], aggregate[key + '_mean'], 'CSV average mismatch: ' + split)
            same_number(avg[key], report['aggregate'][report_key], 'JSON average mismatch: ' + split)
        require(aggregate['score_source'] == status['score_sources'][split], 'Score source differs: ' + split)
        averages[split] = avg
    require(reports['test']['test_used_for_checkpoint_selection'] is False, 'Unexpected test selection policy.')

    _, enhanced = read_csv(folder / 'enhanced_manifest.csv')
    require(len(enhanced) == status['enhanced_image_count'] == 15, 'Expected 15 enhanced images.')
    require({r['image_id'] for r in enhanced} == {r['image_id'] for r in manifest if r['split'] == 'test'},
            'Enhanced IDs differ from test IDs.')
    paths = set()
    for row in enhanced:
        require(row['split'] == 'test' and row['source_name'] == lookup[row['image_id']]['source_name'],
                'Wrong enhanced identity.')
        require(relative_path(row['enhanced_path']), 'Invalid enhanced path.')
        path = (folder / row['enhanced_path']).resolve()
        require(path.is_relative_to(folder) and path.is_file(), 'Missing enhanced PNG: ' + row['image_id'])
        with path.open('rb') as handle:
            header = handle.read(24)
        require(len(header) == 24 and header[:8] == b'\x89PNG\r\n\x1a\n' and header[12:16] == b'IHDR',
                'Invalid PNG header: ' + row['image_id'])
        require(struct.unpack('>II', header[16:24]) == (600, 400), 'Wrong PNG dimensions.')
        paths.add(path)
    require(len(paths) == 15 and paths == {p.resolve() for p in (folder / 'enhanced/test').glob('*.png')},
            'Duplicate or unmapped enhanced paths.')
    return {'status': 'PASS', 'check_type': 'file_consistency_only',
            'manifest_pairs': 500, 'split_counts': counts, 'test_images': 15,
            'validation_images': 50, 'enhanced_images': 15, 'means': averages,
            'note': 'No model inference, dataset-access verification or execution-time verification. PNG headers only.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path, nargs='?', default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        result = check(args.folder)
    except (ValueError, KeyError, OSError, TypeError) as exc:
        parser.exit(1, 'FAIL: ' + str(exc) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
