"""Single-file results preserve metrics, predictions, diagnostics and calibration."""
from pathlib import Path
import copy
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from umsm.config import load_config
from umsm.experiment import run_experiment
from umsm.report import read_report, write_report
from umsm.types import IntegrationSet, Table
from test_ablation import GT, _toy_tables


def test_single_report_keeps_predictions_and_optional_diagnostics():
    source = IntegrationSet('Toy', 'Toy', [Table(name, df) for name, df in _toy_tables(1).items()], GT)
    with tempfile.TemporaryDirectory() as folder:
        cfg = load_config(overrides=['matcher.name=value_overlap_test', 'candidates.mode=all',
                                     'matcher.cache_scores=false'])
        cfg['experiment']['output_dir'] = str(Path(folder) / 'results')
        cfg['paths']['cache_dir'] = str(Path(folder) / 'cache')
        cfg['global_optimizer'] = {'name': 'ilp', 'params': {'theta': .4, 'top_l': None}}
        with patch('umsm.experiment.load_sets', return_value=[source]):
            result = run_experiment(cfg, folder)
        saved = read_report(result['result_file'])
        assert len(list((Path(folder)/'results').rglob('*.*'))) == 1
        assert saved['metrics_mean_over_sets'] == result['metrics_mean_over_sets']
        assert saved['predictions'][0]['matching']['groups']
        assert saved['predictions'][0]['matching']['TP'] + saved['predictions'][0]['matching']['FN'] == 4
        assert saved['config'] == cfg and not saved['diagnostics']
        detailed = copy.deepcopy(cfg)
        for flag in ('save_edges', 'save_uncertainty', 'save_score_samples', 'save_realizations'):
            detailed['experiment'][flag] = True
        with patch('umsm.experiment.load_sets', return_value=[source]):
            full = run_experiment(detailed, folder)
        report = read_report(full['result_file'])
        detail = report['diagnostics'][0]
        assert detail['edges'] and detail['score_samples'] and detail['distributions'] and detail['realizations']
        assert report['predictions'] == saved['predictions'] or report['predictions'][0]['matching']['groups'] == saved['predictions'][0]['matching']['groups']
        assert len(list((Path(folder)/'results').rglob('*.*'))) == 2


def test_report_roundtrip_escapes_untrusted_cells_and_preserves_nulls():
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)/'result.html'
        value = '</script><img src=x onerror=alert(1)>'
        payload = {'run_id': 'test', 'name': value, 'rows': [{'set': value, 'pair_f1': float('nan')}],
                   'summary': [{'pair_f1_mean': .75}], 'config': {'cell': value}}
        write_report(path, payload)
        saved = read_report(path)
        assert saved['config']['cell'] == value and saved['rows'][0]['pair_f1'] is None
        assert value not in path.read_text(encoding='utf-8')
        assert list(Path(folder).iterdir()) == [path]


def test_new_reports_can_be_compared_without_sidecar_files():
    from compare import compare_ablation, compare_against
    with tempfile.TemporaryDirectory() as folder:
        ours, base = Path(folder)/'ours.html', Path(folder)/'base.html'
        rows = [{'variant': 'original', 'set': 'Toy', 'missing': s, 'optimizer': o, 'pair_f1': f}
                for s, o, f in [('observed_only', 'global', .4), ('top1', 'global', .5),
                                 ('Q_mean', 'global', .6), ('Q_risk', 'global', .7), ('Q_risk', 'greedy', .55)]]
        write_report(ours, {'run_id': 'ours', 'config': {'matcher': {'name': 'magneto'}}, 'rows': rows})
        write_report(base, {'run_id': 'base', 'rows': [{**rows[0], 'missing': 'observed_only', 'optimizer': 'local', 'pair_f1': .6}]})
        _, effects = compare_ablation(str(ours))
        cross = compare_against(str(ours), str(base))
        assert abs(effects.set_index('component').loc['global_optimizer','delta_pp']-15)<1e-10
        assert abs(cross.iloc[0].delta_pp-10)<1e-10
        assert len(list(Path(folder).iterdir())) == 2


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
