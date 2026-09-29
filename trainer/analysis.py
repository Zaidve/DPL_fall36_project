"""
    Command line for everything around the experiment matrix (experiment_matrix_spec.md §8).

      python -m trainer.analysis matrix                        run_matrix.csv + counts per experiment
      python -m trainer.analysis status [--models DIR ...]     progress, failures, remaining GPU-hours, session plan
      python -m trainer.analysis merge --from DIR ... --to DIR [--apply]
      python -m trainer.analysis check [--models DIR ...]      consistency warnings
      python -m trainer.analysis select --stage backbone|loss|imbalance [--allow-partial] [--force]
      python -m trainer.analysis select --explain              current choices + provenance
      python -m trainer.analysis significance [--n-boot N]     standard pairs: McNemar per seed + paired bootstrap
      python -m trainer.analysis tables                        results_<dataset>, rq1-rq5, cost, e6_mlm (+ marks)
      python -m trainer.analysis figures                       learning curves, low resource, per class, confusion, ...
      python -m trainer.analysis cross-eval [--device D] [--force]   E7: kept checkpoints on the other dataset
      python -m trainer.analysis all                           significance + tables + figures (+ cross-eval if
                                                               checkpoints exist)

    Every subcommand works on a partially finished matrix and prints what it skipped.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trainer import status as S  # noqa: E402
from utils.common import models_dir, reports_dir  # noqa: E402

def _show(df, title):
    print(f'\n{title}')
    if df is None or len(df) == 0:
        print('  (none)')
        return
    with pd.option_context('display.width', 200, 'display.max_rows', 500, 'display.max_columns', 30,
                           'display.max_colwidth', 90):
        print(df.to_string(index=False))


def cmd_matrix(args):
    matrix = S.build_matrix()
    counts = matrix.groupby(['experiment', 'priority'], sort=False).agg(
        runs=('run_id', 'size'), keep_checkpoint=('keep_checkpoint', 'sum'), placeholders=('placeholder', 'sum'))
    _show(counts.reset_index(), 'Run matrix')
    print(f'\n{len(matrix)} runs, {int(matrix["keep_checkpoint"].sum())} keep checkpoints -> '
          f'{reports_dir() / "tables" / "run_matrix.csv"}')
    return matrix


def cmd_status(args):
    dirs = [Path(d) for d in (args.models or [models_dir()])]
    matrix = S.build_matrix()
    status = S.run_status(matrix, dirs)
    out = reports_dir() / 'tables' / 'run_status.csv'
    status.to_csv(out, index=False)
    _show(S.status_summary(status), f'Progress ({", ".join(map(str, dirs))})')
    failed = status[status['status'] == 'failed'][['run_id', 'error']]
    _show(failed, f'Failed runs ({len(failed)}): see models/<run_id>/error.txt')
    per_exp, plan = S.estimate_remaining(status)
    _show(per_exp, 'Remaining (estimated GPU-hours; blocked runs need selection.json first)')
    _show(plan, f'Session plan (<= {S.SESSION_HOURS:.0f} h per Kaggle session, Must first)')
    print(f'\nper-run status -> {out}')
    return status


def cmd_merge(args):
    log = S.merge_models_dirs(args.sources, args.dest, dry_run=not args.apply)
    summary = log.groupby('action').size().rename('runs').reset_index() if len(log) else log
    _show(summary, 'Merge ' + ('(applied)' if args.apply else '(dry run: add --apply to copy)'))
    conflicts = log[log['action'] == 'conflict']
    _show(conflicts, f'Conflicts ({len(conflicts)})')
    return log


def cmd_check(args):
    dirs = [Path(d) for d in (args.models or [models_dir()])]
    status = S.run_status(S.build_matrix(save=False), dirs)
    warnings = S.check_consistency(status, dirs)
    print(f'\n{len(warnings)} warning(s)')
    for w in warnings:
        print(f'  - {w}')
    return warnings


def cmd_select(args):
    from trainer import selection as SEL
    if args.explain:
        return SEL.explain_selection()
    if not args.stage:
        raise SystemExit('select needs --stage backbone|loss|imbalance (or --explain)')
    dirs = [Path(d) for d in (args.models or [models_dir()])]
    try:
        selection = SEL.write_selection(args.stage, dirs, allow_partial=args.allow_partial, force=args.force)
    except SEL.SelectionError as e:
        print(e)
        return None
    print(f'stage `{args.stage}` written -> {SEL.selection_path()}\n')
    SEL.explain_selection()
    return selection


def _context(args):
    from trainer.tables import load_context
    dirs = [Path(d) for d in (getattr(args, 'models', None) or [models_dir()])]
    ctx = load_context(dirs)
    chosen = ', '.join(f'{d}={v.get("backbone")}' for d, v in ctx['selection'].items()) or 'none'
    print(f'{ctx["runs"]["run_id"].nunique()} finished runs in {", ".join(map(str, dirs))}; selection: {chosen}')
    return ctx


def cmd_significance(args, ctx=None):
    from trainer import significance as SIG
    ctx = ctx or _context(args)
    available = set(ctx['agg']['config']) if len(ctx['agg']) else set()
    pairs = SIG.standard_pairs(ctx['selection'], available, ctx['agg'])
    table = SIG.compare(pairs, ctx['index'], n_boot=args.n_boot)
    _show(table.drop(columns=['config_a', 'config_b'], errors='ignore').assign(
        pair=[f'{a.split("|", 2)[2]} -> {b.split("|", 2)[2]}' for a, b in zip(table['config_a'], table['config_b'])])
        if len(table) else table, f'Significance ({len(pairs)} pairs with finished runs)')
    return table


def cmd_tables(args, ctx=None):
    from trainer.tables import build_all
    ctx = ctx or _context(args)
    tables, skipped = build_all(ctx=ctx)
    written = {k: len(v) for k, v in tables.items() if v is not None and len(v)}
    print(f'\ntables written to {reports_dir() / "tables"}: ' + ', '.join(f'{k} ({n} rows)' for k, n in written.items()))
    if skipped:
        print(f'skipped main-table rows without finished runs ({len(skipped)}): ' + '; '.join(skipped))
    empty = [k for k, v in tables.items() if v is None or not len(v)]
    if empty:
        print('not written yet (no finished runs): ' + ', '.join(empty))
    return tables


def cmd_figures(args, ctx=None):
    from trainer.figures import build_all
    ctx = ctx or _context(args)
    made, skipped = build_all(ctx)
    print(f'\nfigures written to {reports_dir() / "figures"}: ' + (', '.join(made) or 'none'))
    if skipped:
        print('skipped (runs not finished yet): ' + ', '.join(skipped))
    return made


def cmd_cross_eval(args):
    from trainer.cross_eval import cross_eval_all
    dirs = [Path(d) for d in (getattr(args, 'models', None) or [models_dir()])]
    runs, table = cross_eval_all(dirs, device=getattr(args, 'device', None), force=getattr(args, 'force', False))
    if runs.empty:
        print('\nno finished run with best.pt and a sentiment head yet (E1 st_sentiment/sum, E1 mtl/sum, E4 final)')
        return runs, table
    _show(table, f'E7 cross-dataset, 3 sentiment labels ({runs["run_id"].nunique()} checkpoints)')
    print(f'\n-> {reports_dir() / "tables" / "e7_cross_dataset.md"}; per-run predictions in models/<run_id>/cross_*_test.csv')
    return runs, table


def cmd_all(args):
    ctx = _context(args)
    out = {'significance': cmd_significance(args, ctx), 'tables': cmd_tables(args, ctx),
           'figures': cmd_figures(args, ctx)}
    dirs = [Path(d) for d in (getattr(args, 'models', None) or [models_dir()])]
    if any(any(d.glob('*/best.pt')) for d in dirs if d.is_dir()):
        out['cross_eval'] = cmd_cross_eval(args)
    else:
        print('\ncross-eval skipped: no kept checkpoints (best.pt) yet')
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m trainer.analysis', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('matrix', help='write run_matrix.csv, print counts per experiment')
    p = sub.add_parser('status', help='progress, failures, remaining GPU-hours, session plan')
    p.add_argument('--models', nargs='+', help='models/ folders to look in (default: models dir)')
    p = sub.add_parser('merge', help='copy finished runs from several session outputs into one models/')
    p.add_argument('--from', dest='sources', nargs='+', required=True)
    p.add_argument('--to', dest='dest', required=True)
    p.add_argument('--apply', action='store_true', help='copy (default: dry run)')
    p = sub.add_parser('check', help='consistency warnings for finished runs')
    p.add_argument('--models', nargs='+')
    p = sub.add_parser('select', help='write a selection.json stage (validation scores only)')
    p.add_argument('--stage', choices=['backbone', 'loss', 'imbalance'])
    p.add_argument('--allow-partial', action='store_true', help='allow candidates with fewer than 3 seeds')
    p.add_argument('--force', action='store_true', help='re-select a written stage (drops later stages)')
    p.add_argument('--explain', action='store_true', help='print the current choices and their provenance')
    p.add_argument('--models', nargs='+')
    for name, text in (('significance', 'McNemar per seed + paired bootstrap for the standard pairs'),
                       ('tables', 'all result tables (CSV + Markdown)'), ('figures', 'all figures'),
                       ('all', 'significance + tables + figures')):
        p = sub.add_parser(name, help=text)
        p.add_argument('--models', nargs='+')
        p.add_argument('--n-boot', type=int, default=1000, help='bootstrap samples (significance)')
        if name == 'all':
            p.add_argument('--device', default=None)
            p.add_argument('--force', action='store_true', help='re-run cached cross-dataset evaluations')
    p = sub.add_parser('cross-eval', help='E7: evaluate kept checkpoints on the other dataset (3 sentiment labels)')
    p.add_argument('--models', nargs='+')
    p.add_argument('--device', default=None, help='cuda / cpu (default: cuda if available)')
    p.add_argument('--force', action='store_true', help='re-evaluate runs that have cross_eval.json')
    args = parser.parse_args(argv)

    (reports_dir() / 'tables').mkdir(parents=True, exist_ok=True)
    return {'matrix': cmd_matrix, 'status': cmd_status, 'merge': cmd_merge, 'check': cmd_check,
            'select': cmd_select, 'significance': cmd_significance, 'tables': cmd_tables,
            'figures': cmd_figures, 'cross-eval': cmd_cross_eval, 'all': cmd_all}[args.command](args)


if __name__ == '__main__':
    main()
