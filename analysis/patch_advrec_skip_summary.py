from pathlib import Path

p = Path('analysis/adversarial_forget_recovery_eval.py')
s = p.read_text(encoding='utf-8')
orig = s

arg_line = '    ap.add_argument("--smoke_test", action="store_true")\n'
new_arg = arg_line + '    ap.add_argument("--skip_summary", action="store_true")\n'
if '--skip_summary' not in s:
    if arg_line not in s:
        raise SystemExit('arg insertion point not found')
    s = s.replace(arg_line, new_arg)

old = '    summary = write_summary(out_dir, dataset_rows, model_labels, baseline_candidates, args)\n    log("Summary written")\n'
new = '    if args.skip_summary:\n        log("Skipping summary write because --skip_summary is set")\n        return\n\n' + old
if 'Skipping summary write because --skip_summary is set' not in s:
    if old not in s:
        raise SystemExit('summary insertion point not found')
    s = s.replace(old, new)

if s != orig:
    p.write_text(s, encoding='utf-8')
    print('patched')
else:
    print('already patched')
