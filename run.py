import argparse, os, json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    p0 = sub.add_parser('part0')
    p0.add_argument('--symbols', type=str, required=True)
    p0.add_argument('--start', type=str, required=True)
    p0.add_argument('--end', type=str, required=True)
    p0.add_argument('--outdir', type=str, default='/content/hybrid_workspace')
    p0.add_argument('--rep', type=str, default='clip', choices=['clip','dino'])
    p0.add_argument('--seed', type=int, default=0)

    pd = sub.add_parser('det')
    pd.add_argument('--data_yolo', type=str, required=True)
    pd.add_argument('--model', type=str, default='yolov8m')
    pd.add_argument('--epochs', type=int, default=80)

    pf = sub.add_parser('feats')
    pf.add_argument('--workspace', type=str, default='/content/hybrid_workspace')

    pab = sub.add_parser('ab')
    pab.add_argument('--workspace', type=str, default='/content/hybrid_workspace')
    pab.add_argument('--embargo', type=int, default=10)

    pbt = sub.add_parser('bt')
    pbt.add_argument('--workspace', type=str, default='/content/hybrid_workspace')
    pbt.add_argument('--fees_bps', type=float, default=2)
    pbt.add_argument('--slip_bps', type=float, default=2)

    args = p.parse_args()

    if args.cmd == 'part0':
        from freedom44.pipeline.part0_data import build_all
        sym = [s.strip() for s in args.symbols.split(',') if s.strip()]
        out = build_all(sym, args.start, args.end, args.outdir, seed=args.seed)
        print(json.dumps(out, indent=2))
        # embeddings if rep==clip
        if args.rep == 'clip':
            from freedom44.pipeline.clip_embed import build_clip_embeddings
            out_parquet = Path(out['clip_embeddings_parquet'])
            out_parquet.parent.mkdir(parents=True, exist_ok=True)
            build_clip_embeddings(out['images_manifest_csv'], str(out_parquet), model_name='ViT-L-14', batch=128, fp16=True, seed=args.seed)
    elif args.cmd == 'det':
        from freedom44.train.train_yolo import train_yolo, write_data_yaml_from_lists
        model_pref = 'yolov8m.pt' if args.model == 'yolov8m' else 'yolo11s.pt'
        root = Path(args.data_yolo)
        train_list = root / 'train.txt'
        val_list = root / 'val.txt'
        if not (train_list.exists() and val_list.exists()):
            raise RuntimeError(f"train.txt / val.txt not found under {root}. Run part0 first.")
        classes = [
            "fvg_bull","fvg_bear","ob_bull","ob_bear","bos_bull","bos_bear","choch_bull","choch_bear"
        ]
        data_yaml = write_data_yaml_from_lists(str(train_list), str(val_list), classes, str(root / 'data_lists.yaml'))
        train_yolo(data_yaml, model_pref=model_pref, imgsz=896, epochs=args.epochs)
    elif args.cmd == 'feats':
        # Placeholder: integrate with features/build_features.py when available
        print('Feature builder hook: --workspace', args.workspace)
    elif args.cmd == 'ab':
        print('Stage A/B training hook: --workspace', args.workspace, 'embargo', args.embargo)
    elif args.cmd == 'bt':
        print('Backtest hook: --workspace', args.workspace, 'fees_bps', args.fees_bps, 'slip_bps', args.slip_bps)


if __name__ == '__main__':
    main()


