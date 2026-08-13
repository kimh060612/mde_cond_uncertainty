import argparse
from pathlib import Path

from utils.select_canonical import (
    DA3_SMALL_MODEL_ID,
    SelectCanonicalandMatchFrames,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rebuild canonical CSV files with relative DA3 or a metric-depth checkpoint."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Existing canonical CSV file or directory of CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory for generated CSV files; existing files are never overwritten.",
    )
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument(
        "--checkpoint",
        "--checkpoint-path",
        dest="checkpoint",
        type=Path,
        help="Fine-tuned metric-depth checkpoint file or save_pretrained directory.",
    )
    parser.add_argument(
        "--model-version",
        choices=("dav2", "dav3"),
        help="Checkpoint architecture; required with --checkpoint.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Base model id for a state-dict checkpoint; defaults to the Small architecture.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument(
        "--process-res-method",
        default="upper_bound_resize",
        choices=(
            "upper_bound_resize",
            "upper_bound_crop",
            "lower_bound_resize",
            "lower_bound_crop",
        ),
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = args.input_csv.resolve()
    input_csv_paths = (
        sorted(input_path.glob("*.csv"))
        if input_path.is_dir()
        else [input_path]
    )
    selector = SelectCanonicalandMatchFrames(
        data_path=str(args.dataset_root.resolve()) if args.dataset_root else None
    )
    common = dict(
        input_csv_paths=input_csv_paths,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        process_res=args.process_res,
        process_res_method=args.process_res_method,
    )
    if args.checkpoint is not None:
        if args.model_version is None:
            raise ValueError("--model-version is required with --checkpoint.")
        summaries = selector.migrate_csvs_to_metric_checkpoint(
            **common,
            checkpoint_path=args.checkpoint,
            model_version=args.model_version,
            model_id=args.model_id,
        )
    else:
        if args.model_version is not None:
            raise ValueError("--model-version requires --checkpoint.")
        summaries = selector.migrate_csvs_to_da3(
            **common,
            model_id=args.model_id or DA3_SMALL_MODEL_ID,
        )
    for summary in summaries:
        print(
            f"Saved {summary['rows']} rows to {summary['output_csv']} "
            f"(reused={summary['reused']}, registered={summary['registered']}, "
            f"self_oracle={summary['self_oracle']}, failed={summary['failed']})"
        )


if __name__ == "__main__":
    main()
