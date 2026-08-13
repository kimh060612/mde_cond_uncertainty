import argparse
from pathlib import Path

from utils.select_canonical import (
    DA3_SMALL_MODEL_ID,
    SelectCanonicalandMatchFrames,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rebuild Depth Anything v2 canonical CSV files for DA3-Small."
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
        help="New directory for DA3 CSV files; existing files are never overwritten.",
    )
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--model-id", default=DA3_SMALL_MODEL_ID)
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
    summaries = selector.migrate_csvs_to_da3(
        input_csv_paths=input_csv_paths,
        output_dir=args.output_dir,
        model_id=args.model_id,
        device=args.device,
        batch_size=args.batch_size,
        process_res=args.process_res,
        process_res_method=args.process_res_method,
    )
    for summary in summaries:
        print(
            f"Saved {summary['rows']} rows to {summary['output_csv']} "
            f"(reused={summary['reused']}, registered={summary['registered']}, "
            f"self_oracle={summary['self_oracle']}, failed={summary['failed']})"
        )


if __name__ == "__main__":
    main()
