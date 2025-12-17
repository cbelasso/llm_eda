import argparse
from pathlib import Path

from llm_parallelization.new_processor import NEMO, NewProcessor
import yaml

from eda_pipeline_optimized import EDAPipelineOptimized, load_dataframe

# Model mapping
MODEL_MAPPING = {
    "NEMO": NEMO,
}


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Map model string to actual object
    if "processor" in config and "llm" in config["processor"]:
        model_name = config["processor"]["llm"]
        if model_name in MODEL_MAPPING:
            config["processor"]["llm"] = MODEL_MAPPING[model_name]
        else:
            raise ValueError(
                f"Unknown model: {model_name}. Available: {list(MODEL_MAPPING.keys())}"
            )

    return config


def main(config_path: str = "config.yaml", resume: bool = True, clear_checkpoint: bool = False):
    # Load config
    print(f"📋 Loading config from: {config_path}")
    config = load_config(config_path)

    # Setup checkpoint directory
    output_folder = config.get("output", {}).get("output_folder", "./output")
    checkpoint_dir = Path(output_folder) / "checkpoints"

    # Clear checkpoint if requested
    if clear_checkpoint and checkpoint_dir.exists():
        import shutil

        print(f"🧹 Clearing existing checkpoint at {checkpoint_dir}")
        shutil.rmtree(checkpoint_dir)

    # Load data
    print("📂 Loading input data...")
    input_df = load_dataframe(config["input"]["dataframe_path"])

    print("📂 Loading reference data...")
    reference_df = load_dataframe(config["reference"]["dataframe_path"])

    # Filter reference by length
    reference_df = reference_df[
        reference_df[config["reference"]["text_column"]]
        .str.len()
        .between(config["reference"]["min_length"], config["reference"]["max_length"])
    ].reset_index(drop=True)

    print(f"✅ Loaded {len(input_df)} input texts")
    print(f"✅ Loaded {len(reference_df)} reference texts")
    print(f"📋 Input columns: {list(input_df.columns)}")

    quality_config = config.get("quality", {})
    min_similarity = quality_config.get("min_semantic_similarity", 0.65)
    enable_validation = quality_config.get("enable_validation", False)

    # Initialize processor
    processor = NewProcessor(**config["processor"])

    try:
        # Initialize optimized pipeline with checkpointing
        pipeline = EDAPipelineOptimized(
            processor=processor,
            reference_df=reference_df,
            reference_text_col=config["reference"]["text_column"],
            min_semantic_similarity=min_similarity,
            enable_validation=enable_validation,
            checkpoint_dir=str(checkpoint_dir),
        )

        # Run pipeline
        output_df = pipeline.run_pipeline(
            input_df=input_df,
            config=config,
            text_column=config["input"]["text_column"],
            resume=resume,
        )

        # Save final output
        output_path = Path(output_folder) / "final_augmented_dataset.csv"
        if not output_path.exists() or len(output_df) > 0:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_df.to_csv(output_path, index=False)

        print(f"\n🎉 Final dataset saved to: {output_path}")
        print(f"📊 Total texts: {len(output_df)}")
        print(f"📈 Augmentation multiplier: {len(output_df) / len(input_df):.1f}x")
        print(f"📋 Output columns: {list(output_df.columns)}")

    finally:
        processor.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EDA augmentation pipeline")
    parser.add_argument(
        "config",
        nargs="?",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--no-resume", action="store_true", help="Start fresh, don't resume from checkpoint"
    )
    parser.add_argument(
        "--clear-checkpoint",
        action="store_true",
        help="Clear existing checkpoint before starting",
    )

    args = parser.parse_args()

    main(
        config_path=args.config,
        resume=not args.no_resume,
        clear_checkpoint=args.clear_checkpoint,
    )
