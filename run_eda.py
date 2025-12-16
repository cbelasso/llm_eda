from pathlib import Path

from llm_parallelization.new_processor import NEMO, NewProcessor
import yaml

from eda_pipeline import EDAPipeline, load_dataframe

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


def main(config_path: str = "config.yaml"):
    # Load config
    print(f"📋 Loading config from: {config_path}")
    config = load_config(config_path)

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

    quality_config = config.get("quality", {})
    min_similarity = quality_config.get("min_semantic_similarity", 0.65)
    enable_validation = quality_config.get("enable_validation", False)

    # Initialize processor
    processor = NewProcessor(**config["processor"])

    try:
        # Initialize pipeline
        pipeline = EDAPipeline(
            processor=processor,
            reference_df=reference_df,
            reference_text_col=config["reference"]["text_column"],
            min_semantic_similarity=min_similarity,  # Adjust threshold
            enable_validation=enable_validation,  # Enable quality checks
        )

        # Run pipeline
        output_df = pipeline.run_pipeline(
            input_df=input_df,
            config=config,
            text_column=config["input"]["text_column"],
        )

        # Save final output
        output_path = Path(config["output"]["output_folder"]) / "final_augmented_dataset.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_path, index=False)

        print(f"\n🎉 Final dataset saved to: {output_path}")
        print(f"📊 Total texts: {len(output_df)}")
        print(f"📈 Augmentation multiplier: {len(output_df) / len(input_df):.1f}x")

        # Print round statistics
        print("\n📊 Round Statistics:")
        for round_name, texts in pipeline.all_texts.items():
            print(f"  {round_name}: {len(texts)} texts")

    finally:
        processor.terminate()


if __name__ == "__main__":
    import sys

    # Allow specifying config path as command line argument
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(config_path)
