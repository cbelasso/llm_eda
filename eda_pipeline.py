from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional

from llm_parallelization.new_processor import NewProcessor
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from prompts import AUGMENTATION_REGISTRY, get_augmentation


# Schema
class AugmentedText(BaseModel):
    rewritten: str = Field(description="The augmented text")


class EDAPipeline:
    def __init__(
        self,
        processor: NewProcessor,
        reference_df: Optional[pd.DataFrame] = None,
        reference_text_col: str = "text",
        min_semantic_similarity: float = 0.75,  # Add threshold
        enable_validation: bool = False,  # Add flag
        similarity_model_name: str = "all-MiniLM-L6-v2",  # Make configurable
    ):
        self.processor = processor
        self.reference_df = reference_df
        self.reference_text_col = reference_text_col
        self.all_texts = {}  # {round_name: List[dict]}
        self.min_semantic_similarity = min_semantic_similarity
        self.enable_validation = enable_validation

        # Load semantic similarity model if validation enabled
        if self.enable_validation:
            print(f"📊 Loading semantic similarity model: {similarity_model_name}...")
            self.similarity_model = SentenceTransformer(similarity_model_name)
            print("✅ Similarity model loaded")
        else:
            self.similarity_model = None

    def _validate_semantic_similarity(
        self,
        original_text: str,
        augmented_text: str,
        source_id: int = None,
        debug: bool = False,
    ) -> bool:
        """Check if augmented text is semantically similar to original."""
        if not self.enable_validation or self.similarity_model is None:
            return True

        embeddings = self.similarity_model.encode([original_text, augmented_text])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )

        is_valid = similarity >= self.min_semantic_similarity

        if debug and not is_valid:
            print(f"  Similarity: {similarity:.3f} (threshold: {self.min_semantic_similarity})")
            print(f"  Original: {original_text[:80]}...")
            print(f"  Augmented: {augmented_text[:80]}...")

        return is_valid

    def build_prompt(self, aug_type: str, text: str, params: Dict[str, Any]) -> str:
        """Build prompt using the augmentation registry."""
        augmentation = get_augmentation(aug_type)

        # Handle reference requirement
        if augmentation.requires_reference:
            if "reference" not in params:
                params["reference"] = self._sample_reference(text)

        return augmentation.build_prompt(text, params)

    def _sample_reference(self, text: str, used_refs: Optional[set] = None) -> str:
        """Sample a reference text for style rewriting."""
        if self.reference_df is None or self.reference_df.empty:
            raise ValueError(
                "Reference dataframe not provided for reference-based augmentation"
            )

        text_len = len(text)
        min_len = max(5, int(text_len * 0.7))
        max_len = int(text_len * 1.3)

        # Filter by length
        eligible = self.reference_df[
            self.reference_df[self.reference_text_col].str.len().between(min_len, max_len)
        ]

        if eligible.empty:
            eligible = self.reference_df

        # Sample
        ref_row = eligible.sample(n=1)
        return ref_row[self.reference_text_col].values[0]

    def apply_round(
        self,
        round_config: Dict[str, Any],
        round_name: str,
        batch_size: int = 25,
    ) -> List[Dict[str, Any]]:
        """Apply a single round of augmentations."""

        print(f"\n🔄 Starting {round_name}...")

        # Get texts to apply to
        apply_to = round_config.get("apply_to", ["original"])
        source_texts = []
        for source_round in apply_to:
            if source_round in self.all_texts:
                source_texts.extend(self.all_texts[source_round])

        if not source_texts:
            print(f"⚠️  No source texts found for {round_name}, skipping.")
            return []

        # Build prompts for all operations
        all_prompts = []
        operations = round_config.get("operations", [])

        for op in operations:
            op_type = op["type"]

            if op_type == "compound":
                prompts = self._build_compound_prompts(source_texts, op, round_name)
            else:
                prompts = self._build_single_prompts(source_texts, op, round_name)

            all_prompts.extend(prompts)

        if not all_prompts:
            return []

        # Process with LLM
        print(f"📝 Processing {len(all_prompts)} prompts...")
        prompt_texts = [p["prompt"] for p in all_prompts]

        self.processor.process_with_schema(
            prompts=prompt_texts, schema=AugmentedText, batch_size=batch_size
        )
        results: List[AugmentedText] = self.processor.parse_results_with_schema(
            schema=AugmentedText
        )

        # Merge results with validation
        output_texts = []
        failed_count = 0

        for prompt_data, result in zip(all_prompts, results):
            if result is None or not result.rewritten:
                failed_count += 1
                continue

            augmented_text = result.rewritten.strip()

            # Get original text for validation
            source_id = prompt_data["source_id"]
            original_text = self.all_texts["original"][source_id]["text"]

            # Validate semantic similarity
            if self.enable_validation:
                is_valid = self._validate_semantic_similarity(original_text, augmented_text)
                if not is_valid:
                    print(
                        f"❌ Failed validation (source_id={source_id}): {augmented_text[:100]}..."
                    )
                    failed_count += 1
                    continue

            prompt_data["text"] = augmented_text
            prompt_data.pop("prompt", None)
            output_texts.append(prompt_data)

        print(f"✅ Generated {len(output_texts)} texts in {round_name}")
        if failed_count > 0:
            print(f"⚠️  {failed_count} texts failed validation or parsing")

        return output_texts

    def _build_single_prompts(
        self,
        source_texts: List[Dict[str, Any]],
        operation: Dict[str, Any],
        round_name: str,
    ) -> List[Dict[str, Any]]:
        """Build prompts for a single operation."""

        op_type = operation["type"]
        count = operation.get("count", 1)
        params = operation.get("params", {})

        prompts = []
        for source in source_texts:
            for i in range(count):
                # Build operation-specific params
                op_params = params.copy()

                if op_type == "style_rewrite" and "reference_pool" in params:
                    # Use specific round as reference pool
                    ref_pool = self.all_texts.get(params["reference_pool"], [])
                    if ref_pool:
                        op_params["reference"] = random.choice(ref_pool)["text"]

                prompt_text = self.build_prompt(op_type, source["text"], op_params)

                # FIX: Exclude fields that will be overridden
                prompt_data = {
                    **{
                        k: v
                        for k, v in source.items()
                        if k
                        not in [
                            "text",
                            "prompt",
                            "text_id",
                            "round",
                            "augmentation_chain",
                            "depth",
                            "parent_text_id",
                        ]
                    },
                    "prompt": prompt_text,
                    "source_id": source["source_id"],
                    "parent_text_id": source.get("text_id"),
                    "round": round_name,  # This now won't be overridden
                    "augmentation_chain": source.get("augmentation_chain", []) + [op_type],
                    "depth": source.get("depth", 0) + 1,
                }
                prompts.append(prompt_data)

        return prompts

    def _build_compound_prompts(
        self,
        source_texts: List[Dict[str, Any]],
        operation: Dict[str, Any],
        round_name: str,
    ) -> List[Dict[str, Any]]:
        """Build prompts for compound operations (chains)."""
        sequences = operation.get("sequences", [])
        count = operation.get("count", 1)

        prompts = []
        for source in source_texts:
            for seq_idx, sequence in enumerate(sequences):
                for i in range(count):
                    # Build compound prompt
                    compound_prompt = self._build_compound_prompt_text(source["text"], sequence)

                    # FIX: Exclude fields that will be overridden
                    prompt_data = {
                        **{
                            k: v
                            for k, v in source.items()
                            if k
                            not in [
                                "text",
                                "prompt",
                                "text_id",
                                "round",
                                "augmentation_chain",
                                "depth",
                                "parent_text_id",
                            ]
                        },
                        "prompt": compound_prompt,
                        "source_id": source["source_id"],
                        "parent_text_id": source.get("text_id"),
                        "round": round_name,  # This now won't be overridden
                        "augmentation_chain": source.get("augmentation_chain", []) + sequence,
                        "depth": source.get("depth", 0) + len(sequence),
                    }
                    prompts.append(prompt_data)

        return prompts

    def _build_compound_prompt_text(self, text: str, sequence: List[str]) -> str:
        """Build a compound prompt that chains multiple operations."""
        operations_desc = ", then ".join(sequence)

        prompt = f"""Apply the following transformations in sequence to the text:
{operations_desc}

CRITICAL: Preserve exact semantic content and severity throughout all transformations.

Input: {text}

Respond in JSON: {{"rewritten": "..."}}"""
        return prompt

    def run_pipeline(
        self,
        input_df: pd.DataFrame,
        config: Dict[str, Any],
        text_column: str = "text",
    ) -> pd.DataFrame:
        """Run the full augmentation pipeline."""
        print("🚀 Starting EDA Pipeline...")

        # Get batch size from config
        batch_size = config.get("pipeline", {}).get("batch_size", 25)

        # Initialize with original texts
        original_texts = []
        for idx, row in input_df.iterrows():
            original_texts.append(
                {
                    "text_id": idx,
                    "source_id": idx,
                    "text": row[text_column],
                    "round": "original",
                    "augmentation_chain": [],
                    "depth": 0,
                    **{k: v for k, v in row.items() if k != text_column},
                }
            )

        self.all_texts["original"] = original_texts

        # Run each round
        rounds = config.get("rounds", [])
        for round_config in rounds:
            round_name = round_config["name"]
            round_outputs = self.apply_round(round_config, round_name, batch_size=batch_size)

            # Assign unique text IDs
            start_id = (
                max([t["text_id"] for texts in self.all_texts.values() for t in texts]) + 1
            )
            for i, text in enumerate(round_outputs):
                text["text_id"] = start_id + i

            self.all_texts[round_name] = round_outputs

            # Save intermediate if requested
            if config.get("save_intermediate", False):
                self._save_intermediate(round_name, config.get("output_folder", "./output"))

        # Combine all texts
        all_output = []
        for round_name, texts in self.all_texts.items():
            all_output.extend(texts)

        output_df = pd.DataFrame(all_output)
        print(f"\n✅ Pipeline complete! Generated {len(output_df)} total texts.")

        return output_df

    def _save_intermediate(self, round_name: str, output_folder: str):
        """Save intermediate round results."""
        output_path = Path(output_folder) / f"{round_name}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(self.all_texts[round_name])
        df.to_csv(output_path, index=False)
        print(f"💾 Saved {round_name} to {output_path}")


def load_dataframe(file_path: str) -> pd.DataFrame:
    """Load dataframe from various formats."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(file_path)
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)
    elif ext == ".jsonl":
        return pd.read_json(file_path, lines=True)
    elif ext in [".pkl", ".pickle"]:
        return pd.read_pickle(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
