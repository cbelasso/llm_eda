from collections import defaultdict
from datetime import datetime
import json
import multiprocessing
from multiprocessing import Event, Queue, get_context
import os
from queue import Empty
import time
from typing import Dict, Generator, List, Optional, Tuple, Type, Union
import uuid

from lmformatenforcer import CharacterLevelParser
from lmformatenforcer.integrations.vllm import (
    build_vllm_logits_processor,
    build_vllm_token_enforcer_tokenizer_data,
)
from pydantic import BaseModel
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, RequestOutput, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

multiprocessing.set_start_method("spawn", force=True)

NEMO = "casperhansen/mistral-nemo-instruct-2407-awq"


class NewProcessor:
    def __init__(
        self,
        gpu_list: list[int] = None,
        llm: str = "meta-llama/Llama-3.2-3B-Instruct",
        multiplicity: int = 1,
        use_tqdm: bool = False,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 2048,
        default_guided_config: Optional[Dict] = None,
        tokenizer: str | None = None,
        tokenizer_mode: str = "auto",
        config_format: str | None = None,
        load_format: str | None = None,
        tensor_parallel_size: int = 1,
        skip_tokenizer_init: bool = False,
        _worker_mode: bool = False,
        **extra_llm_args,
    ):
        self.llm = llm
        self.default_guided_config = default_guided_config or {}
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self.skip_tokenizer_init = skip_tokenizer_init

        self.llm_kwargs = {
            "model_path": self.llm,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "tokenizer_mode": tokenizer_mode,
            "tensor_parallel_size": tensor_parallel_size,
            **extra_llm_args,
        }

        if tokenizer is not None:
            self.llm_kwargs["tokenizer"] = tokenizer
        if config_format is not None:
            self.llm_kwargs["config_format"] = config_format
        if load_format is not None:
            self.llm_kwargs["load_format"] = load_format

        if _worker_mode:
            self.model = self._load_llm(**self.llm_kwargs)
            self.base_sampling_params = self._get_sampling_params()

        else:
            self.gpu_list = gpu_list
            self.multiplicity = multiplicity
            self.use_tqdm = use_tqdm
            self.num_gpus = len(gpu_list)

            self.tokenizer = None
            if not skip_tokenizer_init:
                try:
                    tokenizer_path = tokenizer or self.llm
                    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                    print(f"✓ External tokenizer loaded: {tokenizer_path}")
                except Exception as e:
                    print(f"⚠ Could not load external tokenizer: {e}")
                    print("  Will rely on vLLM's internal tokenizer for chat formatting")

            self.task_queue: Queue = Queue()
            self.response_queue: Queue = Queue()
            self.load_signal_queue: Queue = Queue()
            self.processes = []
            self.stop_event: Event = Event()
            self.responses = []

            self._prepare_processes()

            print("🔄 NewProcessor initialized - ready for runtime schema switching")

    def _load_llm(self, **llm_kwargs) -> LLM:
        init_kwargs = {
            "model": llm_kwargs["model_path"],
            "trust_remote_code": True,
            "enforce_eager": llm_kwargs.get("enforce_eager", True),
            "gpu_memory_utilization": llm_kwargs.get("gpu_memory_utilization", 0.9),
            "max_model_len": llm_kwargs.get("max_model_len", None),
            "dtype": llm_kwargs.get("dtype", "auto"),
        }

        handled_params = {
            "model_path",
            "gpu_num",
            "enforce_eager",
            "gpu_memory_utilization",
            "max_model_len",
            "dtype",
        }

        for key, value in llm_kwargs.items():
            if key not in handled_params and value is not None:
                init_kwargs[key] = value

        return LLM(**init_kwargs)

    def _get_sampling_params(self) -> SamplingParams:
        return SamplingParams(
            temperature=0.95,
            top_k=50,
            top_p=0.95,
            max_tokens=4098,
            frequency_penalty=2,
            repetition_penalty=1.1,
        )

    def _create_guided_sampling_params(
        self, json_schema: Optional[Dict] = None, guided_config: Optional[Dict] = None
    ) -> SamplingParams:
        config = {**self.default_guided_config, **(guided_config or {})}

        sampling_params = SamplingParams(
            temperature=config.get("temperature", 0.1),
            top_k=config.get("top_k", 50),
            top_p=config.get("top_p", 0.95),
            max_tokens=config.get("max_tokens", 1000),
            frequency_penalty=config.get("frequency_penalty", 0.0),
            repetition_penalty=config.get("repetition_penalty", 1.0),
        )

        if json_schema:
            guided_decoding_params = GuidedDecodingParams(json=json_schema)
            sampling_params.guided_decoding = guided_decoding_params

        return sampling_params

    def _generate_with_json_schema(
        self,
        prompts: Union[str, List[str]],
        json_schema: Optional[Dict] = None,
        guided_config: Optional[Dict] = None,
        use_tqdm: bool = True,
    ) -> List[RequestOutput]:
        prompt_list = [prompts] if isinstance(prompts, str) else prompts
        sampling_params = self._create_guided_sampling_params(json_schema, guided_config)
        return self.model.generate(
            prompts=prompt_list,
            sampling_params=sampling_params,
            use_tqdm=use_tqdm,
        )

    def _generate(
        self, prompts: Union[str, List[str]], use_tqdm: bool = True
    ) -> List[RequestOutput]:
        prompt_list = [prompts] if isinstance(prompts, str) else prompts
        return self.model.generate(
            prompts=prompt_list,
            sampling_params=self.base_sampling_params,
            use_tqdm=use_tqdm,
        )

    def format_prompt(self, prompt: str) -> str:
        """
        Format prompt using tokenizer chat template.

        If external tokenizer is not available, returns the prompt as-is.
        For models like Mistral, you should pre-format prompts before passing them in.
        """
        if self.tokenizer is None:
            # No external tokenizer - return prompt as-is
            return prompt

        try:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception as e:
            print(f"⚠ Chat template formatting failed: {e}")
            print("  Returning prompt as-is")
            return prompt

    def create_batches(self, prompts: list, batch_size: int) -> list:
        return [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]

    @staticmethod
    def worker(
        llm_kwargs,
        gpu_num,
        task_queue,
        response_queue,
        stop_event,
        load_signal_queue,
        multiplicity_index,
        default_guided_config=None,
    ):
        try:
            llm_kwargs["gpu_num"] = gpu_num
            model = NewProcessor(
                default_guided_config=default_guided_config,
                _worker_mode=True,
                **llm_kwargs,
            )
            load_signal_queue.put((gpu_num, multiplicity_index))
        except Exception as e:
            load_signal_queue.put(
                f"Error loading model on GPU {gpu_num} (multiplicity {multiplicity_index}): {str(e)}"
            )
            return

        while not stop_event.is_set():
            request_id = None
            corr_id = None
            request = None

            try:
                task_data = task_queue.get(timeout=1)

                if len(task_data) == 3:
                    request_id, request, corr_id = task_data
                    if request_id == -1:
                        break
                    response = model._generate(prompts=request, use_tqdm=False)
                else:
                    request_id, request, corr_id, json_schema, guided_config = task_data

                    response = model._generate_with_json_schema(
                        prompts=request,
                        json_schema=json_schema,
                        guided_config=guided_config,
                        use_tqdm=False,
                    )

                response_queue.put((request_id, response, corr_id, request))
            except Empty:
                continue
            except Exception as e:
                print(f"Worker error: {e}")
                if request_id is not None and corr_id is not None and request is not None:
                    response_queue.put((request_id, None, corr_id, request))

    def _prepare_processes(self) -> None:
        ctx = get_context("spawn")

        if self.tensor_parallel_size > 1:
            gpu_chunks = [
                self.gpu_list[i : i + self.tensor_parallel_size]
                for i in range(0, len(self.gpu_list), self.tensor_parallel_size)
            ]

            print(
                f"🔗 Tensor parallelism: Creating {len(gpu_chunks)} model(s) "
                f"with {self.tensor_parallel_size} GPUs each"
            )

            for i in range(self.multiplicity):
                # TODO double check the multiplying factor for each multiplicity
                gpu_memory_utilization = min(self.gpu_memory_utilization + i * 0.2, 1.0)
                print(
                    f"Starting multiplicity round {i + 1} with GPU memory utilization: {gpu_memory_utilization}"
                )

                for chunk_idx, gpu_chunk in enumerate(gpu_chunks):
                    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_chunk))
                    process = ctx.Process(
                        target=NewProcessor.worker,
                        args=(
                            self.llm_kwargs,
                            gpu_chunk[0],
                            self.task_queue,
                            self.response_queue,
                            self.stop_event,
                            self.load_signal_queue,
                            i,
                            self.default_guided_config,
                        ),
                    )
                    process.start()
                    self.processes.append(process)

                self._wait_for_models_to_load(expected_count=len(gpu_chunks))
        else:
            for i in range(self.multiplicity):
                # TODO double check the multiplying factor for each multiplicity
                gpu_memory_utilization = min(self.gpu_memory_utilization + i * 0.2, 1.0)
                print(
                    f"Starting multiplicity round {i + 1} with GPU memory utilization: {gpu_memory_utilization}"
                )

                for gpu_num in self.gpu_list:
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
                    process = ctx.Process(
                        target=NewProcessor.worker,
                        args=(
                            self.llm_kwargs,
                            gpu_num,
                            self.task_queue,
                            self.response_queue,
                            self.stop_event,
                            self.load_signal_queue,
                            i,
                            self.default_guided_config,
                        ),
                    )
                    process.start()
                    self.processes.append(process)

                self._wait_for_models_to_load(expected_count=self.num_gpus)

    def _wait_for_models_to_load(self, expected_count, timeout=None):
        start_time = time.time()
        loaded_models = set()
        errors = []

        while len(loaded_models) < expected_count:
            try:
                result = self.load_signal_queue.get(timeout=1)
                if isinstance(result, tuple):
                    loaded_models.add(result)
                    print(f"Model loaded on GPU {result[0]} (multiplicity {result[1]})")
                else:
                    errors.append(result)
                    print(result)
            except Empty:
                pass

            if timeout is not None and time.time() - start_time > timeout:
                print("Timeout waiting for models to load")
                return False

            if len(errors) + len(loaded_models) == expected_count:
                break

        if errors:
            print("Some models failed to load")
            return False

        print(f"All {expected_count} models in this round loaded successfully")
        return True

    def process_with_schema(
        self,
        prompts: Union[str, List[str]],
        schema: Optional[Type[BaseModel]] = None,
        batch_size: int = 25,
        formatted: bool = False,
        guided_config: Optional[Dict] = None,
        on_batch_end=None,
        timeout=10,
    ) -> List[RequestOutput]:
        prompt_list = [prompts] if isinstance(prompts, str) else prompts

        if formatted:
            formatted_prompt_list = prompt_list
        else:
            formatted_prompt_list = [
                self.format_prompt(prompt=prompt) for prompt in prompt_list
            ]

        batch_prompts = {
            request_id: batch
            for request_id, batch in enumerate(
                self.create_batches(prompts=formatted_prompt_list, batch_size=batch_size)
            )
        }
        book_keeping_indexes = {
            request_id: batch
            for request_id, batch in enumerate(
                self.create_batches(
                    prompts=list(range(0, len(prompt_list))), batch_size=batch_size
                )
            )
        }

        total_requests = len(batch_prompts)
        response_counter = 0
        current_corr_id = uuid.uuid4()

        json_schema = None
        if schema:
            json_schema = schema.model_json_schema()

        for request_id, prompts_ in batch_prompts.items():
            self.task_queue.put(
                (request_id, prompts_, current_corr_id, json_schema, guided_config)
            )

        processed_responses = {}

        with tqdm(
            total=total_requests,
            colour="#B48EAD",
            leave=False,
            desc=f"Process requests with schema {schema.__name__ if schema else 'None'} {current_corr_id}",
        ) as pbar:
            while response_counter < total_requests and not self.stop_event.is_set():
                try:
                    request_id, response, corr_id, prompts_ = self.response_queue.get(timeout=1)

                    if response is None:
                        print(f"Failed on request_id {request_id}")
                        self.task_queue.put(
                            (request_id, prompts_, current_corr_id, json_schema, guided_config)
                        )
                        continue

                    if current_corr_id != corr_id:
                        raise RuntimeError(
                            f"Current correlation id {current_corr_id} does not match result queue correlation id {corr_id}"
                        )

                    response_counter += 1

                    if on_batch_end:
                        on_batch_end(
                            batch_prompts[request_id],
                            book_keeping_indexes[request_id],
                            response,
                        )

                    processed_responses[request_id] = response
                    pbar.update(1)

                except Empty:
                    continue
                except Exception as e:
                    print(f"Processing error: {e}")

        self.responses = [
            processed_responses[request_id] for request_id in sorted(processed_responses.keys())
        ]

        return self.responses

    def parse_results_with_schema(
        self,
        schema: Type[BaseModel],
        responses: Optional[List[RequestOutput]] = None,
        validate: bool = True,
    ) -> List[Union[BaseModel, Dict, str, None]]:
        responses_to_parse = responses or self.responses
        parsed_results = []

        for response in tqdm(
            responses_to_parse, desc=f"Parsing with {schema.__name__ if schema else 'None'}"
        ):
            all_texts = self.extract_all_batch_outputs(response)

            for text_output in all_texts:
                try:
                    text_output = text_output.strip()
                    if text_output.startswith("```json"):
                        text_output = (
                            text_output.replace("```json", "").replace("```", "").strip()
                        )

                    json_data = json.loads(text_output)

                    if validate:
                        validated_obj = schema(**json_data)
                        parsed_results.append(validated_obj)
                    else:
                        parsed_results.append(json_data)

                except Exception as e:
                    print(f"Failed to parse output: {text_output[:100]}...")
                    print(f"Error: {e}")
                    parsed_results.append(None)

        return parsed_results

    def extract_all_batch_outputs(self, response):
        all_texts = []

        if isinstance(response, list):
            for resp in response:
                if hasattr(resp, "outputs"):
                    for output in resp.outputs:
                        all_texts.append(output.text)
        else:
            if hasattr(response, "outputs"):
                for output in response.outputs:
                    all_texts.append(output.text)

        return all_texts

    def terminate(self):
        if hasattr(self, "stop_event") and self.stop_event is not None:
            self.stop_event.set()
            for _ in range(len(self.processes)):
                self.task_queue.put((-1, "TERMINATE", None))
            for process in self.processes:
                process.join(timeout=30)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
            self.processes.clear()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.stop_event = None
        self.task_queue = None
        self.response_queue = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.terminate()

    def __del__(self):
        try:
            self.terminate()
        except:
            pass
