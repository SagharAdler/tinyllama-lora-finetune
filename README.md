
# tinyllama-lora-finetune

This project fine-tunes the [TinyLlama-1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) model using [LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685) on the [Databricks Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) instruction dataset. It leverages the Hugging Face Transformers and Datasets libraries for model loading, tokenization, and dataset preprocessing. It compares the performance of the base and fine-tuned models using standard NLP evaluation metrics such as BLEU, ROUGE-L, and BERTScore across a range of task categories.

## Features

- LoRA-based fine-tuning on TinyLlama
- Evaluation on Dolly 15k's diverse instruction types
- Metrics: BLEU, ROUGE-L, BERTScore
- Category-wise performance breakdown
- Hugging Face Transformers & Datasets integration


## Training Environment

- Hardware accelerator: NVIDIA A100 PCIe GPU
- Model & dataset loading: Hugging Face Transformers and Datasets libraries
- Frameworks used: PyTorch

## Dataset

The [Databricks Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) dataset contains 15,000 instruction-following examples in categories such as:

- `open_qa`
- `creative_writing`
- `classification`
- `closed_qa`
- `information_extraction`
- `brainstorming`
- `summarization`

## Results Summary

| Metric       | Overall Trend |
|--------------|----------------|
| BLEU         | Modest gain in open-ended tasks |
| ROUGE-L      | Stable overall |
| BERTScore    | Stable overall |

LoRA fine-tuning helped with fluency and creativity but showed limited or no improvement in structured tasks like classification or extraction.

## Conclusion

This experiment reinforces that lightweight LLMs like TinyLlama can benefit from parameter-efficient tuning like LoRA for generative tasks. However, high-precision tasks may require more task-specific alignment or constraints.


