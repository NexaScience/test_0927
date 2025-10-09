
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
dataset_info:
  features:
  - name: image
    dtype: image
  - name: prompt
    dtype: string
  - name: image_width
    dtype: int64
  - name: image_height
    dtype: int64
  - name: img_path
    dtype: string
  splits:
  - name: train
    num_bytes: 1520494577466.375
    num_examples: 2810669
  download_size: 1520284961545 (1.52 TB)
  dataset_size: 1520494577466.375 (1.52 TB)
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---

Output:
{
    "extracted_code": ""
}
