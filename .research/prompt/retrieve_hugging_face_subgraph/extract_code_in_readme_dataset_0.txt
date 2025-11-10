
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
license: mit
tags:
- Imitation Learning
- Expert Trajectory
pretty_name: CartPole-v1 Expert Dataset
size_categories:
- 10M<n<100M
---

# CartPole-v1 - Imitation Learning Datasets

This is a dataset created by [Imitation Learning Datasets](https://github.com/NathanGavenski/IL-Datasets) project. 
It was created by using Stable Baselines weights from a PPO policy from [HuggingFace](https://huggingface.co/sb3/ppo-CartPole-v1).

## Description

The dataset consists of 1,000 episodes with an average episodic reward of 500.
Each entry consists of:
```
obs (list): observation with length 4.
action (int): action (0 or 1).
reward (float): reward point for that timestep.
episode_returns (bool): if that state was the initial timestep for an episode.
```

## Usage

Feel free to download and use the `teacher.jsonl` dataset as you please.
If you are interested in using our PyTorch Dataset implementation, feel free to check the [IL Datasets](https://github.com/NathanGavenski/IL-Datasets/blob/main/src/imitation_datasets/dataset/dataset.py) project.
There, we implement a base Dataset that downloads this dataset and all other datasets directly from HuggingFace.
The Baseline Dataset also allows for more control over train and test splits and how many episodes you want to use (in cases where the 1k episodes are not necessary).

## Citation

```{bibtex}
@inproceedings{gavenski2024ildatasets,
  author = {Gavenski, Nathan and Luck, Michael and Rodrigues, Odinaldo},
  title = {Imitation Learning Datasets: A Toolkit For Creating Datasets, Training Agents and Benchmarking},
  year = {2024},
  isbn = {9798400704864},
  publisher = {International Foundation for Autonomous Agents and Multiagent Systems},
  address = {Richland, SC},
  abstract = {Imitation learning field requires expert data to train agents in a task. Most often, this learning approach suffers from the absence of available data, which results in techniques being tested on its dataset. Creating datasets is a cumbersome process requiring researchers to train expert agents from scratch, record their interactions and test each benchmark method with newly created data. Moreover, creating new datasets for each new technique results in a lack of consistency in the evaluation process since each dataset can drastically vary in state and action distribution. In response, this work aims to address these issues by creating Imitation Learning Datasets, a toolkit that allows for: (i) curated expert policies with multithreaded support for faster dataset creation; (ii) readily available datasets and techniques with precise measurements; and (iii) sharing implementations of common imitation learning techniques. Demonstration link: https://nathangavenski.github.io/#/il-datasets-video},
  booktitle = {Proceedings of the 23rd International Conference on Autonomous Agents and Multiagent Systems},
  pages = {2800–2802},
  numpages = {3},
  keywords = {benchmarking, dataset, imitation learning},
  location = {<conf-loc>, <city>Auckland</city>, <country>New Zealand</country>, </conf-loc>},
  series = {AAMAS '24}
}
```
Output:
{
    "extracted_code": ""
}
