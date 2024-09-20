# MMSearch 🔥🔍: Benchmarking the Potential of Large Models as Multi-modal Search Engines

![MultimodalSearch](https://img.shields.io/badge/Task-Multimodal_Search-red) 
![Multimodal AI Search Engine](https://img.shields.io/badge/Task-Multimodal_AI_Search_Engine-red) 
![Multi-Modal](https://img.shields.io/badge/Task-Multi--Modal-red) 

![GPT-4o](https://img.shields.io/badge/Model-GPT--4o-green) 
![GPT-4V](https://img.shields.io/badge/Model-GPT--4V-green)
![Claude-3.5](https://img.shields.io/badge/Model-Claude--3.5-green)

Official repository for "[MMSearch: Benchmarking the Potential of Large Models as Multi-modal Search Engines](https://arxiv.org/pdf/2409.12959)".

🌟 For more details, please refer to the project page with dataset exploration and visualization tools.

[[🌐 Webpage](https://mmsearch.github.io/)] [[📖 Paper](https://arxiv.org/pdf/2409.12959)] [[🤗 Huggingface Dataset](https://huggingface.co/datasets/CaraJ/MMSearch)] [[🏆 Leaderboard](https://mmsearch.github.io/#leaderboard)] [[🔍 Visualization](https://huggingface.co/datasets/CaraJ/MMSearch/viewer)]


## 💥 News

- **[2024.09.20]** 🚀 We release the [arXiv paper](https://arxiv.org/abs/2409.12959) and some data samples in the [visualizer](https://huggingface.co/datasets/CaraJ/MMSearch/viewer).

## 📌 ToDo

- Coming soon: *MMSearch-Engine* and *Evaluation codes*

## 👀 About MMSearch

The capabilities of **Large Multi-modal Models (LMMs)** in **multimodal search** remain insufficiently explored and evaluated. To fill the blank of a framework for LMM to conduct multimodal AI search engine, we first design a delicate pipeline **MMSearch-Engine** to facilitate **any LMM** to function as a multimodal AI search engine

<p align="center">
    <img src="figs/fig1.png" width="90%"> <br>
</p>

To further evaluate the potential of LMMs in the multimodal search domain, we introduce **MMSearch**, an all-around multimodal search benchmark designed for assessing the multimodal search performance. The benchmark contains 300 manually collected instances spanning 14 subfields, which involves no overlap with the current LMMs' training data, ensuring the correct answer can only be obtained within searching.

<p align="center">
    <img src="figs/fig2.png" width="90%"> <br>
    An overview of <b>MMSearch</b>.
</p>

In addition, we propose a **step-wise evaluation strategy** to better understand the LMMs' searching capability. The models are evaluated by performing **three individual tasks (requery, rerank, and summarization)**, and **one challenging end-to-end task** with a complete searching process. The final score is weighted by the four tasks.

<p align="center">
    <img src="figs/fig3.png" width="90%"> <br>
    Outline of Evaluation Tasks, Inputs, and Outputs.
</p>

<details>
<summary>🔍 An example of LMM input, output, and ground truth for four evaluation tasks</summary>

<p align="center">
    <img src="figs/fig4.png" width="50%"> <br>
</p>
</details>

## 🏆 Leaderboard

### Contributing to the Leaderboard

🚨 The [Leaderboard](https://mmsearch.github.io/#leaderboard) is continuously being updated, welcoming the contribution of your excellent LMMs!

### Data Usage

We release the MMSearch data for benchmarking on the leaderboard, which contains *300* queries and the middle results for step-wise evaluation.

You can download the dataset from the [🤗 Huggingface](https://huggingface.co/datasets/CaraJ/MMSearch) by the following command (make sure that you have installed [related packages](https://huggingface.co/docs/datasets/quickstart)):

```python
from datasets import load_dataset

dataset = load_dataset("CaraJ/MMSearch")
```


## :white_check_mark: Citation

If you find **MMSearch** useful for your research and applications, please kindly cite using this BibTeX:

```latex
@article{jiang2024mmsearch,
  title={MMSearch: Benchmarking the Potential of Large Models as Multi-modal Search Engines},
  author={Dongzhi Jiang, Renrui Zhang, Ziyu Guo, Yanmin Wu, Jiayi Lei, Pengshuo Qiu, Pan Lu, Zehui Chen, Guanglu Song, Peng Gao, Yu Liu, Chunyuan Li, Hongsheng Li},
  booktitle={arXiv},
  year={2024}
}
```

## 🧠 Related Work

Explore our additional research on **Vision-Language Large Models**:

- **[MathVerse]** [MathVerse: Does Your Multi-modal LLM Truly See the Diagrams in Visual Math Problems?](https://mathverse-cuhk.github.io/)
- **[MathVista]** [MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts](https://github.com/lupantech/MathVista)
- **[LLaMA-Adapter]** [LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention](https://github.com/OpenGVLab/LLaMA-Adapter)
- **[LLaMA-Adapter V2]** [LLaMA-Adapter V2: Parameter-Efficient Visual Instruction Model](https://github.com/OpenGVLab/LLaMA-Adapter)
- **[ImageBind-LLM]** [Imagebind-LLM: Multi-modality Instruction Tuning](https://github.com/OpenGVLab/LLaMA-Adapter/tree/main/imagebind_LLM)
- **[SPHINX]** [The Joint Mixing of Weights, Tasks, and Visual Embeddings for Multi-modal LLMs](https://github.com/Alpha-VLLM/LLaMA2-Accessory/tree/main/SPHINX)
- **[SPHINX-X]** [Scaling Data and Parameters for a Family of Multi-modal Large Language Models](https://github.com/Alpha-VLLM/LLaMA2-Accessory/tree/main/SPHINX)
- **[Point-Bind & Point-LLM]** [Multi-modality 3D Understanding, Generation, and Instruction Following](https://github.com/ZiyuGuo99/Point-Bind_Point-LLM)
- **[PerSAM]** [Personalize segment anything model with one shot](https://github.com/ZrrSkywalker/Personalize-SAM)
- **[CoMat]** [CoMat: Aligning Text-to-Image Diffusion Model with Image-to-Text Concept Matching](https://caraj7.github.io/comat/)
