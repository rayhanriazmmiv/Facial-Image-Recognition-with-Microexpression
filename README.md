# 🧠 HTNet (Modified Version with Enhanced Attention) for Micro-Expression Recognition

> ✅ This is a modified and extended implementation of the official [HTNet](https://arxiv.org/abs/2307.14637) model designed for micro-expression recognition.

This version focuses on **restructuring transformer blocks**, **changing architectural layers**, and **adding custom attention mechanisms** to improve recognition of subtle facial microexpressions. It retains the hierarchical feature extraction design from the original HTNet while introducing architectural enhancements that improve sensitivity to fine-grained facial movements.

---

## 🔄 How This Version Differs from the Original HTNet

| Feature | Original HTNet (Paper) | Modified Version |
|--------|--------------------------|------------------------|
| **Transformer Blocks** | Standard multi-head self-attention | Enhanced attention mechanism (customized) |
| **Feature Aggregation** | Basic concatenation | Re-weighted or layered fusion approach |
| **Layer Modifications** | Original positional encoders and FFN | Adjusted layer sizes, activations, or structure |
| **Classifier** | Softmax classifier (unchanged) | Same classifier, but fed improved attention features |
| **Focus** | Global + Local attention | **Finer regional attention control** with custom modules |

---

## 🧰 Architecture Diagram (Modified HTNet)

![HTNet Modified Architecture](./images/A_flowchart_diagram_of_a_hierarchical_transformer_.png)

- Preprocessing → Facial Landmark Detection  
- Facial Region Segmentation → Transformer Feature Extraction  
- Attention-enhanced global feature aggregation  
- Final Classification (softmax)

---

## 📂 Dataset Usage

This project supports the following microexpression datasets:
- **CASME II**
- **SAMM**
- **CASME3**

Dataset folder structure:

Reference to the original paper : 
@article{wang2024htnet,
  title={HTNet for Micro-Expression Recognition},
  author={Wang, Zhifeng and Zhang, Kaihao and Luo, Wenhan and Sankaranarayana, Ramesh},
  journal={Neurocomputing},
  volume={602},
  pages={128196},
  year={2024},
  publisher={Elsevier}
}
