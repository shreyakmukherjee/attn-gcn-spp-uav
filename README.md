<h1 align="center"> 🌿 Soybean Leaf Disease Detection Using Graph Convolutional Networks (GCN) in PyTorch </h1>

This repository presents a pipeline for detecting diseases in soybean leaves using **Graph Convolutional Networks (GCN)**. It classifies leaves into four categories — **Healthy**, **Rust**, **Mosaic**, and **Pest** — using a relational graph-based approach implemented in PyTorch Geometric.

---

## 🧾 Project Overview
<div align="center">

| Attribute              | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| 📚 Framework           | PyTorch, PyTorch Geometric, Scikit-learn                                    |
| 🧠 Models Used         | Graph Convolutional Network (GCN), Relational GCN                           |
| 📷 Input Format        | Drone-captured Images (~256×256 px)                                         |
| 🎯 Output              | 4-Class Image Classification                                                 |
| 🧪 File                | `gcn-code.ipynb`                                                             |
| 📁 Dataset Source      | [Soybean Leaf Dataset (Kaggle)](https://data.mendeley.com/datasets/hkbgh5s3b7/1) |

</div>

---

## 🗂 Dataset Overview

This dataset includes drone-captured images of soybean plants from various growth stages, classified into four types based on the disease:

<h3 align="center">📊 Class Distribution</h3>

<table align="center">
  <tr>
    <td align="center"><img src="Sample_Input_Images/healthy.jpg" width="120px"></td>
    <td align="center"><img src="Sample_Input_Images/rust.jpg" width="120px"></td>
    <td align="center"><img src="Sample_Input_Images/mosaic.jpg" width="120px"></td>
    <td align="center"><img src="Sample_Input_Images/pest.jpg" width="120px"></td>
  </tr>
  <tr>
    <td align="center"><b>Healthy</b></td>
    <td align="center"><b>Rust</b></td>
    <td align="center"><b>Mosaic</b></td>
    <td align="center"><b>Pest</b></td>
  </tr>
</table>

- Total Images: ~1,000+  
- Format: `.jpg`  
- Structure: Class-wise folders  
- Preprocessing: Resizing, patch extraction, graph construction  

---

## 🧪 Notebook Workflow
<div align="center">
  
| Step                | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| 📥 Data Loading      | Dataset parsed and structured into graph format                             |
| 🧼 Preprocessing      | Image → Superpixels → Graph (nodes/edges)                                  |
| 🧠 Model              | R-GCN architecture using PyTorch Geometric                                  |
| 🔁 Training           | CrossEntropyLoss + Adam optimizer                                           |
| 📊 Evaluation         | Accuracy, Confusion Matrix, ROC-AUC, Graph Visuals                          |
| 🔍 Inference          | Predict class for new images using trained GCN                             |

</div>

---

<h2>⚙️ Model Configuration &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 📈 Performance Summary</h2>
<div align="center">
<table>
  <tr>
    <td>

<!-- Left Table -->
<b>⚙️ Model Configuration</b>

<table>
  <tr><th>Parameter</th><th>Value</th></tr>
  <tr><td>Graph Layers</td><td>R-GCN (2-layer)</td></tr>
  <tr><td>Input Features</td><td>16</td></tr>
  <tr><td>Hidden Features</td><td>32</td></tr>
  <tr><td>Output Classes</td><td>4</td></tr>
  <tr><td>Optimizer</td><td>Adam</td></tr>
  <tr><td>Loss Function</td><td>CrossEntropy</td></tr>
  <tr><td>Batch Size</td><td>32</td></tr>
</table>

  </td>
  <td style="width: 40px;"></td>
  <td>

<!-- Right Table -->
<b>📈 Performance Summary</b>

<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>✅ Final Accuracy</td><td><b>94.50%</b></td></tr>
  <tr><td>🔍 ROC-AUC</td><td><b>0.96 (macro avg)</b></td></tr>
  <tr><td>📉 Final Loss</td><td><b>0.217</b></td></tr>
  <tr><td>📊 Confusion Matrix</td><td>4×4</td></tr>
</table>

  </td>
  </tr>
</table>
</div>

---

## 📊 Model Outputs

<p align="center">
  <img src="Output_Images/Accuracy_Curve.png" height="220px" style="margin-right: 10px;">
  <img src="Output_Images/Loss_Curve.png" height="220px">
</p>

<p align="center">
  <img src="Output_Images/Confusion_Matrix.png" height="220px" style="margin-right: 10px;">
  <img src="Output_Images/ROC_Curve.png" height="220px">
</p>

<p align="center">
  <img src="Output_Images/Graph_Visualization.png" height="220px" style="margin-right: 10px;">
  <img src="System_Architecture_Images/Flow_Diagram.png" height="220px">
</p>

---

## ✅ Conclusion

This work shows that **Graph Neural Networks** — specifically **Relational GCNs** — can effectively model spatial and topological relationships within drone-captured crop images, achieving over **94% accuracy** in detecting:

- 🟢 Healthy
- 🍂 Rust
- 🧬 Mosaic
- 🐛 Pest

GCNs provide a compact, explainable architecture for crop health monitoring and precision agriculture.

---

## 🔮 Future Enhancements

- 🛰️ **Integrate remote sensing data** (NDVI, multispectral)  
- 🌐 **Real-time GCN inference** using ONNX or TensorRT  
- 🔍 **Model Explainability** using Graph Attention or Grad-CAM  
- 🚀 **Deploy with Flask/Streamlit** for mobile/web-based monitoring  
- 🧪 Test **GAT / GIN / ChebNet** for improved disease segmentation  

---

## 🚀 Getting Started

#### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/soybean-gcn.git
cd soybean-gcn
