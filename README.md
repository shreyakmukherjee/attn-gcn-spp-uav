<h1 > 🌿 Attention-Enhanced GCN with SPP for
 UAV-Captured Plant Imagery </h1>

This repository presents a pipeline for detecting diseases in soybean leaves using **Graph Convolutional Networks (GCN)**. It classifies leaves into four categories — **Healthy**, **Rust**, **Mosaic**, and **Pest** — using a relational graph-based approach implemented in PyTorch Geometric.

---

## 🧾 Project Overview
<div style="width:100%; overflow-x:auto; margin:20px 0; font-family:Arial, sans-serif; box-shadow:0 2px 8px rgba(0,0,0,0.1); border-radius:8px;">

<table style="width:100%; border-collapse:collapse;">
    <thead>
        <tr style="background-color:#f8f9fa; border-bottom:2px solid #e0e0e0;">
            <th style="padding:12px 15px; text-align:left; font-weight:600; color:#333; min-width:120px;">Attribute</th>
            <th style="padding:12px 15px; text-align:left; font-weight:600; color:#333;">Description</th>
        </tr>
    </thead>
    <tbody>
        <tr style="border-bottom:1px solid #e0e0e0;">
            <td style="padding:12px 15px; font-weight:500; color:#555;">📚 Framework</td>
            <td style="padding:12px 15px;">PyTorch, PyTorch Geometric, Scikit-learn</td>
        </tr>
        <tr style="border-bottom:1px solid #e0e0e0;">
            <td style="padding:12px 15px; font-weight:500; color:#555;">🧠 Models Used</td>
            <td style="padding:12px 15px;">Graph Convolutional Network (GCN), Relational GCN</td>
        </tr>
        <tr style="border-bottom:1px solid #e0e0e0;">
            <td style="padding:12px 15px; font-weight:500; color:#555;">📷 Input Format</td>
            <td style="padding:12px 15px;">Drone-captured Images (~256×256 px)</td>
        </tr>
        <tr style="border-bottom:1px solid #e0e0e0;">
            <td style="padding:12px 15px; font-weight:500; color:#555;">🎯 Output</td>
            <td style="padding:12px 15px;">4-Class Image Classification</td>
        </tr>
        <tr style="border-bottom:1px solid #e0e0e0;">
            <td style="padding:12px 15px; font-weight:500; color:#555;">🧪 File</td>
            <td style="padding:12px 15px;"><code>gcn-code.ipynb</code></td>
        </tr>
        <tr>
            <td style="padding:12px 15px; font-weight:500; color:#555;">📁 Dataset Source</td>
            <td style="padding:12px 15px;"><a href="https://data.mendeley.com/datasets/hkbgh5s3b7/1" target="_blank" style="color:#1a73e8; text-decoration:none;">Soybean Leaf Dataset (Kaggle)</a></td>
        </tr>
    </tbody>
</table>

</div>

---

## 🗂 Dataset Overview

This dataset includes drone-captured images of soybean plants from various growth stages, classified into four types based on the disease:

<h3 >📊 Class Distribution</h3>

<table >
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
<div>
  
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
<div>
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
  <tr><td>Batch Size</td><td>16</td></tr>
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

<p >
  <img src="Output_Images/Accuracy_Curve.png" height="220px" style="margin-right: 10px;">
  <img src="Output_Images/Loss_Curve.png" height="220px">
</p>

<p >
  <img src="Output_Images/Confusion_Matrix.png" height="220px" style="margin-right: 10px;">
  <img src="Output_Images/ROC_Curve.png" height="220px">
</p>

<p >
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
git clone https://github.com/suman2896/attn-gcn-spp-uav.git
cd attn-gcn-spp-uav
```

# Install project-specific packages
```bash
pip install -r requirements.txt
```

