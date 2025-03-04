# shapenet_chairs_GAN

## **Shape Generation using Spatially Partitioned Point Clouds**

### **What’s This?**
This is my implementation of **"Shape Generation using Spatially Partitioned Point Clouds"** by Matheus Gadelha, Subhransu Maji, and Rui Wang.  
The paper proposes a method to generate 3D shapes using **point clouds** by spatially ordering them with **kd-trees**, applying **PCA**, and training a **GAN** to learn shape distributions.   

I found this paper online and thought it was cool, so I decided to implement it using a dataset I got from the internet—**ShapeNet chairs** in `.pcd` format.   

### **How It Works**
1. **kd-tree Sorting:** Orders points in a consistent way across shapes.
2. **PCA Basis Learning:** Reduces dimensionality by learning a shape basis.
3. **GAN Training:** Learns a distribution over shape coefficients.
4. **Shape Generation:** Uses the trained GAN to generate new point clouds.  

### **Project Structure**
```
project_root/
│── data/                       # Store raw and processed data
│   ├── shapenet-chairs-pcd/    # Folder containing .pcd files
│── models/                     # Store saved models
│── logs/                       # TensorBoard logs
│── scripts/                    # Main scripts
│   ├── train.py                # Trains the GAN, logs to TensorBoard
│   ├── inference.py            # Generates new shapes as point clouds
│── utils/                      # Utility functions
│   ├── dataset.py              # Loads and preprocesses point clouds
│   ├── visualize.py            # Visualization tools for point clouds
│   ├── models.py               # Generator & Discriminator models
│── requirements.txt            # Dependencies
│── README.md                   # You’re reading this
```

### **Setup & Training**
#### 1️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```
#### 2️⃣ **Train the Model**
```bash
python scripts/train.py
```
- Uses TensorBoard for logging, so you can track progress with:
```bash
tensorboard --logdir=logs
```

#### 3️⃣ **Generate Shapes**
```bash
python scripts/inference.py
```
This loads a trained model and generates new 3D shapes as **point clouds**.

### **Why This?**
- **Point clouds > Voxels** → Less memory, better resolution.
- **kd-tree sorting** → Adds structure to unordered points.
- **GAN-based** → More expressive than simple PCA models.
- **Runs on a mid-tier GPU** → Doesn’t require insane compute.

### **What’s Next?**
- Maybe add **normal/color attributes** to the point clouds?  
- Try training on **other ShapeNet categories** (tables, sofas, etc.).  
- Improve the generator for **better diversity** in shapes.
