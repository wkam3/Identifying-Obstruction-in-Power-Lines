# Identifying Obstructions in Power Lines

This guide will help you set up a Python environment to run the object detection notebook.

---

## **Setup Instructions with Anaconda Prompt**  

### **1. Clone the Repository**  
```bash
git clone https://github.com/YOUR_USERNAME/Identifying-Obstruction-in-Power-Lines.git
cd Identifying-Obstruction-in-Power-Lines
```

---

### **2. Create a New Conda Environment**  
```bash
conda create --name object-detection-env python=3.10
```

Activate the environment:  
```bash
conda activate object-detection-env
```

---

### **3. Install Dependencies**  
```bash
pip install -r requirements.txt
```

---

### **4. Start Jupyter Notebook**  
```bash
pip install notebook
jupyter notebook
```

Open the `ObjectDetection_Final.ipynb` file and run the cells.

---

### **5. Troubleshooting**  
If you encounter issues with `ultralytics` or `roboflow`, ensure you have the latest versions:  
```bash
pip install --upgrade ultralytics roboflow
```

If OpenCV (`cv2`) does not work properly, try installing:  
```bash
pip install opencv-python-headless
```

---

### **6. Deactivating the Environment**  
When you're done, deactivate the virtual environment:  
```bash
conda deactivate
```

