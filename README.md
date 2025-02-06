# Identifying Obstructions in Power Lines

This guide will help you set up a Python environment to run the object detection notebook.

---

## **Setup Instructions with Anaconda Prompt**  

### **1. Log onto DSMLP for gpu access (campus vpn method)**  
```bash
ssh <username>@dsmlp-login.ucsd.edu
launch-scipy-ml.sh -W DSC180A_FA24_A00 -g 1
```

---

### **2. Clone the Repository**  
```bash
git clone https://github.com/YOUR_USERNAME/Identifying-Obstruction-in-Power-Lines.git
cd Identifying-Obstruction-in-Power-Lines/Project_code
```

---

### **3. Create a New Conda Environment**  
```bash
conda create --name object-detection-env python=3.10
```

Activate the environment:  
```bash
conda activate object-detection-env
```

---

### **4. Install Dependencies**  
```bash
pip install -r requirements.txt
```

---

### **5. Run the Script**  
```bash
python script.py setup data train annotate
```
The whole script should take 20-30 minutes due to training. Final annotated images are stored in the Annotated Folder.

---

### **6. Troubleshooting**  
If you encounter issues with `ultralytics` or `roboflow`, ensure you have the latest versions:  
```bash
pip install --upgrade ultralytics roboflow
```

If OpenCV (`cv2`) does not work properly, try installing:  
```bash
pip install opencv-python-headless
```

If the error still persists, restart your terminal.

---

### **7. Deactivating the Environment**  
When you're done, deactivate the virtual environment:  
```bash
conda deactivate
```

