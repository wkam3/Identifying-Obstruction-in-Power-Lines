# Identifying Obstructions in Power Lines

You can access our website here: https://wkam3.github.io/Identifying-Obstruction-in-Power-Lines/
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
cd Identifying-Obstruction-in-Power-Lines
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
You need to specify which model you want to run the model options availabe are AllClasses, Obstruction-Only, and NMS. For NMS the functions are a bit different and are shown below the other two models are the same except for the model type parameter.
```bash
cd Project_code
#NMS model
python script.py NMS setup data train_nms_model annotate_nms_model evaluate_nms_model
#All Classes
python script.py AllClasses setup data train annotate evaluate
#Obstruction Only
python script.py ObstructionsOnly setup data train annotate evaluate
```
The whole script should take 20-30 minutes due to training for obstruction only and all classes model while the NMS takes 40-60 minutes as we are training three models in this approach. Final annotated images are stored in respective Annotated Folders.

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

If you are having issues with the environment and running in DSMLP, launch the scipy-ml environment and try:
```bash
conda activate object-detection-env
```
```bash
pip install ipykernel
```
```bash
python -m ipykernel install --user --name object-detection-env --display-name "Python (object-detection-env)"
```
Then select the kernel "Python (object-detection-env)" in the notebook.

If you are having gpu errors such as error running pods, you may have to delete all pods and ssh in again,=.
```bash
kubectl delete --all pods
```
---

### **7 Activating the Demo App
1. Open the project
2. activate environment (described earlier)
3. run (for mac. May be different for windows.):
```bash
python Project_code/app.py
```
4. The output in the terminal should have a local URL and a public URL. Click on either.
5. You're ready to use the app! 😊

---

### **8. Deactivating the Environment**  
When you're done, deactivate the virtual environment:  
```bash
conda deactivate
```

