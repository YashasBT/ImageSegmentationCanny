🖼️ Edge-Based Image Segmentation using Canny Algorithm
📌 Overview

This project implements edge-based image segmentation using the Canny Edge Detection algorithm and evaluates its performance against ground truth annotations using standard segmentation metrics.

The goal is to analyze how classical image processing techniques perform compared to annotated datasets and understand their limitations.

----------------------------------------------------------------------------------------

🚀 Features
Canny Edge Detection for segmentation
Ground truth extraction from .mat files (BSDS dataset format)
Morphological operations (closing, dilation) for refinement
Evaluation metrics:
Dice Coefficient (F1 Score)
IoU (Jaccard Index)
Pixel Accuracy
Precision & Recall
Per-image visualization (Original, Ground Truth, Prediction, Overlay)
Results saved as images and CSV table

----------------------------------------------------------------------------------------

📂 Project Structure
IP_PROJ/
│── Images/            # Input images
│── Groundtruth/       # .mat ground truth files
│── outputs/           # Output visualizations
│── main.py            # Main execution file
│── results_table.csv  # Generated results table
│── README.md          # Project documentation

----------------------------------------------------------------------------------------

▶️ Usage

Run the main script:

python main.py

----------------------------------------------------------------------------------------

📊 Output
Segmented images saved in outputs/
Performance metrics displayed in terminal
Results table saved as:
results_table.csv

----------------------------------------------------------------------------------------

📈 Evaluation Metrics

The segmentation performance is evaluated using:

Dice Coefficient – Measures overlap between prediction and ground truth
IoU (Jaccard Index) – Intersection over union
Pixel Accuracy – Correct pixel classification ratio
Precision – Correct positive predictions
Recall – Detection completeness

----------------------------------------------------------------------------------------

🧠 Methodology
Convert image to grayscale
Apply Canny Edge Detection
Perform morphological operations
Load and process ground truth from .mat
Compare prediction with ground truth
Compute evaluation metrics

----------------------------------------------------------------------------------------

⚠️ Limitations
Detects only edges, not full object regions
Lower overlap with dense ground truth masks
Sensitive to noise and threshold selection

----------------------------------------------------------------------------------------
🔮 Future Scope
Use deep learning models like U-Net for better segmentation
Hybrid approach combining edge detection + AI models
Real-time segmentation systems

----------------------------------------------------------------------------------------

📌 Conclusion

This project demonstrates the effectiveness of classical edge detection techniques while highlighting their limitations compared to modern AI-based segmentation methods.

----------------------------------------------------------------------------------------

👨‍💻 Author
    Yashas BT
----------------------------------------------------------------------------------------    
📄 License

This project is for academic purposes.
