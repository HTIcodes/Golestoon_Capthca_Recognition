# Golestoon CAPTCHA Recognition

Collected and developed by **HTIcodes** — a project for recognizing Golestoon CAPTCHA images using **character segmentation** and **Convolutional Neural Networks (CNNs)**.  
This repository contains scripts for image preprocessing, segmentation, model training, and CAPTCHA prediction.

---

## Table of Contents
- `Golestoon_Captcha_Segmentation.py` — Script for segmenting CAPTCHA images into individual characters.
- `Golestoon_Captcha_Model.py` — Defines the CNN architecture and includes functions for training and loading the model.
- `Golestoon_Captcha_Breaker.py` — Main script that takes a CAPTCHA image, segments it, and predicts the characters using the trained model.
- `Golestoon_captcha_fetcher.py` — *(Optional)* Script for scraping or collecting Golestoon CAPTCHA samples for dataset preparation.
- `captcha_model_h5.h5` / `captca_model_keras.keras` — Trained model weights required for inference.
- `char_to_idx.json` / `idx_to_char.json` — Character–index mappings for decoding model predictions.
- `.gitignore` — Specifies ignored files for Git.

---

## Example Output

Here’s an example of a predicted CAPTCHA result:

![Predicted CAPTCHA](samples/predicted_example.png)
![Predicted CAPTCHA](samples/predicted_example.png)

## Dependencies
It’s recommended to use a **virtual environment** (venv or conda).

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate   # macOS / Linux
# venv\Scripts\activate     # Windows PowerShell

# Install required packages
pip install --upgrade pip
pip install numpy opencv-python matplotlib pillow tensorflow keras scikit-learn tqdm
````

---

## Quick Start — Run Inference

1. Make sure you have the trained model (`captcha_model_h5.h5` or `captca_model_keras.keras`) and the mapping files (`char_to_idx.json`, `idx_to_char.json`) in the project directory.
2. Run the CAPTCHA breaker script on an image of your choice.

Example:

```bash
# Command line execution
python Golestoon_Captcha_Breaker.py --image samples/example_captcha.png

# Or from Python shell
python -c "from Golestoon_Captcha_Breaker import predict; print(predict('samples/example_captcha.png'))"
```

**Output Example:**

```
Predicted CAPTCHA: aB3dq
```

---

## Project Workflow

1. **Fetch CAPTCHA images** using `Golestoon_captcha_fetcher.py` (optional).
2. **Preprocess and segment** each CAPTCHA using `Golestoon_Captcha_Segmentation.py`:

   * Noise reduction and binarization
   * Contour detection for character regions
   * Character extraction and resizing
3. **Train or load** the CNN model defined in `Golestoon_Captcha_Model.py`.
4. **Predict** CAPTCHA text using `Golestoon_Captcha_Breaker.py`.

---

## Model Training

If you want to train your own model from scratch, use the training functions defined in `Golestoon_Captcha_Model.py`.

```bash
python Golestoon_Captcha_Model.py --train_dir dataset/train --val_dir dataset/val --epochs 30 --batch_size 64
```

After training:

```python
model.save('captcha_model_h5.h5')
```

> **Tip:** For better accuracy, use data augmentation (rotation, shifting, noise), class balancing, and proper learning rate scheduling.

---

## Dataset Structure Example

```
dataset/
  train/
    A/
      img1.png
      img2.png
    B/
      ...
  val/
    ...
  test/
    ...
```

Alternatively, you can store metadata in a CSV file with columns like `image_path` and `label`.

---

## Troubleshooting

* **Low prediction accuracy**

  * Ensure the same preprocessing steps are used in both training and inference.
  * Verify that the mapping JSON files (`char_to_idx`, `idx_to_char`) match the model.
* **Segmentation issues**

  * Adjust thresholding or contour filtering parameters.
  * Apply morphological operations (erode/dilate) to remove noise.
* **Improvement tips**

  * Try deeper CNN architectures or transfer learning.
  * Combine predictions from multiple models (ensemble).

---

## Example Use Cases

* **Offline CAPTCHA recognition**

  * Use `Golestoon_Captcha_Breaker.py` in batch mode to decode stored CAPTCHA images.
* **Dataset evaluation**

  * Measure accuracy and visualize a confusion matrix on the test set.


## Contributing

Contributions are welcome!
To contribute:

1. Open an issue describing your idea or bug report.
2. Fork the repo and create a feature branch.
3. Submit a Pull Request with clear documentation and testing steps.

---

## Contact

Developed by [**HTIcodes**](https://github.com/HTIcodes).
Feel free to open an issue or reach out through GitHub for questions or collaboration.

---

## Acknowledgements

Special thanks to open-source libraries such as **OpenCV**, **TensorFlow**, **Keras**, and **scikit-learn** which made this project possible.

Big thanks to my friends [Pezhman](https://github.com/Pezhm4n), [Aydin](https://github.com/Aydinthr2004), Pouria and Sina
for their time and effort in labeling the CAPTCHA images.  
Your help was essential to build and train the dataset for this project.
---
