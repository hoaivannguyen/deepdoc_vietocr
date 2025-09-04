<p align="center">
  <a href="./README.md">Tiếng Việt</a> |
  <a href="./README_en.md">English</a> |
</p>

# *Deep*Doc + VietOCR - Fast and Cost-effective OCR Tool for Vietnamese

- [1. Introduction](#1)
- [2. Architecture](#2)
- [3. Installation & Running](#3)

<a name="1"></a>

## 1. Introduction

With a wide range of documents from various sources and formats, along with diverse retrieval requirements, an accurate extraction tool is essential for any business. Today, I'd like to introduce DeepDoc, a very fast and cost-efficient OCR tool that only requires running on a CPU. In addition, it also comes with Layout Recognizer and Table Structure Recognizer features, which help preserve the document's formatting after OCR.

However, DeepDoc has not yet been standardized for Vietnamese, so I replaced the Text Recognizer with VietOCR and the ONNX version to achieve better Vietnamese text recognition. You can also check out the original version of DeepDoc [here](https://github.com/infiniflow/ragflow/blob/main/deepdoc/README.md). Moreover, since DeepDoc is essentially a data processing component for the RAG pipeline in the RAGFlow project, I separated it into an independent Git repository so the application can be customized more conveniently.

<a name="2"></a>

## 2. Architecture
### 2.1 OCR
In this part, DeepDoc uses PaddleOCR - a very popular open-source tool developed by Baidu - after converting it into ONNX. Basically, ONNX (Open Neural Network Exchange) is an open format for AI models, allowing export and import of models between multiple frameworks (PyTorch, TensorFlow, etc.). It enables cross-platform compatibility, optimizes inference speed on CPU/GPU, and reduces infrastructure costs when deployed (we won't go too deep into this topic here).

DeepDoc does not specify which version it uses, since after conversion to ONNX it's difficult to determine. To get an idea of how it works, I'll refer to the OCR architecture PP-OCRv5 from the latest PaddleOCR 3.0, which includes four main components:

- Image Preprocessing Module: Enhances image quality, handles rotation/skew using orientation classification (PP-LCNet) and unwarping (UVDoc).

- Text Detection: Upgraded from PP-OCRv4 with backbone PP-HGNetV2, knowledge distillation from GOT-OCR2.0, and data augmentation (synthetic generation, rotation, blur, distortion). Retains PFHead and DSR from the previous version.

- Text Line Orientation Classification: Automatically detects and corrects text line orientation (flipped, rotated) to prepare for recognition.

- Text Recognition: Two-branch architecture with PP-HGNetV2, trained with GTC-NRTR (attention-based) to guide SVTR-HGNet (CTC, lightweight, fast). Training data is augmented with documents, PDFs, e-books, and synthetic handwriting samples.

<div align="center" style="margin-top:20px;margin-bottom:20px;">
    <img src="img\x6.png" width="900"/>
</div>

For more details about PP-OCRv5, you can refer to the official documentation [here](https://arxiv.org/html/2507.05595v1).

As mentioned above, the Recognition module of Paddle has been replaced with VietOCR and its ONNX version to achieve more accurate Vietnamese text recognition. VietOCR is already a very popular OCR tool in Vietnam, so I won't go into details here - you can explore more about it [here](https://github.com/pbcquoc/vietocr). For the process of converting VietOCR into the ONNX format, I referred to [this article](https://viblo.asia/p/chuyen-doi-mo-hinh-hoc-sau-ve-onnx-bWrZnz4vZxw).

### 2.2 Layout Recognizer & Table Structure Recognizer
In this part, DeepDoc uses YOLOv10 (You Only Look Once) - also a popular object detection method - in its ONNX version.

The basic architecture consists of three main components:
- Backbone: Extracts features from the image, using a lightweight and efficient design (retaining the ideas from YOLOv8 but improving the blocks to reduce computation).
- Neck: Combines multi-scale features (an improved FPN/PAN) to detect both small and large objects effectively.
- Head: Uses an anchor-free decoupled head (separating classification and regression branches), which improves accuracy and makes training easier.

<div align="center" style="margin-top:20px;margin-bottom:20px;">
    <img src="img\af645ed9-7301-4ec4-81e7-cb996ddf2d7f.webp" width="900"/>
</div>


In DeepDoc, YOLOv10 is trained to recognize label types for both Layout Recognizer and Table Structure Recognizer, covering most common cases.

For Layout Recognizer, there are 10 categories:
- Text
- Title
- Image
- Image Caption
- Table
- Table Caption
- Header
- Footer
- Reference
- Equation

For Table Structure Recognition, there are 5 types:
- Column
- Row
- Column header
- Projected row header
- Spanning cell

To understand more about YOLOv10, you can refer to the official documentation [here](https://arxiv.org/pdf/2405.14458).

<a name="3"></a>

## 3. Installation and Testing

First, clone the git repository:
```bash
git clone https://github.com/hoaivannguyen/deepdoc_vietocr.git
```
Some setup options before running the program:
```bash
python t_ocr.py -h
usage: t_ocr.py [-h] --inputs INPUTS [--output_dir OUTPUT_DIR]

options:
  -h, --help            Display this help message and exit
  --inputs INPUTS       Directory containing images or PDF files, or a file path to a single image or PDF file
  --output_dir OUTPUT_DIR
                        Directory to store output images. Default:'./ocr_outputs'
```
```bash
python t_recognizer.py -h
usage: t_recognizer.py [-h] --inputs INPUTS [--output_dir OUTPUT_DIR] [--threshold THRESHOLD] [--mode {layout,tsr}]

options:
  -h, --help            Display this help message and exit
  --inputs INPUTS       Directory containing images or PDF files, or a file path to a single image or PDF file
  --output_dir OUTPUT_DIR
                        Directory to store output images. Default: './layouts_outputs'
  --threshold THRESHOLD
                        Threshold for filtering detections. Default: 0.5
  --mode {layout,tsr}   Task mode: layout recognizer (layout) or table structure recognizer (tsr)
```
### 3.1. OCR
To test OCR, you can use the following command:
 ```bash
python deepdoc/vision/t_ocr.py --inputs=path_to_images_or_pdfs --output_dir=path_to_store_result
```
The input can be a directory containing images or PDFs, or a single image or PDF file. The output will include 1 image with the detected bounding boxes and 1 text file containing the OCR text.
<div align="center" style="margin-top:20px;margin-bottom:20px;">
<img src="img\Screenshot 2025-08-28 171633.png" width="900"/>
</div>

I'm currently using VietOCR Seq2seq as the default since it runs relatively fast and accurately. You can switch to VietOCR Transformer in module/ocr.py, but I don't recommend it because the processing time is much longer while the accuracy doesn't improve significantly. If you want maximum speed, you can switch to the ONNX version by importing ocr_onnx instead of ocr, though the accuracy will decrease slightly.

### 3.2. Layout Recognizer
Try the following command to see the result of the Layout Recognizer:
```bash
python deepdoc/vision/t_recognizer.py --inputs=path_to_images_or_pdfs --threshold=0.2 --mode=layout --output_dir=path_to_store_result
```
The input can be a directory containing images or PDFs, or a single image or PDF file. The output will include 1 image with the detected labels as shown below:
<div align="center" style="margin-top:20px;margin-bottom:20px;">
<img src="img\49806-Article Text-153529-1-10-20200804_page-0002.jpg" width="1000"/>
</div>

### 3.3. Table Structure Recognizer
Try the following command to see the TSR result:
```bash
python deepdoc/vision/t_recognizer.py --inputs=path_to_images_or_pdfs --threshold=0.2 --mode=tsr --output_dir=path_to_store_result
```

The input can be a directory containing images or PDFs, or a single image or PDF file. The output will include 1 image with the detected labels and 1 markdown file with the table content.
<div align="center" style="margin-top:20px;margin-bottom:20px;">
<img src="img\Screenshot 2025-08-28 182132.png" width="1000"/>
</div>

## Conclusion
I hope you find this tool useful and applicable in practice. If you have any feedback, please leave it in the comments below. Thank you for reading!

## References
DeepDoc repo: https://github.com/infiniflow/ragflow/blob/main/deepdoc/README.md

PP-OCRv5: https://arxiv.org/html/2507.05595v1

VietOCR: https://github.com/pbcquoc/vietocr

VietOCR ONNX: https://viblo.asia/p/chuyen-doi-mo-hinh-hoc-sau-ve-onnx-bWrZnz4vZxw

YOLOv10: https://arxiv.org/pdf/2405.14458
