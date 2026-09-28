# Image Captioning and Classification with Vision-Language Models

A practical exploration of Vision-Language Models (VLMs) using BLIP and CLIP with Hugging Face Transformers.

This project demonstrates two important multimodal AI capabilities:

1. Image captioning using BLIP
2. Zero-shot image classification using CLIP

The project also compares the two approaches and explores their practical applications, strengths, and limitations.

## Overview

Vision-Language Models combine visual and textual information to perform tasks that require understanding both images and natural language.

In this project, two different VLM approaches are explored.

BLIP is used to generate natural-language descriptions of images, while CLIP is used to classify images using natural-language labels without additional task-specific training.

The overall workflow is:

```text
                    Vision-Language Models
                            |
                +-----------+-----------+
                |                       |
                v                       v
               BLIP                    CLIP
                |                       |
                v                       v
       Image Captioning        Zero-Shot Classification
                |                       |
                v                       v
      Natural Language Caption    Predicted Text Label
```

## Objectives

The main objectives of this project are to:

* Understand Vision-Language Models
* Generate captions from images using BLIP
* Perform zero-shot image classification using CLIP
* Work with pre-trained multimodal models
* Understand image-text representations
* Analyze model confidence
* Study the effect of label selection on classification
* Compare captioning and classification approaches
* Explore practical VLM applications

## Technologies Used

| Technology                | Purpose                               |
| ------------------------- | ------------------------------------- |
| Python                    | Core programming language             |
| PyTorch                   | Model inference and tensor operations |
| Hugging Face Transformers | BLIP and CLIP models                  |
| Torchvision               | Vision-related utilities              |
| Pillow                    | Image processing                      |
| Matplotlib                | Visualization                         |
| NumPy                     | Numerical operations                  |
| Requests                  | Loading images from URLs              |
| Jupyter Notebook          | Development and experimentation       |

## Models Used

### BLIP

The project uses:

```text
Salesforce/blip-image-captioning-base
```

BLIP is used for image captioning.

The model receives an image and generates a natural-language description of the visual content.

### CLIP

The project uses:

```text
openai/clip-vit-base-patch32
```

CLIP is used for zero-shot image classification.

Instead of training a classifier specifically for the target classes, the image is compared with a set of natural-language labels.

## Task 1: Image Captioning with BLIP

The first task uses BLIP to generate a description for an input image.

The model and processor are loaded using:

```python
from transformers import BlipProcessor, BlipForConditionalGeneration

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
```

An image is loaded and converted to RGB:

```python
image = Image.open("aayush.jpg").convert("RGB")
```

The image is then processed:

```python
inputs = processor(
    images=image,
    return_tensors="pt"
)
```

The model generates the caption:

```python
output = model.generate(**inputs)

caption = processor.decode(
    output[0],
    skip_special_tokens=True
)
```

### BLIP Workflow

```text
Input Image
     |
     v
BLIP Processor
     |
     v
Image Representation
     |
     v
BLIP Model
     |
     v
Generated Text
```

### Caption Analysis

The generated captions can be analyzed based on:

* Accuracy
* Relevance
* Main objects identified
* Scene description
* Level of detail
* Alignment with the actual image

The notebook also emphasizes testing diverse images because evaluating the model on only one type of image may not provide a complete understanding of its behavior.

## Task 2: Zero-Shot Classification with CLIP

The second task uses CLIP for zero-shot image classification.

The model is initialized using:

```python
from transformers import CLIPProcessor, CLIPModel

model_name = "openai/clip-vit-base-patch32"

model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)
```

An image is loaded and a list of natural-language labels is created.

Example labels used in the notebook include:

```text
man standing at park
man standing in stadium
man standing at mountain
```

The image and text labels are processed together:

```python
inputs = processor(
    text=labels,
    images=image,
    return_tensors="pt",
    padding=True
)
```

The model calculates image-text similarity scores:

```python
outputs = model(**inputs)

logits_per_image = outputs.logits_per_image
```

The scores are converted into probabilities using softmax:

```python
probabilities = torch.softmax(
    logits_per_image,
    dim=1
)
```

The label with the highest probability is selected:

```python
predicted_index = probabilities.argmax(
    dim=1
).item()

predicted_label = labels[predicted_index]
```

## CLIP Workflow

```text
Input Image
     |
     v
Image Encoder
     |
     v
Image Representation
     |
     |
     +----------------------+
                            |
                            v
                     Similarity Scores
                            ^
                            |
     +----------------------+
     |
     v
Text Labels
     |
     v
Text Encoder
     |
     v
Text Representations
```

The label with the strongest similarity to the image is selected as the prediction.

## Zero-Shot Classification

One of the important concepts demonstrated in this project is zero-shot classification.

Instead of training a new classification model for the target categories, CLIP compares the image against natural-language descriptions.

For example:

```text
Image
  |
  +---- "A man standing at a park"
  |
  +---- "A man standing in a stadium"
  |
  +---- "A man standing at a mountain"
  |
  v
Similarity Comparison
  |
  v
Highest Scoring Label
```

This allows CLIP to perform classification using labels provided at inference time.

## Importance of Label Selection

The notebook highlights that the quality of the labels can influence classification performance.

Labels that are:

* Clear
* Relevant
* Specific
* Descriptive

can provide better classification signals.

A limited or poorly chosen label set may fail to capture important differences between images.

Therefore, zero-shot classification should consider both model predictions and the quality of the candidate labels.

## Task 3: BLIP vs CLIP

The project compares the two VLM approaches.

| Feature                    | BLIP                         | CLIP                     |
| -------------------------- | ---------------------------- | ------------------------ |
| Primary Task               | Image Captioning             | Zero-Shot Classification |
| Input                      | Image                        | Image + Text Labels      |
| Output                     | Natural-language description | Predicted label          |
| Main Capability            | Describe visual content      | Match image and text     |
| Training Required for Task | No additional training       | No additional training   |
| Output Type                | Free-form text               | Candidate label          |

### BLIP

BLIP is primarily used when the objective is to describe an image using natural language.

Example:

```text
Input:
Image

Output:
"A person standing outside near a building."
```

### CLIP

CLIP is useful when the objective is to determine which of several text descriptions best matches an image.

Example:

```text
Input:
Image

Labels:
Park
Stadium
Mountain

Output:
Park
```

## Strengths and Limitations

### BLIP Strengths

* Generates natural-language descriptions
* Useful for image captioning
* Can describe objects and scenes
* Does not require task-specific fine-tuning for basic caption generation

### BLIP Limitations

* Generated descriptions may not always capture every detail
* Captions can sometimes be overly general
* Performance depends on the visual content of the image

### CLIP Strengths

* Supports zero-shot classification
* Does not require additional training for new candidate labels
* Can compare images with natural-language descriptions
* Useful for image search and categorization

### CLIP Limitations

* Results depend on the quality of candidate labels
* A limited label vocabulary can miss important visual distinctions
* Similarity scores should be interpreted in the context of the candidate labels

## Real-World Applications

Vision-Language Models can be used in a wide range of applications.

### Image Captioning

BLIP-style models can support:

* Accessibility tools
* Automatic image descriptions
* Content management
* Image search
* Digital asset management

### Zero-Shot Classification

CLIP-style models can support:

* Image categorization
* Visual search
* Content moderation
* Product classification
* Image retrieval

### E-Commerce

VLMs can help connect product images with natural-language descriptions and categories.

### Accessibility

Image captioning can help provide descriptions of visual content for users who cannot directly access the image.

### Robotics

Vision-language models can help systems connect visual observations with natural-language instructions.

### Digital Content Curation

VLMs can automatically describe, categorize, and organize large collections of images.

## Learning Outcomes

This project provides practical experience with:

* Vision-Language Models
* Multimodal AI
* BLIP
* CLIP
* Image captioning
* Zero-shot classification
* Image-text similarity
* Natural-language labels
* Softmax probabilities
* Hugging Face Transformers
* PyTorch
* Multimodal model evaluation

## Installation

Install the required dependencies:

```bash
pip install transformers torch torchvision pillow matplotlib requests
```

## How to Run

### 1. Clone the Repository

```bash
git clone <your-repository-link>
```

### 2. Navigate to the Project

```bash
cd <repository-name>
```

### 3. Install Dependencies

```bash
pip install transformers torch torchvision pillow matplotlib requests
```

### 4. Add the Required Images

The notebook currently references:

```text
aayush.jpg
ayush1.jpeg
```

Place the required images in the project directory or update the image paths in the notebook.

### 5. Open the Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
vlms.ipynb
```

### 6. Run the Notebook

Run the cells sequentially.

The notebook will:

1. Install and import the required libraries
2. Load the BLIP model
3. Generate an image caption
4. Load the CLIP model
5. Define natural-language classification labels
6. Calculate image-text similarity
7. Generate classification probabilities
8. Select the predicted label
9. Compare BLIP and CLIP
10. Analyze practical applications and limitations

## Project Structure

```text
Vision-Language-Models/
|
├── vlms.ipynb
├── aayush.jpg
├── ayush1.jpeg
└── README.md
```

## Limitations

This project is designed as a practical introduction to VLMs rather than a comprehensive benchmark.

The notebook:

* Uses a small number of images
* Uses manually selected labels for CLIP
* Does not fine-tune either model
* Does not perform large-scale quantitative evaluation
* Does not compare multiple VLM architectures
* Does not use a dedicated benchmark dataset

The classification results are dependent on the labels selected for the experiment.

## Future Improvements

The project can be extended by:

* Testing BLIP on a larger image dataset
* Evaluating caption quality using standard metrics
* Testing CLIP on larger classification datasets
* Experimenting with different label prompts
* Comparing multiple VLM architectures
* Building an image search system
* Creating an interactive Streamlit application
* Combining BLIP and CLIP into a single multimodal application
* Exploring Visual Question Answering
* Fine-tuning VLMs on custom datasets

## Conclusion

This project provides a practical introduction to Vision-Language Models through two complementary tasks.

BLIP demonstrates how visual information can be converted into natural-language descriptions, while CLIP demonstrates how images can be matched against natural-language labels for zero-shot classification.

Together, these experiments show how modern VLMs bridge visual and textual information and provide a foundation for building applications in image search, accessibility, e-commerce, content analysis, robotics, and other multimodal AI systems.

## Author

Aayush Kumar
