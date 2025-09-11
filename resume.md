---
layout: page
title: Resume
permalink: /resume/
---

[GitHub 🐱‍💻](https://github.com/Mrkomiljon) &#124;  [LinkedIn 🔗](https://www.linkedin.com/in/komiljon-mukhammadiev/) &#124; [Medium 📝](https://medium.com/@uzbrainai)

## About Me

I am a dedicated **AI Developer** with **3+ years of industry** and **3+ years of academic experience** in machine learning, deep learning, NLP, and computer vision. My expertise spans **speech-to-text (STT), large language models (LLMs), Retrieval-Augmented Generation (RAG), DeepFake detection, AI-generated content classification, and multi-agent orchestration**.  

I have successfully built and deployed **real-time AI systems** for **voice authentication, phishing detection, audio watermarking, and AI-driven audio/text detection**, as well as **RAG-based knowledge retrieval pipelines** and **multi-agent platforms** for domain-specific use cases.  

I specialize in **model quantization and optimization (ONNX, TensorFlow Lite, TensorRT)**, ensuring scalable AI solutions for both mobile and server deployments. My current focus includes **multilingual speech recognition, AI-based security systems, orchestration of intelligent agents, and real-time inference acceleration**, delivering efficient and production-ready AI applications.

---
**Core Skills**

- **Programming Languages:** Python, C/C++, Java  
- **Database Management:** MySQL, PostgreSQL, PySpark  
- **ML & AI Frameworks:** PyTorch, TensorFlow, HuggingFace Transformers, Scikit-learn, PyTorch Lightning  
- **Speech & NLP:** Whisper, KoBERT, Speech-to-Text (STT), Large Language Models (LLMs), Retrieval-Augmented Generation (RAG)  
- **MLOps & Optimization:** Docker, Kubernetes, MLflow, FastAPI, TorchServe, TensorRT, ONNX, TensorFlow Lite  
- **Development Tools:** Git/GitHub, CI/CD, Docker-compose  
- **Cloud & Deployment:** AWS, GCP, Azure, Edge AI & Mobile Deployment (ONNX/TFLite/TensorRT)  

---
**Main Competencies**

- **Computer Vision & Image Processing:** Object Detection & Tracking, OCR, Image Enhancement, Medical Imaging  
- **Speech & NLP:** Multilingual STT, AI vs Human Text Detection, Large Language Models (LLMs), Vision-Language Models, Generative AI  
- **AI Security:** DeepFake Detection (voice & video), Voice Spoofing Detection, Audio Watermarking  
- **AI Model Development & Optimization:** Model Quantization, Real-time Inference Acceleration, Edge/Mobile AI Optimization  
- **End-to-End AI Solutions:** Scalable AI Pipelines, Multi-Agent Orchestration (LangChain/LangGraph), Cloud AI Deployment  
---
**Recent Achievements** 

- **Retrieval-Augmented Generation (RAG)**
  - Developed multilingual (UZ/RU/EN) RAG pipelines with indexing, retrieval, and synthesis.  
  - Integrated Pinecone/Qdrant/FAISS with BGE re-ranking, using LangChain & LangGraph with guardrails.  

- **Text Detection (AI vs Human)**
  - Built a hybrid model combining stylometric features and transformers (RoBERTa/BERT).  
  - Set up monitoring with accuracy, F1, ROC-AUC, and drift detection metrics.  

- **AI Agent Orchestration**
  - Designed multi-agent workflows (Planner → Router → Tool Agents → Critic) for complex tasks.  
  - Deployed production-ready pipelines with LangGraph, FastAPI, Celery/Redis, and OpenTelemetry.  

- **Voice Security / Speech**
  - Fine-tuned Whisper STT for low-latency, real-time chunked inference; optimized for on-device ONNX/TFLite.  
  - Implemented pipelines and APIs for AI-generated vs Human Voice detection (**95%+ accuracy on benchmark dataset**).  

- **Infra & MLOps**
  - Implemented automated CI/CD with GitHub Actions, model versioning with MLflow, and **A/B evaluation + Canary releases**.  
  - Delivered edge/mobile deployment with TensorRT, ONNX Runtime, TFLite; cloud deployment on GCP and Azure.  
---
  
## Work Experience

### **AI Developer**
**[Museblossome](https://info.museblossom.com/)** | **Nov 2024 - Present**

- **DeepVoice – Real-time Voice Phishing Detection**  
  - Achieved 98% accuracy by integrating Speech-to-Text (STT) and a fine-tuned KoBERT model for phishing detection.  
  - Optimized inference speed by processing audio in chunks for efficient streaming.  
  - Deployed the model on Android using ONNX and TensorFlow Lite.
  - [Project implementaion](https://github.com/Mrkomiljon/DEEPVOICE)

- **DeepVoiceGuard – AI vs Human Voice Classification**  
  - Created a model that identifies AI-generated voices in phone conversations.  
  - Collected and processed ASVspoof2019 dataset of real and AI-generated voices.  
  - Achieved 95.8% accuracy and deployed on huggingface and local servers via FastAPI.
  - [Project implementaion](https://huggingface.co/Mrkomiljon/DeepVoiceGuard)

- **Fine-Tuning Whisper for Korean and Uzbek Speech Recognition**  
  - Fine-tuned Whisper-medium model for Uzbek speech-to-text using a fully custom dataset collected and curated specifically for the Uzbek language.  
  - Achieved a Word Error Rate (WER) of 6.48% on the evaluation set, demonstrating high accuracy for real-world Uzbek speech scenarios.  
  - Built a Korean speech dataset combining voice phishing, phone conversations, and AI Hub data for domain-specific STT.
  - Achieved a WER of 9.17% on Korean test data, emphasizing the model’s effectiveness in voice security contexts.  

- **AI-Generated vs Real Music Classification**  
  - Curated a dataset of 1M+ samples across 10 classes for music classification.  
  - Developed a custom AI model to distinguish real vs AI-generated music.  
  - Applied quantization (ONNX, TensorFlow Lite) for high-performance mobile inference.
    
- **[AudioDefence](https://audiodefence.com/) – Audio Watermarking & Spectrogram Classification**
  - Built a system to embed 16-character Morse code serials into audio and detect them using deep learning.
  - Created a dataset of 72,000+ spectrogram images across 36 classes (STFT & Mel).
  - Trained EfficientNet and ResNet50 models with high classification accuracy.
  - Developed LUFS-adaptive thresholding and STFT-based decoders for reliable watermark extraction.


### AI Research Engineer

**[Aria Studios Co. Ltd](https://showaria.com/)** &#124; **Jun 2023 - Nov 2024**

- **Real-time Live Portrait Optimization**:
    - Optimized the Live_Portrait model for real-time performance using webcam and monitor setups, achieving seamless and responsive operation.
    - This project has gained significant recognition on Git-hub, receiving a high number of stars and positive feedback from the community.
    - Technologies: Real-time Image Processing, Webcam&Monitor Integration, Model Optimization, Python.
    - [Project implementation](https://github.com/Mrkomiljon/Live_Portrait_Monitor)
- **Image Enhancement & Deep-fake Creation for Broadcast**:
    - Enhanced image quality and restored facial features to improve the realism of Deep-fake videos.
    - Produced high-quality Deep-fake videos for KBS election coverage, showcasing the potential of advanced ML techniques in media.
    - Technologies: Image Enhancement, Face Restoration, Deep-fake Generation, Python,GANs,  Open-CV.
    - [Project implementation](https://www.youtube.com/live/CGbvG8S7HHo)
- **Multimodal User Interaction System**:
    - Created an integrated system combining gaze tracking, emotion estimation, and audio-to-text conversion to enhance user interaction.
    - Enabled real-time adaptive responses for entertainment applications, significantly improving user engagement.
    - Technologies: Gaze Tracking, Emotion Estimation, Audio Processing, Python, Machine Learning.
- **Interactive Hyundai Car Models**:
    - Developed a model pipeline using the IP-Adapter model to generate interactive 3D car models from grayscale images and user-provided text prompts.
    - Enabled users to design and visualize both classic and futuristic car models in real-time, preserving the target logos and aesthetics.
    - Technologies: 3D Modeling, Image-to-3D Conversion, User Interaction, Python, TensorFlow.
- **3D Scene Creation for Interactive Films**:
    - Implemented 3D Gaussian splatting to create detailed and lifelike 3D reconstructions from point cloud data.
    - Optimized 3D rendering processes to enhance the realism of environments and characters in interactive films.
    - Technologies: 3D Reconstruction, Gaussian Splatting, Point Cloud Processing, Open-CV, PyTorch.
- **Facial Performance Transfer System**:
    - Developed an advanced system for AI avatars to deliver multilingual speech with highly realistic facial expressions and lip synchronization.
    - Leveraged state-of-the-art deep learning and real-time video processing techniques, resulting in a 30% increase in user engagement.
    - Utilized a combination of neural networks for generating high-quality deep-fake videos based on driver video and audio inputs.
    - Technologies: Deep Learning, Real-time Video Processing, Facial Animation, Python, TensorFlow.

## Research Experience

### Research Assistant

**CNU AI & A Lab** &#124; **Sep 2020 - Fev 2023**

- **Uzbek Sign Language Detection System**:
    -   Developed a real-time system using Mediapipe and OpenCV to recognize Uzbek sign language with 98% accuracy.
    -   Translated hand gestures into text, providing an effective communication tool for the hearing impaired.
    -   Technologies: Sign Language Recognition, Computer Vision, Mediapipe, OpenCV, Python.
    -   [Project Implementation](https://github.com/Mrkomiljon/uzbek-sign-language) 
- **License Plate Detection System**:
    -   Implemented a high-precision license plate detection system using the YOLOv7 model and CCPD dataset.
    -   Achieved robust performance in diverse environments, enhancing automated vehicle monitoring and access control systems.
    -   Technologies: Object Detection, YOLOv7, Image Processing, Python.
- **Early Lung Cancer Detection Model**:
    -   Built and optimized a classification and segmentation model to detect early-stage lung cancer, improving diagnostic accuracy by 20%.
    -   Technologies: Medical Imaging, Machine Learning, Image Segmentation, Python, TensorFlow.
Please visit [https://github.com/Mrkomiljon](https://github.com/Mrkomiljon) to see more implementations of different ML models.


## Education

| **Institution**                                 | **Degree**                                                                | **Duration**        |
| ----------------------------------------------- | ------------------------------------------------------------------------- | ------------------- |
| Chonnam National University                               | MSc in Computer Engineering; advised by Prof. Chang Gyoon Lim; GPA: 3.63/4.5 | Sep 2019 - Feb 2023 |
| Tashkent University of Information Technologies | BSc in Computer Engineering; GPA(%): 85/100                               | Sep 2014 - Jun 2018 |

## Publications
 “An efficient stacking ensemble learning method for customer [churn prediction](https://github.com/Mrkomiljon/Churn-prediction)”, (2023)


## Languages

- **English:** Full Professional Proficiency 
- **Korean:** Limited Working Proficiency 
- **Uzbek:** Native Proficiency
- **Russian:** Limited Working Proficiency

Last Updated: 2025-09-11


