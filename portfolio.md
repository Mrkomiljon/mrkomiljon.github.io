---
layout: page
title: Portfolio
permalink: /portfolio/
---

<style>
.projects-container {
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
    justify-content: space-between;
}
.project {
    flex: 1 1 calc(50% - 40px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    padding: 10px;
    margin: 10px 0;
    border-radius: 5px;
    background-color: #fff;
}
.project img {
    width: 100%;
    border-radius: 5px;
}
@media (max-width: 768px) {
    .project {
        flex: 1 1 100%;
    }
}
button {
    margin: 5px;
    padding: 10px;
    border: none;
    background-color: #007BFF;
    color: white;
    border-radius: 5px;
    cursor: pointer;
}
button.active {
    background-color: #0056b3;
}
</style>

<div>
    <button class="filter-button" data-filter="all">All</button>
    <button class="filter-button" data-filter="classification">Classification</button>
    <button class="filter-button" data-filter="detection">Detection</button>
    <button class="filter-button" data-filter="segmentation">Segmentation</button>
    <button class="filter-button" data-filter="face">Face</button>
    <button class="filter-button" data-filter="audio">Audio</button>
</div>

<div class="projects-container">
    <div class="project" data-tags="detection">
        <h3>Live Portrait Monitor</h3>
        <a href="https://github.com/Mrkomiljon/Live_Portrait_Monitor" target="_blank">
            <img src="https://opengraph.githubassets.com/1/Mrkomiljon/Live_Portrait_Monitor" alt="Live Portrait Monitor GitHub Preview">
        </a>
        <p>A deep learning-based application for animating portraits displayed on a monitor, leveraging advanced face reenactment techniques. <a href="https://github.com/Mrkomiljon/Live_Portrait_Monitor" target="_blank">Learn more on GitHub</a>.</p>
    </div>
    <div class="project" data-tags="detection">
        <h3>Webcam Live Portrait</h3>
        <a href="https://github.com/Mrkomiljon/Webcam_Live_Portrait" target="_blank">
            <img src="https://opengraph.githubassets.com/1/Mrkomiljon/Webcam_Live_Portrait" alt="Webcam Live Portrait GitHub Preview">
        </a>
        <p>Real-time portrait animation using a webcam feed, utilizing deep learning-based face tracking and reenactment methods. <a href="https://github.com/Mrkomiljon/Webcam_Live_Portrait" target="_blank">Learn more on GitHub</a>.</p>
    </div>
    <div class="project" data-tags="classification">
        <h3>VoiceGuard</h3>
        <a href="https://github.com/Mrkomiljon/voiceguard" target="_blank">
            <img src="https://opengraph.githubassets.com/1/Mrkomiljon/voiceguard" alt="VoiceGuard GitHub Preview">
        </a>
        <p>An AI-powered system designed to detect voice phishing in real time, ensuring enhanced security against fraudulent audio-based threats. <a href="https://github.com/Mrkomiljon/voiceguard" target="_blank">Learn more on GitHub</a>.</p>
    </div>
    <div class="project" data-tags="segmentation">
        <h3>Face Segmentation</h3>
        <a href="https://github.com/Mrkomiljon/face-segmentation_pytorch" target="_blank">
            <img src="https://opengraph.githubassets.com/1/Mrkomiljon/face-segmentation_pytorch" alt="Face Segmentation GitHub Preview">
        </a>
        <p>Semantic segmentation of facial features using PyTorch, enabling applications in augmented reality, digital makeup, and face modification. <a href="https://github.com/Mrkomiljon/face-segmentation_pytorch" target="_blank">Learn more on GitHub</a>.</p>
    </div>
    <div class="project" data-tags="classification">
        <h3>Deep-Live Monitor</h3>
        <a href="https://github.com/Mrkomiljon/Deep-Live-Monitor" target="_blank">
            <img src="https://opengraph.githubassets.com/1/Mrkomiljon/Deep-Live-Monitor" alt="Deep-Live Monitor GitHub Preview">
        </a>
        <p>A sophisticated deep learning system for animating images displayed on a monitor, leveraging advanced computer vision techniques. <a href="https://github.com/Mrkomiljon/Deep-Live-Monitor" target="_blank">Learn more on GitHub</a>.</p>
    </div>
    <div class="project" data-tags="audio">
        <h3>DeepVoiceGuard</h3>
        <a href="https://github.com/Mrkomiljon/DeepVoiceGuard" target="_blank">
            <img src="https://opengraph.githubassets.com/1/Mrkomiljon/DeepVoiceGuard" alt="DeepVoiceGuard GitHub Preview">
        </a>
        <p>DeepVoiceGuard is a robust solution for detecting spoofed audio in Automatic Speaker Verification (ASV) systems. This project utilizes the RawNet2 model, trained on the ASVspoof 2019 dataset, and deploys the trained model using FastAPI for real-time inference. <a href="https://github.com/Mrkomiljon/DeepVoiceGuard" target="_blank">Learn more on GitHub</a>.</p>
    </div>
    <div class="project" data-tags="audio">
        <h3>VoiceGUARD2</h3>
        <a href="https://github.com/Mrkomiljon/VoiceGUARD2" target="_blank">
            <img src="https://opengraph.githubassets.com/1/Mrkomiljon/VoiceGUARD2" alt="VoiceGUARD2 GitHub Preview">
        </a>
        <p>VoiceGUARD2 offers an end-to-end solution for classifying audio as human or AI-generated using the Wav2Vec2 model. It supports multi-class classification, distinguishing between real voices and synthetic audio produced by models such as DiffWave and WaveNet... The project encompasses dataset preparation, preprocessing, fine-tuning, inference, and API deployment for real-time predictions via FastAPI. <a href="https://github.com/Mrkomiljon/VoiceGUARD2" target="_blank">Learn more on GitHub</a>.</p>
    </div>
    <div class="project" data-tags="detection">
        <h3>face_detection_onnx</h3>
        <a href="https://github.com/Mrkomiljon/face_detection_onnx" target="_blank">
            <img src="https://opengraph.githubassets.com/1/Mrkomiljon/face_detection_onnx" alt="face_detection_onnx GitHub Preview">
        </a>
        <p>This repository implements face detection using the SCRFD model, a fast and lightweight solution optimized for edge devices. The project employs the ONNX format for the model and leverages OpenCV for processing images and videos, enabling efficient and accurate face detection across various media formats. <a href="https://github.com/Mrkomiljon/face_detection_onnx" target="_blank">Learn more on GitHub</a>.</p>
    </div>
    <div class="project" data-tags="detection">
        <h3>License-Plate-Detection_ONNX</h3>
        <a href="https://github.com/Mrkomiljon/License-Plate-Detection_ONNX" target="_blank">
            <img src="https://opengraph.githubassets.com/1/Mrkomiljon/License-Plate-Detection_ONNX" alt="License-Plate-Detection_ONNX GitHub Preview">
        </a>
        <p>This repository provides code and instructions for performing license plate detection using YOLOv5 with ONNX Runtime. It supports inference on images, videos, and webcam feeds, utilizing GPU acceleration for efficient processing. The project includes Python scripts for easy deployment and integration into various applications. <a href="https://github.com/Mrkomiljon/License-Plate-Detection_ONNX" target="_blank">Learn more on GitHub</a>.</p>
    </div>
    <div class="project" data-tags="classification">
        <h3>Uzbek Sign Language Recognition</h3>
        <a href="https://github.com/Mrkomiljon/uzbek-sign-language" target="_blank">
            <img src="https://opengraph.githubassets.com/1/Mrkomiljon/uzbek-sign-language" alt="Uzbek Sign Language Recognition GitHub Preview">
        </a>
        <p>This project focuses on recognizing Uzbek Sign Language (USL), the primary language for deaf and hard of hearing individuals in Uzbekistan. The system aims to facilitate communication by translating USL gestures into text, benefiting both the deaf community and those seeking to communicate with them. The dataset comprises images representing various USL gestures, and the model is trained to accurately classify these signs. <a href="https://github.com/Mrkomiljon/uzbek-sign-language" target="_blank">Learn more on GitHub</a>.</p>
    </div>
    <div class="project" data-tags="audio classification">
    <h3>VoiceVerifier-vv</h3>
    <a href="https://github.com/Mrkomiljon/VoiceVerifier-vv" target="_blank">
        <img src="https://opengraph.githubassets.com/1/Mrkomiljon/VoiceVerifier-vv" alt="VoiceVerifier-vv GitHub Preview">
    </a>
    <p>
        VoiceVerifier-vv is a FastAPI-based speaker classification system that removes silence from audio files, extracts speaker embeddings using SpeechBrain's ECAPA-TDNN model, and performs classification using cosine similarity and a Random Forest classifier.  
        <a href="https://github.com/Mrkomiljon/VoiceVerifier-vv" target="_blank">Learn more on GitHub</a>.
    </p>
    </div>
    <div class="project" data-tags="face">
        <h3>Gaze Emotion Recognition</h3>
        <a href="https://github.com/Mrkomiljon/Gaze_emotion" target="_blank">
            <img src="https://opengraph.githubassets.com/1/Mrkomiljon/Gaze_emotion" alt="Gaze Emotion Recognition GitHub Preview">
        </a>
        <p>This project integrates gaze tracking and facial emotion estimation to analyze user emotions in real-time. Utilizing OpenCV for face detection and DeepFace for emotion recognition, it processes webcam input to display emotion labels on detected faces. Additionally, it includes an audio-to-text feature, enhancing the multimodal analysis capabilities. <a href="https://github.com/Mrkomiljon/Gaze_emotion" target="_blank">Learn more on GitHub</a>.</p>
    </div>
    <div class="project" data-tags="misc">
        <h3>Additional Projects</h3>
        <p>Explore more of my work on <a href="https://github.com/Mrkomiljon" target="_blank">GitHub</a>.</p>
    </div>
</div>

<script>
document.addEventListener('DOMContentLoaded', function () {
    const buttons = document.querySelectorAll('.filter-button');
    const projects = document.querySelectorAll('.project');

    buttons.forEach(button => {
        button.addEventListener('click', () => {
            buttons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');

            const filter = button.getAttribute('data-filter');

            projects.forEach(project => {
                if (filter === 'all' || project.getAttribute('data-tags').includes(filter)) {
                    project.style.display = 'block';
                } else {
                    project.style.display = 'none';
                }
            });
        });
    });
});
</script>
