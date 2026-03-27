---
layout: archive
title: "CV"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}

About Me
======
I'm Pengbin Li, a master's student at Iwate University (Japan) with a background in Information and Computing Science. My undergraduate studies covered a broad CS foundation including Java/C/C++/Python programming, data structures, algorithm design, operating systems, information security, and software engineering. During my master's program, I've focused on computer vision, computer network systems, and image synthesis.

My current research centers on **GAN-based intrusion detection systems** — using Generative Adversarial Networks to generate synthetic network traffic data and fusing it with real-world data for IDS training. The goal is to improve the generalization capability of intrusion detection models and strengthen their defense against adversarial attacks.

Outside of research, I enjoy vibe coding, CTF challenges, basketball, and gaming.

Education
======
**Iwate University (Japan)** <span style="float:right;">2024.10 – Present</span>
* Master of Design and Media Engineering
* GPA: 3.2/4.0
* Coursework: Computer Vision, Computer Network Systems, Computer Animation, Design Representation, Image Synthesis
* Research Focus: Intrusion detection systems using Generative Adversarial Networks. Generating synthetic traffic data via GANs and combining it with real data for IDS training to improve generalization and adversarial attack defense.

**Central South University of Forestry and Technology (China)** <span style="float:right;">2019.09 – 2023.06</span>
* B.Sc. in Information and Computing Science
* GPA: 3.0/5.0
* Coursework: Java/C/C++/Python Programming, MySQL, Computer Organization, Operating Systems, Algorithm Design, Data Structures, Information Security, Big Data Visualization, Software Engineering

Work Experience
======
**Hunan Kechuang Information Technology Co., Ltd. — Java Development Engineer** <span style="float:right;">2023.07 – 2023.11</span>

Built the backend for Changsha's municipal procurement system using Java, Spring Boot, and MySQL. Created a bid document parser that handled multiple formats and extracted searchable data. Also involved in requirements analysis, testing, and troubleshooting.

Research Interests
======
* **Computer Vision:** Object detection, image segmentation, feature extraction
* **Deep Learning:** Transformer architectures, Generative Adversarial Networks, multimodal learning
* **Applications:** Network data analysis, intelligent transportation, remote sensing imagery

Technical Skills
======
* **Programming Languages:** Python, Java, Go, C/C++
* **Deep Learning:** PyTorch, Scikit-learn, HuggingFace Transformers, Keras
* **Computer Vision:** OpenCV, image segmentation, object detection, feature engineering, ViT, CNN
* **Data Processing:** PCA, ANOVA feature selection, data balancing (SMOTE, GAN), time series analysis

Project Experience
======
  <ul>{% for post in site.projects reversed %}
    {% include project-experience-cv.html %}
  {% endfor %}</ul>

Personal Statement
======
* **Research Focus:** My graduate work focused on deep learning for cybersecurity, specifically how to make generative and discriminative models work better in intrusion detection systems.
* **What Problems I Care About:** I'm interested in the messy parts: imbalanced data, weak feature representations, models that don't generalize well. I've gone from defining these problems to validating solutions end-to-end.
* **Skills:** I can run the full research cycle myself — literature review, designing methods, running experiments, synthesizing results. I think carefully about methodology.
* **What I'm Learning:** I follow AI developments closely, especially LLMs and Agent systems. I'm exploring how they might apply to security and automation.
* **Broader Interests:** The intersection of AI with cybersecurity, systems engineering, and cognitive intelligence fascinates me. I enjoy connecting ideas across fields.
* **PhD Goals:** I want to study how intelligent models behave in complex, real environments — specifically their reliability, generalization, and interpretability. The goal is trustworthy AI for safety-critical systems.
