# Phishing Website Detection Using Machine Learning

## 🎯 Objective:
To develop a machine learning model that accurately classifies websites as phishing or legitimate based on various extracted features from URLs and web content.

## 📚 Problem Statement:
Phishing is a major cybersecurity threat where attackers trick users into revealing sensitive information by mimicking trustworthy websites. Manual detection is impractical at scale, hence the need for an automated classification model.

## 🧾 Dataset Description:
Each row represents a website with various features extracted from its URL and HTML structure. The target column is Result:

-   1: Legitimate

-   -1: Phishing

## ✅ Features Include:
    Presence of IP address (having_IP_Address)

    Use of shortening services (Shortining_Service)

    Presence of “@” symbol (having_At_Symbol)

    Double slashes in redirection (double_slash_redirecting)

    HTTPS token usage (HTTPS_token)

    SSL certificate status (SSLfinal_State)

    Age of domain (age_of_domain)

    Web traffic, Google indexing, etc.



## Project Structure

```
└── network-security-project
    └── .github
        └── workflows
            └── main.yml
    └── network_data
    └── networksecurity
        └── __init__.py
        └── cloud
            └── __init__.py
        └── components
            └── __init__.py
        └── constant
            └── __init__.py
        └── entity
            └── __init__.py
        └── exception
            └── __init__.py
        └── logging
            └── __init__.py
        └── pipelines
            └── __init__.py
        └── utils
            └── __init__.py
    └── notebooks
    └── .env
    └── .gitignore
    └── Dockerfile
    └── README.md
    └── requirements.txt
    └── setup.py
```