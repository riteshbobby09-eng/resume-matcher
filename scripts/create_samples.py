"""
Create sample JD and resume PDFs for testing the pipeline.
"""

import pymupdf
from pathlib import Path


def create_sample_jd(output_path: str = "data/sample_jd.pdf") -> Path:
    """Create a sample Job Description PDF."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    jd_text = """Senior Software Engineer - Backend

About Us
We are a fast-growing fintech company building next-generation payment solutions 
for businesses worldwide. Our platform processes millions of transactions daily.

Role Overview
We are looking for a Senior Software Engineer to join our backend team. 
You will design, build, and maintain scalable microservices.

Requirements
Must have 5+ years of experience in software development.
Required: Strong proficiency in Python and at least one other language (Java, Go, or Rust).
Experience with cloud platforms (AWS or GCP) is essential.
Must have experience with microservices architecture and containerization (Docker, Kubernetes).
Required: Experience with relational databases (PostgreSQL) and caching (Redis).
Knowledge of event-driven architecture (Kafka, RabbitMQ) is required.
Must have experience with CI/CD pipelines.

Responsibilities
Design and implement scalable, high-performance backend services.
Lead code reviews and enforce coding standards across the team.
Collaborate with product and frontend teams to define API contracts.
Mentor junior engineers and contribute to technical hiring.
Drive continuous improvement in development practices and tooling.
Monitor and optimize service performance and reliability.

Skills Required
Python, Java or Go, SQL, Docker, Kubernetes
Cloud platforms: AWS or GCP
CI/CD: Jenkins, GitHub Actions, ArgoCD
Databases: PostgreSQL, MongoDB, Redis
Message queues: Apache Kafka, RabbitMQ
Monitoring: Prometheus, Grafana, ELK Stack

Education
Bachelor's degree in Computer Science, Engineering, or related field.
Master's degree is a plus.

Nice to Have
Experience with GraphQL and gRPC.
Contributions to open-source projects.
Experience with distributed tracing (Jaeger, Zipkin).
Knowledge of machine learning deployment pipelines."""

    doc = pymupdf.open()
    
    lines = jd_text.strip().split("\n")
    page = doc.new_page()
    y_pos = 60
    
    for line in lines:
        if y_pos > 760:
            page = doc.new_page()
            y_pos = 60
        
        text = line.strip()
        if not text:
            y_pos += 10
            continue
            
        # Check if heading (ALL CAPS or known section)
        fontsize = 10
        if text.isupper() or text in [
            "About Us", "Role Overview", "Requirements", "Responsibilities",
            "Skills Required", "Education", "Nice to Have",
            "Senior Software Engineer - Backend"
        ]:
            fontsize = 12
        
        page.insert_text(pymupdf.Point(50, y_pos), text, fontsize=fontsize)
        y_pos += 16

    doc.save(str(path))
    doc.close()
    print(f"✅ Created sample JD: {path}")
    return path


def create_sample_resume(output_path: str = "data/sample_resume.pdf") -> Path:
    """Create a sample resume PDF."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    resume_text = """Rahul Sharma
Email: rahul.sharma@email.com | Phone: +91-9876543210 | LinkedIn: linkedin.com/in/rahulsharma

PROFESSIONAL SUMMARY
Senior Software Engineer with 6+ years of experience in building scalable backend systems.
Expert in Python, Java, cloud-native development, and microservices architecture.
Passionate about clean code, system design, and mentoring teams.

TECHNICAL SKILLS
Languages: Python, Java, Go, SQL, JavaScript
Frameworks: FastAPI, Flask, Spring Boot, Django
Cloud: AWS (EC2, S3, Lambda, ECS, RDS), GCP (GKE, BigQuery)
Containers: Docker, Kubernetes, Helm
Databases: PostgreSQL, MongoDB, Redis, Elasticsearch
Message Queues: Apache Kafka, RabbitMQ, AWS SQS
CI/CD: Jenkins, GitHub Actions, ArgoCD, Terraform
Monitoring: Prometheus, Grafana, ELK Stack, Datadog
Tools: Git, Jira, Confluence, Postman

WORK EXPERIENCE
Senior Software Engineer | PayTech Solutions | 2021 - Present
• Designed and built a high-throughput payment processing microservice handling 50K+ transactions per second using Python and Kafka
• Led migration from monolith to microservices architecture, reducing deployment time by 70%
• Implemented event-driven architecture using Apache Kafka for real-time transaction processing
• Set up comprehensive CI/CD pipelines using GitHub Actions and ArgoCD
• Mentored team of 4 junior developers through code reviews and pair programming
• Achieved 99.99% uptime for critical payment services through automated monitoring and alerting
• Designed and implemented PostgreSQL database schemas optimized for high-write workloads
• Deployed services on Kubernetes (AWS EKS) with auto-scaling and health checks

Software Engineer | DataFlow Inc. | 2019 - 2021
• Built RESTful APIs using Python FastAPI serving 2M+ daily active users
• Implemented Redis caching layer reducing API response time by 60%
• Developed data pipeline for ETL processing using Apache Kafka and Python
• Worked in Agile environment with 2-week sprints and daily standups
• Created automated test suites achieving 90%+ code coverage with pytest
• Contributed to system design discussions and technical documentation

Junior Software Engineer | WebSoft Technologies | 2018 - 2019
• Developed backend APIs using Python Flask and Java Spring Boot
• Worked with PostgreSQL and MongoDB for data storage
• Participated in code reviews and followed coding standards
• Deployed applications using Docker containers on AWS EC2

EDUCATION
Bachelor of Technology in Computer Science and Engineering
Indian Institute of Technology (IIT), Kanpur | 2014 - 2018
CGPA: 8.5/10.0

CERTIFICATIONS
AWS Solutions Architect - Associate (2022)
Google Cloud Professional Cloud Developer (2023)
Certified Kubernetes Administrator (CKA) (2023)

PROJECTS
• Open-source contributor to FastAPI framework - Added WebSocket improvements
• Built a real-time fraud detection system processing 10K events/sec using Kafka Streams
• Developed a CLI tool for automated database migrations in Python

ACHIEVEMENTS
• Received "Star Performer" award at PayTech Solutions for Q3 2023
• Speaker at PyCon India 2022 on "Building High-Performance APIs with FastAPI"
• Published technical blog with 10K+ monthly readers on backend engineering topics"""

    doc = pymupdf.open()
    
    lines = resume_text.strip().split("\n")
    page = doc.new_page()
    y_pos = 50
    
    for line in lines:
        if y_pos > 770:
            page = doc.new_page()
            y_pos = 50
        
        text = line.strip()
        if not text:
            y_pos += 8
            continue
        
        fontsize = 10
        if text.isupper() and len(text) > 3:
            fontsize = 11
        elif text == lines[0].strip():
            fontsize = 14
        
        page.insert_text(pymupdf.Point(50, y_pos), text, fontsize=fontsize)
        y_pos += 14

    doc.save(str(path))
    doc.close()
    print(f"✅ Created sample resume: {path}")
    return path


if __name__ == "__main__":
    create_sample_jd()
    create_sample_resume()
    print("\n✅ Sample files created in data/ directory")
