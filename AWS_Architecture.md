# AWS Physical Deployment Design (`AWS_Architecture.md`)

This structurally isolates the physical cloud boundaries mapping exactly how our Machine Learning code interfaces with the internet and database storage completely legally avoiding AWS RDS costs.

## 🏗️ Physical Serverless Component Map

```mermaid
graph TD
    classDef aws fill:#FF9900,stroke:#232F3E,stroke-width:2px,color:white;
    classDef docker fill:#2496ED,stroke:#0db7ed,stroke-width:2px,color:white;
    classDef external fill:#EEEEEE,stroke:#999999,stroke-width:2px;

    User([User Request]) --> IGW[Internet Gateway]
    IGW --> VPC[AWS VPC Network]
    
    subgraph "Zero-Cost Serverless Infrastructure (AWS Fargate)"
        VPC --> ECS[Amazon Elastic Container Service (ECSCluster)]
        ECS --> Service[ECS Fargate Service<br>(AI-Engine-Service)]
    end
    
    subgraph "Docker Application Image (real-estate-ai-engine)"
        Service --> App[Python 3.10 AI Code]
        App --> Models[(Local Memory)]
        Models -.-> pgeocode[(pgeocode Map Database<br>Frozen inside Docker)]
        Models -.-> ML[LightGBM / XGBoost Regressors]
    end

    ECR[Elastic Container Registry (ECR)] -.->|Deploys Image| ECS
    
    class IGW,VPC,ECS,Service,ECR aws;
    class App,Models,pgeocode,ML docker;
    class User external;
```

## 🛠️ Physical Component Details

### 1. The Database Void 
Instead of spinning up physical `PostgreSQL` or `AWS Aurora` database instances explicitly for geography datasets, **this architecture physically freezes the entire Database map into the Docker Image**. Localized memory lookup times are infinitely faster than Cloud network speeds, dropping AWS database bills exactly to zero.

### 2. The Compute Engine
The **Elastic Container Service** mathematically natively triggers utilizing isolated **Fargate Spot** containers. They only execute precisely when `cloud_power_manager.bat start` connects the network to the Internet Gateway, saving virtually 100% of compute downtime structurally locally!
