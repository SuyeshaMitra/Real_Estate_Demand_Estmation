---
# ☁️ AWS Deployment: Zero-Cost Novice Manager (v1.0)
---

This guide provides a professional "On-Demand" path to host the **Real Estate Demand AI Engine** entirely on AWS for **almost $0**.
We bypass expensive EC2 computers and RAG databases entirely. Instead, we use highly dense **ECS Fargate Containers**.
1. **The Database Trick**: The `pgeocode` geographic machine learning dataset is ~20MB. Instead of paying Amazon for a running SQL database, we literally freeze the entire database natively into the Docker snapshot. Local latency is 0ms, and Database cost is $0!
2. **The "1-Click" Power Manager**: To save you from manually clicking through the complex AWS Console networking layers, we have packaged the entire CloudFormation initialization, Elastic Container Registry Docker login, Image Compilation, and ECS scaling parameters into an automated bat file!

Open your terminal in the project folder. You never have to manually log into AWS again!

---

## 🚀 Phase 1: First-Time Setup & Installation

If this is your first time putting the tracker onto the internet, run this command:
```powershell
.\cloud_power_manager.bat deploy
```
* **What it does autonomously**: 
  1. Spools up the `aws_cloudformation.yaml` stack intelligently building VPC networks, load balancers, and Fargate ECS clusters correctly entirely remotely.
  2. Creates and tags internal Fargate Application structural layers natively.
  3. Binds your local Docker Desktop safely to the AWS ECR registry.
  4. Natively builds (`Image Create`) the `Python 3.10` ML container natively trapping the `pgeocode` database securely in memory!
  5. Automatically `Pushes` the snapshot to the cloud storage cleanly.

---

## ☀️ Phase 2: Daily Operations (Morning & Evening)

Because we pay an hourly rate (~$0.01/hr) to run AWS Fargate, we strategically built a simple mechanism to freeze charges when you leave the office.

### 🌅 Starting the Engine (Daily Morning)
When you arrive and need to validate the Live Engine:
```powershell
.\cloud_power_manager.bat start
```
* **Result**: Automatically powers up the Fargate container from `desired-count=0` to `desired-count=1`, connecting it directly back to the Internet Gateway flawlessly. 

### 🌙 Stopping Charges (Daily Evening)
When you finish work, put the engine to sleep:
```powershell
.\cloud_power_manager.bat stop
```
* **Result**: Powers down the massive computing engine (`desired-count=0`), immediately halting container charges. Your IP tracking maps and CloudFormation configurations are securely preserved for tomorrow at entirely $0.00!

---

## 🔄 Phase 3: Algorithm Updates (Deploying New Code)

If you modify **`07B_OSM_LightGBM_modeling.py`** (or if you change Python parameters locally) and want to deploy the updated AI explicitly to AWS without destroying the networking variables, use the Update command cleanly!
```powershell
.\cloud_power_manager.bat update
```
* **What it does autonomously**: 
  1. Instantly compiles an exact fresh Docker Application Image using your new local codebase natively!
  2. Secures AWS Authentication effectively pushing the revised Image to AWS ECR cleanly.
  3. Commands the AWS ECS Fargate cluster natively running on the internet to dynamically fetch and swap to your new code parameters flawlessly natively without dropping internet connections!

---

## 🗑️ Phase 4: Permanent Deletion (Cleanup)

If the data-engineering project completely concludes and you want to permanently strictly tear down every Amazon server instance structurally to protect your business billing accounts strictly natively:
```powershell
.\cloud_power_manager.bat cleanup
```
* **Result**: Safely permanently force formats exactly the ECR image repository explicitly preventing Docker caching locks, and then explicitly elegantly executes an aggressive deletion cascade logically successfully wiping the CloudFormation Stack absolutely clean erasing all evidence of the platform structurally natively correctly!

---

### 🏗️ Physical Cloud Deployment Architecture Matrix

```mermaid
graph TD
    classDef aws fill:#FF9900,stroke:#232F3E,stroke-width:2px,color:white;
    classDef docker fill:#2496ED,stroke:#0db7ed,stroke-width:2px,color:white;
    classDef external fill:#EEEEEE,stroke:#999999,stroke-width:2px;

    User(["User Request"]) --> IGW["Internet Gateway"]
    IGW --> VPC["AWS VPC Network"]
    
    subgraph "Zero-Cost Serverless Infrastructure (AWS Fargate)"
        VPC --> ECS["Amazon Elastic Container Service (ECS Cluster)"]
        ECS --> Service["ECS Fargate Service<br/>(AI-Engine-Service)"]
    end
    
    subgraph "Docker Application Image (real-estate-ai-engine)"
        Service --> App["Python 3.10 AI Code"]
        App --> Models[("Local Memory")]
        Models -.-> pgeocode[("pgeocode Map Database<br/>Frozen inside Docker")]
        Models -.-> ML["07B Model Vector Tracker Arrays"]
    end

    ECR["Elastic Container Registry (ECR)"] -.->|Deploys Image| ECS
    
    class IGW,VPC,ECS,Service,ECR aws;
    class App,Models,pgeocode,ML docker;
    class User external;
```
