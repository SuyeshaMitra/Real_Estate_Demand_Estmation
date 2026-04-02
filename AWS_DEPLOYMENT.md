---
# ☁️ AWS Deployment: Zero-Cost Automation (v1.0)
---

This guide provides a professional "On-Demand" path to host the **Real Estate Demand AI Engine** entirely on AWS for **almost $0**.

## 💰 The Platform Architecture Strategy

We bypass expensive EC2 computers and RAG databases entirely. Instead, we use highly dense **ECS Fargate Containers**.
1. **The Database Trick**: The `pgeocode` geographic machine learning dataset is ~20MB. Instead of paying Amazon for a running SQL database, we literally freeze the entire database into the Docker snapshot. Local latency is 0ms, and Database cost is $0!
2. **Power Manager**: To save you from manually clicking through the AWS Console, we have provided a **`cloud_power_manager.bat`** script allowing you to:
   * **Start** the engine targeting an ALB.
   * **Stop** all charges immediately.
   * **Cleanup** the platform.

---

## 🚀 Phase 1: Automated Launch

1. Log into your **AWS Console** -> **CloudFormation**.
2. Click **Create stack** -> **Upload a template file** -> **`aws_cloudformation.yaml`**.
3. **Parameters**:
   - **Environment**: Setup as `production`.
   - **CostSavingMode**: Set to `true` (Limits CPU usage heavily out-of-the-box).
4. Wait for it to hit exactly `CREATE_COMPLETE`.

---

## 📦 Phase 2: Build & Push the Logic (One-Time)

You must bundle your local ML files and the Database map perfectly into an Amazon cloud "snapshot" layer (a Docker Image).

1. Go to AWS Console -> **Elastic Container Registry (ECR)** -> Select `real-estate-ai-engine`.
2. Click the **View push commands** button.
3. Open a terminal in the root of your project where `Dockerfile` lives, and blindly execute the 4 commands Amazon gives you:
   1. `aws ecr get-login-password ... docker login`
   2. `docker build -t real-estate-ai-engine .` *(Note: This step downloads the pgeocode map locally!)*
   3. `docker tag real-estate-ai-engine:latest ...`
   4. `docker push ...`

---

## ⚡ Phase 3: Power Management Script

Open your terminal in the project folder. You never have to log into AWS again.

### **A. Start The Platform**
```powershell
./cloud_power_manager.bat start
```
* **Result**: Powers up the Fargate container, connects it to the Internet Gateway, and begins ML processing requests. 
* **Cost**: Matches 70% cheaper Fargate Spot rates (~$0.01 per hour).

### **B. Stop Charging (Zero-Cost Sleep)**
```powershell
./cloud_power_manager.bat stop
```
* **Result**: Powers down the massive computing engine, saving your settings totally natively.
* **COST: $0.00**. Run this blindly whenever you finish work!

### **C. Permanent Cleanup**
```powershell
./cloud_power_manager.bat cleanup
```
* **Result**: Force formats the ECR image repository and executes a deletion cascade of the CloudFormation Stack. Use this when the real estate project concludes permanently.

---

## 🔄 Updating Algorithms

If you edit `07_Features_LightGBM_modeling.py` internally locally, follow this strictly to jump the changes safely to the internet cloud:
1. Repeat the **ECR Push Commands** exactly as Step 2 (rebuilding the container locally).
2. Go to **AWS ECS** -> **Clusters** -> `RealEstateDemandCluster`.
3. Select your Service (`AI-Engine-Service`) -> Click **Update**.
4. Check **Force new deployment** natively -> Click **Update**.
AWS handles draining traffic automatically, swapping the active ML engine without generating 502 errors!
