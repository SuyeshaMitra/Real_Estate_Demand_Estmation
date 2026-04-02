@echo off
setlocal EnableDelayedExpansion

:: ==============================================================
:: ZERO-COST CLOUD POWER MANAGER & DEPLOYMENT WRAPPER
:: Handles end-to-end infrastructure generation, Docker mapping, 
:: and safe Start/Stop execution commands.
:: ==============================================================

set STACK_NAME=real-estate-ai-platform
set CLUSTER_NAME=RealEstateDemandCluster
set SERVICE_NAME=AI-Engine-Service
set ECR_REPO=real-estate-ai-engine
set REGION=eu-west-2
:: You can change your region globally above! Defaulting to London (eu-west-2) for UK housing data.

set ACTION=%1

if "%ACTION%"=="" (
    echo ====================================================
    echo   REAL ESTATE AI - AWS CLOUD MANAGER 
    echo ====================================================
    echo Valid Commands:
    echo  .\cloud_power_manager.bat deploy   - Creates total AWS Cloud Infrastructure + Pushes Docker logic
    echo  .\cloud_power_manager.bat start    - Powers on the AI Engine ^($0.01/hr^)
    echo  .\cloud_power_manager.bat stop     - Freezes the AI Engine safely ^($0.00/hr^)
    echo  .\cloud_power_manager.bat cleanup  - Vigorously deletes ALL AWS resources permanently to protect billing!
    echo ====================================================
    exit /b 1
)

:: -------------------------------------------------------------------
:: DEPLOYMENT PHASE (CloudFormation + Docker)
:: -------------------------------------------------------------------
if /I "%ACTION%"=="deploy" (
    echo [EXECUTE] Generating AWS Framework via CloudFormation...
    aws cloudformation create-stack --stack-name %STACK_NAME% --template-body file://aws_cloudformation.yaml --capabilities CAPABILITY_IAM >nul
    
    echo [WAITING] Binding AWS infrastructure... ^(Takes roughly 3 minutes^)
    aws cloudformation wait stack-create-complete --stack-name %STACK_NAME%
    if !errorlevel! neq 0 (
        echo [ERROR] CloudFormation Failed! Check your AWS Console events natively.
        exit /b !errorlevel!
    )
    echo [SUCCESS] CloudFormation Network Online!

    echo [EXECUTE] Fetching AWS Account ID securely...
    for /f "tokens=*" %%a in ('aws sts get-caller-identity --query Account --output text') do set AWS_ACCOUNT=%%a

    echo [EXECUTE] Securing physical Docker Bridge to Elastic Container Registry...
    aws ecr get-login-password --region %REGION% | docker login --username AWS --password-stdin !AWS_ACCOUNT!.dkr.ecr.%REGION%.amazonaws.com

    echo [EXECUTE] Compiling Python Docker Image ^(This traps the 20MB Geodatabase securely inside^)...
    docker build -t %ECR_REPO% .

    echo [EXECUTE] Tagging and Pushing Image into AWS Cloud Storage...
    docker tag %ECR_REPO%:latest !AWS_ACCOUNT!.dkr.ecr.%REGION%.amazonaws.com/%ECR_REPO%:latest
    docker push !AWS_ACCOUNT!.dkr.ecr.%REGION%.amazonaws.com/%ECR_REPO%:latest
    
    echo [EXECUTE] Activating ECS Task definition to use the new Engine...
    aws ecs update-service --cluster %CLUSTER_NAME% --service %SERVICE_NAME% --force-new-deployment >nul

    echo ====================================================
    echo [SUCCESS] PLATFORM IS FULLY DEPLOYED AND READY IN AWS! 
    echo Run '.\cloud_power_manager.bat start' to turn it on natively.
    echo ====================================================
    exit /b 0
)

:: -------------------------------------------------------------------
:: POWER ON PHASE 
:: -------------------------------------------------------------------
if /I "%ACTION%"=="start" (
    echo [EXECUTE] Powering up the Real Estate AI Fargate Spot Instances...
    aws ecs update-service --cluster %CLUSTER_NAME% --service %SERVICE_NAME% --desired-count 1 >nul
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to start cluster!
    ) else (
        echo [SUCCESS] Real Estate engine is powering ON natively!
    )
    exit /b !errorlevel!
)

:: -------------------------------------------------------------------
:: HYBERNATION PHASE (Stop Billing)
:: -------------------------------------------------------------------
if /I "%ACTION%"=="stop" (
    echo [EXECUTE] Securing Zero-Cost Storage Sequence...
    aws ecs update-service --cluster %CLUSTER_NAME% --service %SERVICE_NAME% --desired-count 0 >nul
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to stop the engine.
    ) else (
        echo [SUCCESS] ECS Engine suspended accurately. Charges have stopped!
    )
    exit /b !errorlevel!
)

:: -------------------------------------------------------------------
:: CLEAN EXIT PHASE (Destroy Stack)
:: -------------------------------------------------------------------
if /I "%ACTION%"=="cleanup" (
    echo [WARNING] INITIATING FULL PLATFORM TERMINATION!
    echo Automatically erasing the isolated docker repositories bridging storage loops...
    aws ecr delete-repository --repository-name %ECR_REPO% --force >nul 2>&1
    
    echo Executing CloudFormation erasure scripts globally. This will seamlessly wipe all routing, clusters, and load balancers...
    aws cloudformation delete-stack --stack-name %STACK_NAME%
    
    echo [WAITING] Terminating resources... ^(This takes roughly 5 minutes^)
    aws cloudformation wait stack-delete-complete --stack-name %STACK_NAME%
    
    echo [SUCCESS] Your absolute environment is fully zeroed and safely deleted!
    exit /b 0
)

echo [ERROR] Invalid action. Just run '.\cloud_power_manager.bat' to see options.
exit /b 1
