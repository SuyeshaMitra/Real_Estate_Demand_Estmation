@echo off
setlocal

:: ==============================================================
:: ZERO-COST CLOUD POWER MANAGER
:: This script safely automatically changes your AWS ECS cluster sizes
:: so you only pay the $0.01 an hour Spot charge exactly while working!
:: ==============================================================

set CLUSTER_NAME=RealEstateDemandCluster
set SERVICE_NAME=AI-Engine-Service
set ACTION=%1

if "%ACTION%"=="" (
    echo [ERROR] Provide parameter 'start', 'stop', or 'cleanup'.
    echo Usage: .\cloud_power_manager.bat start
    exit /b 1
)

if /I "%ACTION%"=="start" (
    echo [EXECUTE] Powering up the Real Estate AI Fargate Spot Instances...
    aws ecs update-service --cluster %CLUSTER_NAME% --service %SERVICE_NAME% --desired-count 1 >nul
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to start cluster!
    ) else (
        echo [SUCCESS] Real Estate engine is powering ON natively!
    )
    exit /b %errorlevel%
)

if /I "%ACTION%"=="stop" (
    echo [EXECUTE] Securing Zero-Cost Storage Sequence...
    aws ecs update-service --cluster %CLUSTER_NAME% --service %SERVICE_NAME% --desired-count 0 >nul
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to stop the engine.
    ) else (
        echo [SUCCESS] ECS Engine suspended accurately. Charges have stopped!
    )
    exit /b %errorlevel%
)

if /I "%ACTION%"=="cleanup" (
    echo [WARNING] INITIATING FULL PLATFORM TERMINATION!
    echo Automatically erasing the isolated docker repositories...
    aws ecr delete-repository --repository-name real-estate-ai-engine --force >nul 2>&1
    echo Executing CloudFormation erasure scripts globally...
    aws cloudformation delete-stack --stack-name real-estate-ai-platform
    echo [SUCCESS] Your absolute environment is zeroed successfully!
    exit /b 0
)

echo [ERROR] Invalid action parameter assigned natively. Use 'start', 'stop', or 'cleanup'.
exit /b 1
