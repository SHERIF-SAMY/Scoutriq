# ScoutAI Pipeline Architecture

This document illustrates the evolution of the ScoutAI processing pipeline, showing the upgrades from the initial design to the current production system.

## 1. Legacy Pipeline (v0.x — Replaced)

The original system used simple coordinate bounds, 2D Euclidean distances, and a fixed-weight Exponential Moving Average (EMA) smoother. This led to jittery poses and frequent false-positive touch detections.

```mermaid
flowchart TD
    classDef model fill:#ff9999,stroke:#333,stroke-width:2px,color:black,font-weight:bold;
    classDef track fill:#ffcc99,stroke:#333,stroke-width:2px,color:black,font-weight:bold;
    classDef math fill:#99ccff,stroke:#333,stroke-width:2px,color:black;
    classDef logic fill:#ffffcc,stroke:#333,stroke-width:2px,color:black;
    classDef target fill:#ccc,stroke:#333,stroke-width:3px,color:black,font-weight:bold;

    Start((Start Frame)) --> Obj[YOLO Object Detection]:::model
    Obj --> Tracker[BoT-SORT & Stable Tracker]:::track
    
    Tracker --> Pose[YOLO Pose Estimation]:::model
    Pose --> Smooth[Simple EMA Smoother]:::math
    
    Tracker --> Post[Calibration & Distances]:::math
    Smooth --> Post
    
    Post --> Logic{Touch Logic:<br/>2D Distance < Threshold ?}:::logic
    Logic -- Yes --> Touch[Register Touch]:::target
    Logic -- No --> Skip[Ignore]
    
    Touch --> Overlay[Draw Metrics & Overlay]
    Skip --> Overlay
    Overlay --> End((Next Frame))
```

**Problems with this approach:**
- EMA smoother applied equal smoothing regardless of movement speed → lag during fast motion
- Touch detection relied solely on proximity → false positives when ball is near foot without contact
- Single-frame inference → underutilized GPU

---

## 2. Current Pipeline (v1.x — Production)

The current system separates **Perception** (GPU inference) from **Reasoning** (CPU post-processing). Key upgrades: GPU batching, One Euro Filter, and physics-based ball velocity tracking.

```mermaid
flowchart TD
    classDef model fill:#ff9999,stroke:#333,stroke-width:2px,color:black,font-weight:bold;
    classDef track fill:#ffcc99,stroke:#333,stroke-width:2px,color:black,font-weight:bold;
    classDef math fill:#99ccff,stroke:#333,stroke-width:2px,color:black;
    classDef logic fill:#ffffcc,stroke:#333,stroke-width:2px,color:black;
    classDef target fill:#ccc,stroke:#333,stroke-width:3px,color:black,font-weight:bold;
    classDef upgrade fill:#d9b3ff,stroke:#6600cc,stroke-width:3px,stroke-dasharray: 5 5,color:black,font-weight:bold;

    Start((Start Frame)) --> Obj[YOLO Object Detection]:::model
    Obj --> Tracker[BoT-SORT & Stable Tracker]:::track
    
    Tracker --> Pose[YOLO Pose Estimation]:::model
    
    Tracker --> BallVelocity[BallVelocityTracker<br/>Compute Δv & Direction Change]:::upgrade
    
    Pose --> EuroFilter[One Euro Filter<br/>Adaptive Smoothing]:::upgrade
    
    EuroFilter --> Post[Calibration & Coordinates]:::math
    BallVelocity --> Post
    
    Post --> Logic{Touch Logic:<br/>1. Proximity < Threshold<br/>AND<br/>2. Velocity Spike Δv > Threshold}:::upgrade
    
    Logic -- Yes --> Touch[Register Touch]:::target
    Logic -- No --> Skip[Ignore]
    
    Touch --> Overlay[Draw Metrics & Overlay]
    Skip --> Overlay
    Overlay --> End((Next Frame))
```

**What changed:**

| Component | Before | After |
|---|---|---|
| Pose smoothing | Fixed-weight EMA | One Euro Filter (speed-adaptive) |
| Touch detection | Proximity only | Proximity + velocity spike + direction change |
| GPU inference | 1 frame at a time | Batched (N frames) |
| Precision | FP32 | FP16 (half) |
| Video writing | Synchronous | Async thread with blocking Queue |
| Overlay blending | Full frame copy | ROI-only copy |
