# ScoutAI Complete Execution Pipeline (Updated Architecture)

This flowchart represents the newly upgraded end-to-end technical life cycle of a ScoutAI video analysis. It highlights the major performance upgrades: **GPU Batching**, **One Euro Filter** for poses, and the **Physics Ball Velocity Tracker** for accurate touches.

```mermaid
flowchart TD
    classDef init fill:#e1f5fe,stroke:#0277bd,stroke-width:2px,color:black,font-weight:bold;
    classDef loop_control fill:#ffe0b2,stroke:#ef6c00,stroke-width:2px,color:black,font-weight:bold;
    classDef batch_gpu fill:#ffcdd2,stroke:#c62828,stroke-width:3px,color:black,font-weight:bold,stroke-dasharray: 5 5;
    classDef tracking fill:#d1c4e9,stroke:#4527a0,stroke-width:2px,color:black;
    classDef physics fill:#d9b3ff,stroke:#6600cc,stroke-width:3px,color:black,font-weight:bold;
    classDef logic fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px,color:black;
    classDef drawing fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,color:black;
    classDef export fill:#cfd8dc,stroke:#37474f,stroke-width:2px,color:black,font-weight:bold;
    classDef startend fill:#333,stroke:#fff,stroke-width:4px,color:#fff,font-weight:bold;

    Start((Start Analysis)):::startend --> Phase1

    %% ==========================================
    %% PHASE 1: INITIALIZATION
    %% ==========================================
    subgraph Phase1 [Phase 1: Initialization & Loading]
        direction TB
        A1[Load Video File]:::init --> A2[Load Ultralytics Models]:::init
        A2 --> A3[Initialize Components:<br/>One Euro Smoother, BallVelocityTracker]:::init
        A3 --> A4[Start Async Video Writer Thread]:::export
    end

    Phase1 --> Phase2

    %% ==========================================
    %% PHASE 2: BATCH PROCESSING LOOP
    %% ==========================================
    subgraph Phase2 [Phase 2: Batched Processing Loop]
        direction TB
        
        FrameCheck{Read Next<br/>Batch of N Frames?}:::loop_control
        
        FrameCheck -- Yes --> BatchInfer
        
        %% --- BATCH GPU MUX ---
        subgraph BatchInfer [GPU Batch Inference Phase]
            direction LR
            ObjDetect[YOLO Object Detection<br/>Batch of N Frames]:::batch_gpu
            PoseDetect[YOLO Pose Estimation<br/>Batch of N Frames]:::batch_gpu
        end
        
        BatchInfer --> SeqLoop
        
        %% --- SEQUENTIAL CPU LOGIC ---
        subgraph SeqLoop [Per-Frame Sequential Post-Processing]
            direction TB
            
            ObjTrack[Update Stable Object Tracker<br/>Map BoT-SORT to IDs]:::tracking
            
            Euro[Apply One Euro Filter<br/>Adaptive Anti-Jitter Smoothing]:::physics
            
            Calib[Live Calibration<br/>Update Pixels-per-Meter]:::logic
            
            BallPhys[BallVelocityTracker<br/>Compute Δv & Direction Change]:::physics
            
            DrillMetrics[[compute_drill_metrics<br/>Proximity + Velocity Touch Logic]]:::logic
            
            Visuals[Build Overlay & Draw Bounding Boxes]:::drawing
            
            ObjTrack --> Euro --> Calib --> BallPhys --> DrillMetrics --> Visuals
        end
        
        SeqLoop --> WorkerQueue[(Queue N Frames to<br/>Writer Thread)]:::export
        WorkerQueue -. Loop back .-> FrameCheck
    end

    %% ==========================================
    %% PHASE 3: FINALIZATION
    %% ==========================================
    FrameCheck -- "No (EOF)" --> Phase3

    subgraph Phase3 [Phase 3: Teardown & Export]
        direction TB
        F1[Wait for Writer Thread<br/>to empty queue]:::export --> F2[generate_report<br/>Compile Final JSON Dict]:::logic
        F2 --> F3[Save JSON Report to Disk]:::export
        F3 --> F4[FFmpeg Video Compression<br/>Reduce MP4 file size]:::export
    end

    Phase3 --> End((Finish Analysis)):::startend
```
