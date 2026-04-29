"""
ScoutAI — Production-ready sports drill analysis framework.

Provides a template-method architecture for analyzing player performance
in various football drills from monocular video.

Usage:
    from Scoutriq_Vision.drills.seven_cone import SevenConeDrillAnalyzer
    from Scoutriq_Vision.config import DrillConfig

    config = DrillConfig.from_yaml("configs/seven_cone.yaml")
    analyzer = SevenConeDrillAnalyzer(config)
    report = analyzer.run("video.mp4", "output/")
"""

__version__ = "1.0.0"
