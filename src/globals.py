"""
Global objects and utilities for the data science toolkit.

This module provides globally accessible objects like loggers that can be
imported and used across all modules in the project.
"""

# create a logger object to be global in all modules
from src.logger import ExecutorLogger

logger = ExecutorLogger("globals")
