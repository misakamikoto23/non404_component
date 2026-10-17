"""Entry point for the local review workflow, including from an uninstalled checkout."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from component_lab.review_workflow import main

if __name__=='__main__':main()
