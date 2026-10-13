#!/usr/bin/env python3
"""
CLI script to generate academic markdown report and research findings.
"""

from src.analysis import generate_research_summary_report

def main():
    report = generate_research_summary_report()
    print(report)

if __name__ == "__main__":
    main()
