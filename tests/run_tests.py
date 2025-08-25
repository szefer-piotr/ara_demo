#!/usr/bin/env python3
"""
Test runner script for the bioarxiv pipeline.
Provides different test execution modes and reporting options.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with exit code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run bioarxiv pipeline tests")
    parser.add_argument(
        "--mode", 
        choices=["all", "unit", "integration", "download", "chunking", "evaluation", "search"],
        default="all",
        help="Test mode to run"
    )
    parser.add_argument(
        "--coverage", 
        action="store_true",
        help="Generate coverage reports"
    )
    parser.add_argument(
        "--html", 
        action="store_true",
        help="Generate HTML test report"
    )
    parser.add_argument(
        "--parallel", 
        action="store_true",
        help="Run tests in parallel"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop on first test failure"
    )
    
    args = parser.parse_args()
    
    # Base pytest command
    base_cmd = ["python", "-m", "pytest"]
    
    # Add options based on arguments
    if args.verbose:
        base_cmd.append("-v")
    
    if args.stop_on_failure:
        base_cmd.append("-x")
    
    if args.parallel:
        base_cmd.extend(["-n", "auto"])
    
    if args.coverage:
        base_cmd.extend(["--cov=.", "--cov-report=html:htmlcov", "--cov-report=term-missing"])
    
    if args.html:
        base_cmd.extend(["--html=test-report.html", "--self-contained-html"])
    
    # Add test selection based on mode
    if args.mode == "unit":
        base_cmd.extend(["-m", "unit"])
    elif args.mode == "integration":
        base_cmd.extend(["-m", "integration"])
    elif args.mode == "download":
        base_cmd.extend(["-m", "download"])
    elif args.mode == "chunking":
        base_cmd.extend(["-m", "chunking"])
    elif args.mode == "evaluation":
        base_cmd.extend(["-m", "evaluation"])
    elif args.mode == "search":
        base_cmd.extend(["-m", "search"])
    
    # Add test discovery
    base_cmd.append("tests/")
    
    # Run the tests
    success = run_command(base_cmd, f"Running {args.mode} tests")
    
    if success:
        print("\n🎉 All tests passed!")
        
        if args.coverage:
            print("\n📊 Coverage reports generated:")
            print("   - HTML: htmlcov/index.html")
            print("   - Terminal: See above output")
        
        if args.html:
            print("\n📄 HTML test report generated: test-report.html")
        
        return 0
    else:
        print("\n💥 Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

