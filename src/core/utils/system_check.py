import sys
import shutil
import logging

def check_python_version(min_version=(3, 7)):
    """
    Check if the current Python version meets the minimum requirement.
    """
    if sys.version_info < min_version:
        raise EnvironmentError(f"Python {min_version[0]}.{min_version[1]} or higher is required.")

def check_dependency(tool_name):
    """
    Check if a given tool is available in the system path.
    """
    if shutil.which(tool_name) is None:
        raise EnvironmentError(f"{tool_name} is not installed or not found in system PATH.")

def perform_system_checks():
    """
    Perform all system checks.
    """
    try:
        check_python_version()
        check_dependency('git')  # Example: Checking for git
        logging.getLogger(__name__).info("System checks passed.")
    except EnvironmentError as e:
        logging.getLogger(__name__).error(f"System check failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    perform_system_checks()
