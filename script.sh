#!/bin/bash

# Cache file path
CACHE_FILE=".module_cache"

# Activate virtual environment
VENV_PATH="venv/bin/activate"

if [ -f "$VENV_PATH" ]; then
    if [ -z "$VENV_ACTIVATED" ]; then  # Check if already activated
        source "$VENV_PATH"
        export VENV_ACTIVATED=true
        echo "Virtual environment activated"
    else
        echo "Virtual environment already activated"
    fi
else
    echo "Virtual environment not found"
    exit 1
fi

# Function to check if a module is installed and cache the result
check_module() {
    module_name=$1
    
    # Check the cache first
    if grep -q "$module_name" "$CACHE_FILE" 2> /dev/null; then
        echo "Module $module_name already cached."
        return
    fi

    # Check if the module is installed
    python -c "import $module_name" 2> /dev/null

    if [ $? -eq 0 ]; then
        version=$(python -c "import $module_name; print($module_name.__version__)")
        echo "Module $module_name found, version: $version"
        echo "$module_name" >> "$CACHE_FILE"  # Cache the result
    else
        echo "Module $module_name not found, attempting to install."
        pip install $module_name

        if [ $? -eq 0 ]; then
            echo "$module_name" >> "$CACHE_FILE"  # Cache after successful installation
            check_module $module_name  # Verify installation
        else
            echo "Module $module_name could not be installed"
            exit 1
        fi
    fi
}

# Initialize the cache file if it doesn't exist
if [ ! -f "$CACHE_FILE" ]; then
    touch "$CACHE_FILE"
fi

check_module numpy
check_module tensorflow
check_module keras
check_module streamlit
check_module opencv-python
check_module watchdog

dir="ml-service/app"
file="app.py"

(cd "$(pwd)/$dir" && streamlit run "$file")