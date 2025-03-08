#!/bin/bash

# Script to generate Doxygen documentation for the sparse matrix library

# Check if Doxyfile exists
if [ ! -f "Doxyfile" ]; then
    echo "Error: Doxyfile not found. Please create it with 'doxygen -g' and configure it."
    exit 1
fi

# Run Doxygen to generate documentation
echo "Generating Doxygen documentation..."
doxygen Doxyfile

# Check if documentation was generated successfully
if [ $? -eq 0 ]; then
    echo "Documentation generated successfully in the 'docs' directory."
    echo "Open 'docs/html/index.html' in a browser to view it."
else
    echo "Error: Doxygen failed to generate documentation."
    exit 1
fi

# Optionally, open the documentation in a browser (uncomment if desired)
# xdg-open docs/html/index.html
chromium docs/html/index.html
