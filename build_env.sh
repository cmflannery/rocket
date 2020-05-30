#!/bin/bash
#
# This is a simple script that performs automates some simple things required
# for developing design_optimization
echo 'Running build_env.sh'
echo ''


# Create a python 3 virtual environment
# If a virtual environment already exists, delete it.
if [ -d './venv' ]
then
    echo 'A virtual environment already exists. Deleting and starting from scratch.'
    rm -rf venv
    rm -rf $(find ./ -name '*.egg-info')
fi
echo 'Creating virtual environment.'
echo ''
python3 -m venv venv

source venv/bin/activate
pip install wheel

# Install requirements
echo 'Installing development requirements'
if [ -f 'requirements-dev.txt' ]
then
    pip install -r requirements-dev.txt
fi

echo 'Installing package requirements'
if [ -f 'requirements.txt' ]
then
    pip install -r requirements.txt
fi

echo 'Installing package and dependencies'
pip install -e .
