#!/bin/bash

# Create necessary directories
mkdir -p data/IRs/Audio
mkdir -p data/bg_noises

# Change to data directory
cd data

# Download and extract IRs
if [ ! -d "IRs/Audio" ]; then
    echo "Downloading and extracting IRs..."
    wget -O Audio.zip https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip
    unzip Audio.zip -d IRs/
    rm Audio.zip
else
    echo "IRs already exist, skipping download..."
fi

# Download and extract background noises
if [ ! -d "bg_noises/ESC-50-master" ]; then
    echo "Downloading and extracting background noises..."
    wget -O bg_noises.zip https://github.com/karoldvl/ESC-50/archive/master.zip
    unzip bg_noises.zip -d bg_noises/
    rm bg_noises.zip
else
    echo "Background noises already exist, skipping download..."
fi

# Download and extract HRSP2mix
# Handle HRSP2mix download based on arguments
echo "Downloading and extracting HRSP2mix..."

# Default values
sample_rate="all"  # First arg: 8k, 16k, or all
version="all"      # Second arg: raw, clean, or all

# Check command line arguments
if [ $# -ge 1 ]; then
    sample_rate="$1"
fi
if [ $# -ge 2 ]; then
    version="$2"
fi

# Download based on sample rate and version
if [[ "$sample_rate" == "8k" || "$sample_rate" == "all" ]]; then
    mkdir -p HRSP2mix_8k/
    if [[ "$version" == "raw" || "$version" == "all" ]]; then
        # wget -O temp_8k_raw.tar.gz https://archive.org/download/hrsp-2mix-8000-raw/HRSP2mix_8000_raw.zip
        wget -O https://archive.org/download/hrsp-2mix-8000-human-raw/HRSP2mix_8000_human_raw.zip
        tar -xzf temp_8k_raw.tar.gz -C /HRSP2mix_8k/
        rm temp_8k_raw.tar.gz
    fi
    if [[ "$version" == "clean" || "$version" == "all" ]]; then
        wget -O temp_8k_clean.tar.gz https://archive.org/download/hrsp-2mix-8000-human-clean-0512/HRSP2mix_8000_human_clean.zip
        tar -xzf temp_8k_clean.tar.gz -C HRSP2mix_8k/
        rm temp_8k_clean.tar.gz
    fi
fi

if [[ "$sample_rate" == "16k" || "$sample_rate" == "all" ]]; then
    mkdir -p HRSP2mix_16k/
    if [[ "$version" == "raw" || "$version" == "all" ]]; then
        wget -O temp_16k_raw.tar.gz https://archive.org/download/hrsp-2mix-16000-human-raw-0512/HRSP2mix_16000_human_raw.zip
        tar -xzf temp_16k_raw.tar.gz -C HRSP2mix_16k/
        rm temp_16k_raw.tar.gz
    fi
    if [[ "$version" == "clean" || "$version" == "all" ]]; then
        wget -O temp_16k_clean.tar.gz https://archive.org/download/hrsp-2mix-16000-human-clean-0512/HRSP2mix_16000_human_clean.zip
        tar -xzf temp_16k_clean.tar.gz -C HRSP2mix_16k/
        rm temp_16k_clean.tar.gz
    fi
fi

echo "Download and extraction completed!"