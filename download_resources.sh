#!/bin/bash

# Create necessary directories
mkdir -p data/IRs/Audio
mkdir -p data/bg_noises
mkdir -p data/HRSP2mix

# Change to data directory
cd data

# Download and extract IRs
echo "Downloading and extracting IRs..."
wget -O Audio.zip https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip
unzip Audio.zip -d IRs/Audio/
rm Audio.zip

# Download and extract background noises
echo "Downloading and extracting background noises..."
wget -O bg_noises.zip https://github.com/karoldvl/ESC-50/archive/master.zip
unzip bg_noises.zip -d bg_noises/
rm bg_noises.zip

# # Download and extract HRSP2mix
# echo "Downloading and extracting HRSP2mix..."
# wget -O HRSP2mix.tar.gz https://path/to/your/HRSP2mix.tar.gz
# tar -xzf HRSP2mix.tar.gz -C HRSP2mix/
# rm HRSP2mix.tar.gz

echo "Download and extraction completed!"