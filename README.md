# Human-Robot-Speaker-Separation
Human-Robot-Speaker-Separation(HRSP) is a speech dataset aimed at simulating human-robot interaction. It is available at 8kHz and 16kHz sample rates. For each sample rate and voice type (TTS-based synthetic human voice or real human voice recording), two versions are provided: the `clean` version, which simulates only dialogue overlap, and the `raw` version, which further simulates realistic microphone capture by incorporating reverberation IRs and background noises.

## INSTALL
### Use shell file to install
You can run the `download_resources.sh` with following args:
* sample rate: `8k`, `16k`, or `all` (default: `all`)
* version: `clean`, `raw`, or `all` (default: `all`)
* type: `tts`, `real`, or `all` (default: `all`)

eg:

```
./download_resources.sh 8k clean tts
./download_resources.sh 16k raw real
./download_resources.sh all all all
```

> if permission denied, use command `chmod +x ./download_resources.sh `

### Manually install 
Or just directly install from the storage platform:
* HRSP2mix TTS (Text-to-Speech based human voice) Datasets:
  * [8k TTS clean](https://archive.org/download/hrsp-2mix-8000-human-clean-0512/HRSP2mix_8000_human_clean.zip)
  * [8k TTS raw](https://archive.org/download/hrsp-2mix-8000-human-raw/HRSP2mix_8000_human_raw.zip)
  * [16k TTS clean](https://archive.org/download/hrsp-2mix-16000-human-clean-0512/HRSP2mix_16000_human_clean.zip)
  * [16k TTS raw](https://archive.org/download/hrsp-2mix-16000-human-raw-0512/HRSP2mix_16000_human_raw.zip)
* HRSP2mix REAL (Real human voice recording) Datasets:
  * [8k REAL clean](https://archive.org/download/hrsp-2mix-8000-real-human-clean/HRSP2mix_8000_real_human_clean.zip)
  * [8k REAL raw](https://archive.org/download/hrsp-2mix-8000-real-human-raw/HRSP2mix_8000_real_human_raw.zip)
  * [16k REAL clean](https://archive.org/download/hrsp-2mix-16000-real-human-clean/HRSP2mix_16000_real_human_clean.zip)
  * [16k REAL raw](https://archive.org/download/hrsp-2mix-16000-real-human-raw/HRSP2mix_16000_real_human_raw.zip)
* Related IRs and noises:
  * [MIT Acoustical Reverb](https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip)
  * [ESC-50](https://github.com/karoldvl/ESC-50/archive/master.zip)
  
If you manually install the file, please put them to fit the dir rules as following:

> TIPS: For manual installation, create the directory structure as shown. Each link provides a specific version (e.g. 8k TTS raw). Extract the contents of the downloaded zip into the respective directory (e.g. HRSP2mix_8000_human/). The download script handles placing files correctly based on arguments.
```
 data
 ├── HRSP2mix_8000_human         // Contains 8k TTS (human) datasets (clean/raw versions downloaded here)
 ├── HRSP2mix_8000_real_human    // Contains 8k REAL (real_human) datasets (clean/raw versions downloaded here)
 ├── HRSP2mix_16000_human        // Contains 16k TTS (human) datasets (clean/raw versions downloaded here)
 ├── HRSP2mix_16000_real_human   // Contains 16k REAL (real_human) datasets (clean/raw versions downloaded here)
 ├── IRs
 │   ├── Audio
 │   └── __MACOSX
 └── bg_noises
     └── ESC-50-master
```
  
## DATA PREPARE
`src/prepare_data.py` will split train/test/valid csv file for each valid version of HRSP2mix dataset with `processed` prefix, you can use arg `--path-type` to decide use the (`rel`)relative or (`abs`)absolute path for audio files, default use `rel`.

