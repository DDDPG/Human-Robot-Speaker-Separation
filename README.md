# Human-Robot-Speaker-Separation
Human-Robot-Speaker-Separation(HRSP) is a speech dataset aims to simulate interaction between human and robot, with 8k and 16k sample rate, and each sample rate has two versions: the `clean` version only simulate the overlap during dialogues, the `raw` version use reverb IRs and noises to further simulate daily situations' mic capture from robot.

## INSTALL
### Use shell file to install
You can run the `download_resources.sh` with following args:
* sample rate: `8k`, `16k` or `all`
* version: `clean`, `raw` or `all`

eg:

```
./download_resources.sh 8k clean
./download_resources.sh all all
```

> if permission denied, use command `chmod +x ./download_resources.sh `

### Manually install 
Or just directly install from the storage platform:
* HRSP2mix dataset:
  * [8k clean](https://archive.org/download/hrsp-2mix-8k-clean/HRSP2mix_8k_clean.zip)
  * [8k raw](https://archive.org/download/hrsp-2mix-8k-raw/HRSP2mix_8k_raw.zip)
  * [16k clean](https://archive.org/download/hrsp-2mix-16k-clean/HRSP2mix_16k_clean.zip)
  * [16k raw](https://archive.org/download/hrsp-2mix-16k-raw/HRSP2mix_16k_raw.zip)
* Related IRs and noises:
  * [MIT Acoustical Reverb](https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip)
  * [ESC-50](https://github.com/karoldvl/ESC-50/archive/master.zip)
  
If you manually install the file, please put them to fit the dir rules as following:

> TIPS: `clean` and `raw` is the root folder for the unzip result
```
 data
 ├── HRSP2mix_16k
 │   ├── clean
 │   └── raw
 ├── HRSP2mix_8k
 │   ├── clean
 │   └── raw
 ├── IRs
 │   ├── Audio
 │   └── __MACOSX
 └── bg_noises
     └── ESC-50-master
```
  
## DATA PREPARE
`src/prepare_data.py` will split train/test/valid csv file for each valid version of HRSP2mix dataset with `processed` prefix, you can use arg `--path-type` to decide use the (`rel`)relative or (`abs`)absolute path for audio files, default use `rel`.

