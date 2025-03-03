from curses import raw
from email.mime import audio
import re
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Callable, Dict, NamedTuple, List
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
import tempfile
from madmom.audio.signal import FramedSignal
from madmom.audio.stft import STFT
from madmom.features.onsets import SpectralOnsetProcessor
from audiomentations import Compose, ApplyImpulseResponse, AddBackgroundNoise
from scipy.io import wavfile
import librosa
from TTS_api import get_tts_client, HUMAN_VOICE, ROBOT_VOICE, SAMPLE_RATE
import json
import os

@dataclass(frozen=True)
class AudioConfig:
    """Immutable audio processing configuration"""
    sample_rate: int = SAMPLE_RATE
    silence_duration: float = 0.2 # silence duration in seconds
    min_offset: float = 0.1 # minimum interrupting delay
    max_offset: float = 0.3 # maximum interrupting delay
    min_soundcard_offset: float = 0.05  # minimum soundcard delay
    max_soundcard_offset: float = 0.15  # maximum soundcard delay
    num_workers: int = 4 # number of workers for parallel processing
    irs_path: str = "../data/IRs/Audio/" # MIT IR dataset
    background_noises_path: str = "../data/bg_noises/ESC-50-master/audio/" # ESC-50 dataset
    target_length: float = 3.0 # target length of the merged audio, in seconds
    dataset_path: str = "../data/HRSP2mix/raw"
    clean_dataset_path: str = "../data/HRSP2mix/clean"  # 新增clean版本的保存路径
    max_snr = 20
    min_snr = 10

class AudioMetadata(NamedTuple):
    """Named tuple for audio processing metadata"""
    human_audio_text: str
    robot_audio_text: str
    human_audio_path: str
    robot_audio_path: str
    human_audio_agent: str
    robot_audio_agent: str
    merged_audio_path: str
    background_noise_name: str
    ir_name: str
    robot_start_time: float
    robot_duration: float
    offset: float
    soundcard_offset: float

class AudioProcessor:
    """Memory-efficient and thread-safe audio processor"""
    
    def __init__(self, config: AudioConfig = None):
        self.config = config or AudioConfig()
        self._setup_processors()
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_lock = threading.Lock()
        self.current_ir_name = None  # record the current IR used
        self.current_noise_name = None    # record the current noise used
        self._ir_files = None
        self._noise_files = None
        self._setup_file_lists()
        
    def _setup_file_lists(self) -> None:
        """Setup lists of available IR and noise files"""
        ir_path = Path(self.config.irs_path)
        noise_path = Path(self.config.background_noises_path)
        
        if not ir_path.exists() or not noise_path.exists():
            print(f"IR path: {ir_path}")
            print(f"Noise path: {noise_path}")
            raise RuntimeError("IR or noise directories not found")
            
        self._ir_files = list(ir_path.glob('*.wav'))
        self._noise_files = list(noise_path.glob('*.wav'))
        
        if not self._ir_files or not self._noise_files:
            raise RuntimeError("No WAV files found in IR or noise directories")

    def _get_random_file(self, file_list: List[Path]) -> Path:
        """Randomly select a file from the list"""
        return np.random.choice(file_list)

    def _setup_processors(self) -> None:
        """Initialize basic processors without specific files"""
        self.onset_processor = SpectralOnsetProcessor()
        self.tts_client = get_tts_client()
        # Note: augmenter will be set up per-processing

    def _load_audio_bytes(self, audio_bytes: bytes) -> np.ndarray:
        """Convert audio bytes to numpy array safely"""
        with tempfile.NamedTemporaryFile(suffix='.wav') as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            return librosa.load(tmp.name, sr=self.config.sample_rate)[0]
    
    def _process_audio(self, audio: np.ndarray, noise_file: Path, ir_file: Path) -> np.ndarray:
        """Process audio with specified IR and noise files"""
        # create augmenter with selected files
        augmenter = Compose([
            AddBackgroundNoise(
                sounds_path=str(noise_file),
                p=1.0,
                max_snr_db=self.config.max_snr,
                min_snr_db=self.config.min_snr,
            ),
            ApplyImpulseResponse(
                ir_path=str(ir_file),
                p=1.0,
                leave_length_unchanged=True
            )
        ])
        
        # apply augmentation
        return augmenter(
            samples=audio,
            sample_rate=self.config.sample_rate
        )

    def detect_onset(self, audio: np.ndarray) -> int:
        """Optimized onset detection using FFTW"""
        # Create framed signal first with optimized parameters
        framed_signal = FramedSignal(
            audio, 
            frame_size=2048, 
            hop_size=441,
            sample_rate=self.config.sample_rate,
            fft_parameters={
                'window': 'hann',
                'fftw': True,  # use FFTW
                'num_threads': self.config.num_workers  # parallel processing
            }
        )
        # Create STFT with FFTW optimization
        stft = STFT(
            framed_signal,
            fft_parameters={
                'fftw': True,
                'num_threads': self.config.num_workers,
                'wisdom_file': 'fftw_wisdom.txt'  # save/load FFTW wisdom
            }
        )
        onset_curve = self.onset_processor(stft)
        onset_times = np.where(onset_curve > np.mean(onset_curve))[0]
        return onset_times[0] if len(onset_times) > 0 else 0

    def align_signals(self, audio1: np.ndarray, audio2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Align signals by extending to target length and adding silence padding"""
        # adjust length to be the same
        max_length = max(len(audio1), len(audio2))
        audio1_padded = librosa.util.fix_length(audio1, size=max_length)
        audio2_padded = librosa.util.fix_length(audio2, size=max_length)
        
        # detect onset
        onset1, onset2 = self.detect_onset(audio1_padded), self.detect_onset(audio2_padded)
        
        # calculate target length in samples
        target_samples = int(self.config.target_length * self.config.sample_rate)
        silence_samples = int(self.config.silence_duration * self.config.sample_rate)
        
        # calculate pre-silence duration
        pre_silence1 = max(0, onset2 - onset1)
        pre_silence2 = max(0, onset1 - onset2)
        
        # pad with silence
        audio1_aligned = np.pad(audio1_padded, (pre_silence1 + silence_samples, silence_samples), mode='constant')
        audio2_aligned = np.pad(audio2_padded, (pre_silence2 + silence_samples, silence_samples), mode='constant')
        
        # adjust to target length
        final_length = max(len(audio1_aligned), len(audio2_aligned), target_samples)
        audio1_final = librosa.util.fix_length(audio1_aligned, size=final_length)
        audio2_final = librosa.util.fix_length(audio2_aligned, size=final_length)
        
        return audio1_final, audio2_final

    def _detect_overlap(self, human_audio: np.ndarray, robot_audio: np.ndarray, 
                       total_offset: int) -> bool:
        """detect overlap between human and robot audio"""
        human_active = librosa.effects.split(human_audio, top_db=20)
        robot_active = librosa.effects.split(robot_audio, top_db=20)
        
        if len(human_active) == 0 or len(robot_active) == 0:
            return False
            
        # calculate start and end times for human and robot audio
        human_start = human_active[0][0] + total_offset
        human_end = human_active[-1][1] + total_offset
        robot_start, robot_end = robot_active[0][0], robot_active[-1][1]
        
        # check for overlap
        return not (human_end < robot_start or human_start > robot_end)
    
    def _calculate_robot_info(self, robot_audio: np.ndarray) -> Tuple[float, float]:
        """Calculate robot audio start time and duration"""
        robot_active = librosa.effects.split(robot_audio, top_db=20)
        if len(robot_active) == 0:
            return 0.0, 0.0
            
        start_sample = robot_active[0][0]
        duration_samples = robot_active[-1][1] - robot_active[0][0]
        
        start_time = start_sample / self.config.sample_rate
        duration = duration_samples / self.config.sample_rate
        
        return start_time, duration

    def process_dialogue(self, 
                        human_text: str, 
                        robot_text: str, 
                        output_dir: str,
                        sample_id: str,
                        progress_callback: Optional[Callable[[float], None]] = None) -> AudioMetadata:
        """Enhanced dialogue processing with metadata recording"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # 创建clean版本的目录
            clean_output_dir = output_dir.replace(self.config.dataset_path, self.config.clean_dataset_path)
            os.makedirs(clean_output_dir, exist_ok=True)
            
            # Select IR and noise files at the beginning
            noise_file = self._get_random_file(self._noise_files)
            ir_file = self._get_random_file(self._ir_files)
            
            # record the current IR and noise used
            self.current_ir_name = ir_file.name
            self.current_noise_name = noise_file.name
            
            # generate TTS audio
            with ThreadPoolExecutor(max_workers=2) as executor:
                speech_futures = {
                    'human': executor.submit(self.tts_client.generate_human_speech, human_text),
                    'robot': executor.submit(self.tts_client.generate_robot_speech, robot_text)
                }
                
                audio_data = {k: f.result() for k, f in speech_futures.items()}
                if progress_callback:
                    progress_callback(0.2)

            # load audio bytes
            human_audio = self._load_audio_bytes(audio_data['human'])
            robot_audio = self._load_audio_bytes(audio_data['robot'])
            
            if progress_callback:
                progress_callback(0.4)

            # align signals
            human_audio, robot_audio = self.align_signals(human_audio, robot_audio)
            
            # detect overlap and find valid offset
            max_attempts = 50
            for _ in range(max_attempts):
                offset = int(np.random.uniform(
                    self.config.min_offset,
                    self.config.max_offset
                ) * self.config.sample_rate)
                
                soundcard_offset = int(np.random.uniform(
                    self.config.min_soundcard_offset,
                    self.config.max_soundcard_offset
                ) * self.config.sample_rate)
                
                total_offset = offset + soundcard_offset
                
                if self._detect_overlap(human_audio, robot_audio, total_offset):
                    break
            else:
                raise RuntimeError("Failed to find valid offset with overlap")

            if progress_callback:
                progress_callback(0.6)

            # apply offset to human audio
            human_audio_shifted = np.roll(human_audio, total_offset)

            # 保存clean版本（偏移但没有添加环境因素）
            clean_human_path = os.path.join(clean_output_dir, f"{sample_id}_human.wav")
            clean_robot_path = os.path.join(clean_output_dir, f"{sample_id}_robot.wav")
            clean_merged_path = os.path.join(clean_output_dir, f"{sample_id}_merged.wav")
            
            # 保存clean版本的音频
            clean_merged = human_audio_shifted + robot_audio
            wavfile.write(clean_human_path, self.config.sample_rate, human_audio_shifted.astype(np.float32))
            wavfile.write(clean_robot_path, self.config.sample_rate, robot_audio.astype(np.float32))
            wavfile.write(clean_merged_path, self.config.sample_rate, clean_merged.astype(np.float32))

            # 处理带环境因素的版本
            human_audio_shifted = self._process_audio(human_audio_shifted, noise_file, ir_file)
            robot_audio = self._process_audio(robot_audio, noise_file, ir_file)

            merged = human_audio_shifted + robot_audio

            # calculate robot audio start time and duration
            robot_start_time, robot_duration = self._calculate_robot_info(robot_audio)

            # 保存带环境因素的版本
            human_original_path = os.path.join(output_dir, f"{sample_id}_human.wav")
            robot_original_path = os.path.join(output_dir, f"{sample_id}_robot.wav")
            merged_path = os.path.join(output_dir, f"{sample_id}_merged.wav")
            
            wavfile.write(human_original_path, self.config.sample_rate, human_audio_shifted.astype(np.float32))
            wavfile.write(robot_original_path, self.config.sample_rate, robot_audio.astype(np.float32))
            wavfile.write(merged_path, self.config.sample_rate, merged.astype(np.float32))
            
            if progress_callback:
                progress_callback(1.0)

            # 保存clean版本的metadata
            # clean_metadata_path = os.path.join(clean_output_dir, "metadata.json")
            clean_metadata = AudioMetadata(
                human_audio_text=human_text,
                robot_audio_text=robot_text,
                human_audio_path=clean_human_path,
                robot_audio_path=clean_robot_path,
                human_audio_agent=HUMAN_VOICE,
                robot_audio_agent=ROBOT_VOICE,
                merged_audio_path=clean_merged_path,
                background_noise_name="none",  # 无背景噪音
                ir_name="none",  # 无IR
                robot_start_time=robot_start_time,
                robot_duration=robot_duration,
                offset=offset / self.config.sample_rate,
                soundcard_offset=soundcard_offset / self.config.sample_rate
            )
            
            # with open(clean_metadata_path, 'w') as f:
            #     json.dump(clean_metadata._asdict(), f, indent=2)

            if progress_callback:
                progress_callback(1.0)

            raw_metadata = AudioMetadata(
                human_audio_text=human_text,
                robot_audio_text=robot_text,
                human_audio_path=human_original_path,
                robot_audio_path=robot_original_path,
                human_audio_agent=HUMAN_VOICE,
                robot_audio_agent=ROBOT_VOICE,
                merged_audio_path=merged_path,
                background_noise_name=self.current_noise_name,
                ir_name=self.current_ir_name,
                robot_start_time=robot_start_time,
                robot_duration=robot_duration,
                offset=offset / self.config.sample_rate,
                soundcard_offset=soundcard_offset / self.config.sample_rate
            )
        
            return clean_metadata, raw_metadata

        except Exception as e:
            raise RuntimeError(f"Error processing dialogue: {str(e)}") from e

    def generate_dataset(self, dialogue_pairs: List[Tuple[str, str]], 
                        output_dir: str = None) -> None:
        """Generate dataset from dialogue pairs, first element is human text, second is robot text"""
        output_dir = output_dir or self.config.dataset_path
        os.makedirs(output_dir, exist_ok=True)
        
        for idx, (robot_text, human_text) in enumerate(dialogue_pairs):
            print(f"Processing dialogue pair {idx+1}/{len(dialogue_pairs)}")
            
            # Create subfolder for each dialogue pair
            sample_dir = os.path.join(output_dir, f"sample_{idx:04d}")
            os.makedirs(sample_dir, exist_ok=True)
            
            try:
                clean_metadata, raw_metadata = self.process_dialogue(
                    human_text,
                    robot_text,
                    sample_dir,
                    f"sample_{idx:04d}",
                    progress_callback=lambda p: print(f"Progress: {p*100:.1f}%")
                )
                
                # save metadata
                clean_metadata_path = os.path.join(sample_dir.replace(self.config.dataset_path, self.config.clean_dataset_path), "metadata.json")
                with open(clean_metadata_path, 'w') as f:
                    json.dump(clean_metadata._asdict(), f, indent=2)
                    
                raw_metadata_path = os.path.join(sample_dir, "metadata.json")
                with open(raw_metadata_path, 'w') as f:
                    json.dump(raw_metadata._asdict(), f, indent=2)
                    
            except Exception as e:
                print(f"Error processing dialogue pair {idx}: {str(e)}")
                continue

def main():
    def progress(p: float):
        print(f"Processing: {p*100:.1f}%")

    processor = AudioProcessor()

    # generate dataset test
    test_dialogues = []

    with open("100_mixed_dialogue_pairs.txt") as f:
        pattern = r'"([^"]*?)"\s*,\s*"([^"]*?)"'
        for line in f:
            match = re.match(pattern, line)
            if match:
                test_dialogues.append((match.group(1), match.group(2)))
            else:
                print(f"Invalid line: {line}")
    

    # Shuffle dialogues for randomization
    np.random.shuffle(test_dialogues)

    # print(len(test_dialogues))
    processor.generate_dataset(test_dialogues, processor.config.dataset_path)

if __name__ == "__main__":
    main()

