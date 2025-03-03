import base64
import datetime
import json
import requests
from typing import Tuple, Dict
from functools import lru_cache
from pathlib import Path
import threading

HUMAN_VOICE = "Adam"  # human voice agent
ROBOT_VOICE = "Heather"  # robot voice agent
SAMPLE_RATE = 8000  # sample rate for generated audio

class TTSConfig:
    API_URL = "https://api.cerevoice.com/v2"
    TOKEN_PATH = Path("../config/TTS_TOKEN/tokens.json")
    
    def __init__(self):
        self._load_credentials()
        self._load_tokens()
        
    def _load_credentials(self):
        # In production, these should be loaded from environment variables
        self.username = "kw2021@hw.ac.uk"
        self.password = "Wangkangdi21"
        
    def _load_tokens(self):
        try:
            with self.TOKEN_PATH.open() as f:
                tokens = json.load(f)
                self.access_token = tokens.get("access_token")
                self.refresh_token = tokens.get("refresh_token")
        except (FileNotFoundError, json.JSONDecodeError):
            self.access_token, self.refresh_token = self._get_new_tokens()
            self._save_tokens()
            
    def _save_tokens(self):
        self.TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        with self.TOKEN_PATH.open('w') as f:
            json.dump({
                "access_token": self.access_token,
                "refresh_token": self.refresh_token
            }, f)

    def _get_new_tokens(self) -> Tuple[str, str]:
        response = requests.get(
            f"{self.API_URL}/auth",
            auth=(self.username, self.password)
        )
        response.raise_for_status()
        data = response.json()
        return data["access_token"], data["refresh_token"]

class TTSClient:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.config = TTSConfig()
            self.session = requests.Session()
            self._refresh_lock = threading.Lock()
            self.initialized = True
            self._check_and_refresh_tokens()
    
    def _check_and_refresh_tokens(self):
        """Check token status and refresh if needed"""
        with self._refresh_lock:
            if self._is_token_expired():
                self._refresh_tokens()
    
    @property
    def headers(self):
        self._check_and_refresh_tokens()
        return {
            "Authorization": f"Bearer {self.config.access_token}",
            "Content-Type": "application/json"
        }
        
    def _check_token(self):
        if self._is_token_expired():
            self._refresh_tokens()
            
    def _is_token_expired(self) -> bool:
        try:
            payload = self._decode_token(self.config.access_token)
            expiry = datetime.datetime.fromtimestamp(payload['exp'])
            return expiry < datetime.datetime.utcnow()
        except:
            return True
            
    @staticmethod
    def _decode_token(token: str) -> Dict:
        padding = "=" * ((4 - len(token) % 4) % 4)
        payload = token.split(".")[1] + padding
        return json.loads(base64.b64decode(payload))
        
    def _refresh_tokens(self):
        self.config.access_token, self.config.refresh_token = self.config._get_new_tokens()
        self.config._save_tokens()

    @lru_cache(maxsize=128)
    def generate_speech(self, text: str, voice: str) -> bytes:
        """Generate speech with automatic token refresh"""
        try:
            response = self.session.post(
                f"{self.config.API_URL}/speak",
                params={
                    "voice": voice, 
                    "streaming": "true",
                    "sample_rate": SAMPLE_RATE
                },
                headers={
                    "Authorization": f"Bearer {self.config.access_token}",
                    "Content-Type": "text/plain",
                    "accept": "audio/wav"
                },
                data=text
            )
            response.raise_for_status()
            return response.content
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:  # Unauthorized
                self._check_and_refresh_tokens()
                # Retry once after token refresh
                response = self.session.post(
                    f"{self.config.API_URL}/speak",
                    params={
                        "voice": voice, 
                        "streaming": "true",
                        "sample_rate": SAMPLE_RATE
                    },
                    headers={
                        "Authorization": f"Bearer {self.config.access_token}",
                        "Content-Type": "text/plain",
                        "accept": "audio/wav"
                    },
                    data=text
                )
                response.raise_for_status()
                return response.content
            raise

    def generate_human_speech(self, text: str) -> bytes:
        return self.generate_speech(text, HUMAN_VOICE)
        
    def generate_robot_speech(self, text: str) -> bytes:
        return self.generate_speech(text, ROBOT_VOICE)

# Global instance
_tts_client = None

def get_tts_client() -> TTSClient:
    """Get TTSClient instance with fresh tokens"""
    client = TTSClient()
    client._check_and_refresh_tokens()
    return client

# Convenience functions
def generate_human_agent_speech(text: str) -> bytes:
    return get_tts_client().generate_human_speech(text)

def generate_robot_agent_speech(text: str) -> bytes:
    return get_tts_client().generate_robot_speech(text)

if __name__ == "__main__":
    # Test the TTS client
    client = get_tts_client()
    # print(client._is_token_expired())
    print(client.generate_human_speech("Hello, this is Heather"))
    # print(client.generate_robot_speech("Hello, this is Heather"))
