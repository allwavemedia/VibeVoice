#!/usr/bin/env python3
"""
Example script showing how to add and manage voices in VibeVoice
"""

import os
import shutil
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path

def add_voice_to_vibevoice(
    source_audio_path: str,
    speaker_name: str,
    language_code: str = "en",
    gender: str = "neutral",
    voices_dir: str = "demo/voices"
):
    """
    Add a new voice to VibeVoice system
    
    Args:
        source_audio_path: Path to the source audio file
        speaker_name: Name for the speaker (will be used in scripts)
        language_code: Language code (en, zh, fr, etc.)
        gender: Gender identifier (male, female, neutral)
        voices_dir: Directory where voices are stored
    """
    
    # Create voices directory if it doesn't exist
    os.makedirs(voices_dir, exist_ok=True)
    
    # Generate filename following VibeVoice convention
    filename = f"{language_code}-{speaker_name}_{gender}.wav"
    target_path = os.path.join(voices_dir, filename)
    
    # Load and process audio
    print(f"Processing audio: {source_audio_path}")
    audio, sr = librosa.load(source_audio_path, sr=24000, mono=True)
    
    # Optional: Apply basic preprocessing
    # Remove silence from beginning and end
    audio, _ = librosa.effects.trim(audio, top_db=20)
    
    # Normalize audio
    audio = audio / np.max(np.abs(audio)) * 0.95
    
    # Save processed audio
    sf.write(target_path, audio, 24000)
    print(f"Voice added: {target_path}")
    print(f"Speaker name for scripts: '{speaker_name}'")
    
    return target_path

def create_voice_with_background_music(
    voice_path: str,
    music_path: str,
    speaker_name: str,
    language_code: str = "en",
    voices_dir: str = "demo/voices",
    music_volume: float = 0.1
):
    """
    Create a voice sample with background music
    
    Args:
        voice_path: Path to clean voice audio
        music_path: Path to background music
        speaker_name: Speaker name
        language_code: Language code
        voices_dir: Voices directory
        music_volume: Volume level for background music (0.0-1.0)
    """
    
    # Load voice and music
    voice, sr = librosa.load(voice_path, sr=24000, mono=True)
    music, _ = librosa.load(music_path, sr=24000, mono=True)
    
    # Match lengths
    if len(music) > len(voice):
        music = music[:len(voice)]
    else:
        # Loop music if shorter than voice
        repeats = int(np.ceil(len(voice) / len(music)))
        music = np.tile(music, repeats)[:len(voice)]
    
    # Mix voice and music
    mixed = voice + (music * music_volume)
    
    # Normalize to prevent clipping
    mixed = mixed / np.max(np.abs(mixed)) * 0.95
    
    # Save with BGM suffix
    filename = f"{language_code}-{speaker_name}_bgm.wav"
    target_path = os.path.join(voices_dir, filename)
    sf.write(target_path, mixed, 24000)
    
    print(f"Voice with BGM created: {target_path}")
    return target_path

def list_available_voices(voices_dir: str = "demo/voices"):
    """List all available voices in the system"""
    
    if not os.path.exists(voices_dir):
        print(f"Voices directory not found: {voices_dir}")
        return []
    
    voice_files = [f for f in os.listdir(voices_dir) 
                   if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'))]
    
    print(f"\nAvailable voices ({len(voice_files)} total):")
    print("-" * 50)
    
    voices_info = []
    for voice_file in sorted(voice_files):
        # Parse filename to extract info
        name = os.path.splitext(voice_file)[0]
        
        # Extract speaker name using VibeVoice logic
        speaker_name = name
        if '_' in name:
            speaker_name = name.split('_')[0]
        if '-' in speaker_name:
            speaker_name = speaker_name.split('-')[-1]
        
        # Check for BGM
        has_bgm = 'bgm' in name.lower()
        
        # Get file size
        file_path = os.path.join(voices_dir, voice_file)
        file_size = os.path.getsize(file_path)
        
        voices_info.append({
            'file': voice_file,
            'speaker_name': speaker_name,
            'has_bgm': has_bgm,
            'size_kb': file_size // 1024
        })
        
        bgm_indicator = " [BGM]" if has_bgm else ""
        print(f"  {speaker_name:<15} | {voice_file:<25} | {file_size//1024:>6} KB{bgm_indicator}")
    
    return voices_info

def validate_voice_file(audio_path: str):
    """Validate if an audio file is suitable for VibeVoice"""
    
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        duration = len(audio) / sr
        
        print(f"\nVoice file validation: {audio_path}")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Sample rate: {sr} Hz")
        print(f"  Channels: {'Mono' if len(audio.shape) == 1 else 'Stereo'}")
        
        # Recommendations
        if duration < 2:
            print("  ⚠️  Warning: Very short audio (< 2s). Consider longer sample.")
        elif duration > 30:
            print("  ⚠️  Warning: Very long audio (> 30s). Consider shorter sample.")
        else:
            print("  ✅ Duration looks good")
            
        if sr != 24000:
            print(f"  ℹ️  Will be resampled from {sr}Hz to 24000Hz")
        else:
            print("  ✅ Sample rate is optimal")
            
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

# Example usage
if __name__ == "__main__":
    # Example 1: Add a simple voice
    # add_voice_to_vibevoice(
    #     source_audio_path="path/to/your/voice.wav",
    #     speaker_name="Emma",
    #     language_code="en",
    #     gender="female"
    # )
    
    # Example 2: Create voice with background music
    # create_voice_with_background_music(
    #     voice_path="demo/voices/en-Emma_female.wav",
    #     music_path="path/to/background_music.wav",
    #     speaker_name="Emma",
    #     language_code="en"
    # )
    
    # Example 3: List all available voices
    list_available_voices()
    
    # Example 4: Validate a voice file
    # validate_voice_file("path/to/your/voice.wav")