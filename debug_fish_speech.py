#!/usr/bin/env python3
"""
Debug script to test Fish Speech connectivity and configuration
"""

import asyncio
import yaml
import os
import logging
import base64
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def test_fish_speech_connection(config_path: str):
    """Test Fish Speech connection and configuration"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Get Fish Speech settings from config
    fish_url = config.get('fish_speech_url', 'http://localhost:8000/v1/tts')
    ref_audio_path = config.get('fish_speech_ref_audio')
    
    logging.info(f"Testing Fish Speech connection to: {fish_url}")
    logging.info(f"Reference audio path: {ref_audio_path}")
    
    # Check if reference audio exists
    if not ref_audio_path or not os.path.exists(ref_audio_path):
        logging.error(f"Reference audio file not found: {ref_audio_path}")
        return False
    
    # Read and encode reference audio
    with open(ref_audio_path, 'rb') as f:
        audio_bytes = f.read()
    ref_audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    # Create test payload
    test_payload = {
        'text': 'This is a test of the Fish Speech API.',
        'chunk_length': 200,
        'format': 'mp3',
        'references': [
            {
                'text': '',
                'audio': ref_audio_b64
            }
        ],
        'normalize': True,
        'top_p': 0.7,
        'repetition_penalty': 1.2,
        'temperature': 0.7,
        'streaming': False
    }
    
    # Test connection
    try:
        logging.info("Sending test request to Fish Speech API...")
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(fish_url, json=test_payload)
            
            logging.info(f"Response status code: {response.status_code}")
            
            if response.status_code == 200:
                logging.info("✓ Fish Speech API test successful!")
                logging.info(f"Response content length: {len(response.content)} bytes")
                return True
            else:
                logging.error(f"✗ Fish Speech API returned error: {response.status_code}")
                logging.error(f"Error response: {response.text}")
                return False
                
    except httpx.ConnectError:
        logging.error(f"✗ Cannot connect to Fish Speech API at {fish_url}")
        logging.error("Is the Fish Speech service running?")
        logging.error("To start: cd fish_speech && docker compose up -d")
        return False
    except httpx.TimeoutException:
        logging.error(f"✗ Timeout connecting to Fish Speech API at {fish_url}")
        return False
    except Exception as e:
        logging.error(f"✗ Unexpected error connecting to Fish Speech API: {e}")
        return False

async def main():
    # Test with the main config
    config_path = "config.yaml"
    
    if not os.path.exists(config_path):
        logging.error(f"Config file not found: {config_path}")
        return
    
    success = await test_fish_speech_connection(config_path)
    
    if success:
        logging.info("\n✓ Fish Speech appears to be configured correctly!")
        logging.info("You can now run the main script with tts_provider: 'fish_speech'")
    else:
        logging.info("\n✗ Fish Speech configuration needs to be fixed before running the main script.")
        logging.info("\nTroubleshooting steps:")
        logging.info("1. Make sure Docker is installed and running")
        logging.info("2. Start the Fish Speech service: cd fish_speech && docker compose up -d")
        logging.info("3. Wait ~30 seconds for the service to fully start")
        logging.info("4. Test the connection again")
        logging.info("5. Check that the reference audio file exists and is valid")

if __name__ == "__main__":
    asyncio.run(main())