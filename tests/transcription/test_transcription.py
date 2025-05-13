import argparse
import os
import sys
import wave
import logging
import pandas as pd
import numpy as np
import json
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Any
import requests
from jiwer import wer, cer
import Levenshtein

# Add parent directory to path to import app modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from app.services.transcription_service import TranscriptionService
from app.services.audio_service import AudioService
from app.utils.speech_processor import SpeechProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("wer_calculator")

class WERCalculator:
    """
    Calculator for Word Error Rate between transcriptions
    """
    
    def __init__(self, api_url: Optional[str] = None):
        """
        Initialize the WER calculator
        
        Args:
            api_url: Optional URL to the transcription API
        """
        self.api_url = api_url
        
        # Initialize services directly if no API URL provided
        if not api_url:
            self.transcription_service = TranscriptionService()
            self.audio_service = AudioService()
            self.speech_processor = SpeechProcessor()
        
    def load_audio(self, audio_path: str) -> np.ndarray:
        """
        Load audio data from a file
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Numpy array of audio samples
        """
        try:
            with wave.open(audio_path, 'rb') as wav_file:
                # Get audio parameters
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                frame_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                
                # Read all frames at once
                frames = wav_file.readframes(n_frames)
            
            # Convert to numpy array for processing
            audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
            audio_data = audio_data / 32768.0  # Normalize to [-1.0, 1.0]
            
            logger.info(f"Loaded audio file: {audio_path}")
            logger.info(f"  Channels: {channels}, Sample width: {sample_width}, Frame rate: {frame_rate}")
            logger.info(f"  Total frames: {n_frames}, Duration: {n_frames/frame_rate:.2f} seconds")
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Error loading audio file: {str(e)}")
            raise
    
    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int = 16000, 
                        use_adaptation: bool = False, user_id: Optional[int] = None) -> str:
        """
        Transcribe audio data to text
        
        Args:
            audio_data: Audio samples
            sample_rate: Sample rate in Hz
            use_adaptation: Whether to use speaker adaptation
            user_id: User ID for adaptation
            
        Returns:
            Transcribed text
        """
        try:
            if self.api_url:
                # Use the API for transcription
                # This is a placeholder - you'd need to implement the API call
                logger.error("API transcription not implemented yet")
                return ""
            else:
                # Use direct transcription
                if use_adaptation and user_id is not None:
                    transcription = self.speech_processor.transcribe_with_adaptation(
                        audio_data, user_id, None, sample_rate=sample_rate
                    )
                else:
                    transcription = self.speech_processor.transcribe(
                        audio_data, sample_rate=sample_rate
                    )
                
                logger.info(f"Transcription completed: {len(transcription)} characters")
                return transcription
                
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            return ""
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for fair comparison
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation (keep apostrophes and hyphens for compound words)
        import re
        text = re.sub(r'[^\w\s\'-]', '', text)
        
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def calculate_wer(self, reference: str, hypothesis: str) -> Dict[str, Any]:
        """
        Calculate Word Error Rate between reference and hypothesis
        
        Args:
            reference: Reference text
            hypothesis: Hypothesized text (transcription)
            
        Returns:
            Dictionary with WER metrics
        """
        # Normalize both texts
        reference_norm = self.normalize_text(reference)
        hypothesis_norm = self.normalize_text(hypothesis)
        
        # Split into words
        reference_words = reference_norm.split()
        hypothesis_words = hypothesis_norm.split()
        
        # Calculate WER using jiwer
        wer_score = wer(reference_norm, hypothesis_norm)
        
        # Calculate CER (Character Error Rate)
        cer_score = cer(reference_norm, hypothesis_norm)
        
        # Calculate Levenshtein distance for more detailed metrics
        from jiwer import compute_measures
        measures = compute_measures(reference_norm, hypothesis_norm)
        
        # Create detailed results
        results = {
            "wer": wer_score,
            "cer": cer_score,
            "substitutions": measures["substitutions"],
            "deletions": measures["deletions"],
            "insertions": measures["insertions"],
            "hits": measures["hits"],
            "reference_length": len(reference_words),
            "hypothesis_length": len(hypothesis_words),
            "reference": reference,
            "hypothesis": hypothesis,
            "reference_normalized": reference_norm,
            "hypothesis_normalized": hypothesis_norm
        }
        
        return results
    
    def compare_transcriptions(self, audio_path: str, reference: str, 
                              use_adaptation: bool = False, user_id: Optional[int] = None) -> Dict[str, Any]:
        """
        End-to-end comparison of audio transcription against reference
        
        Args:
            audio_path: Path to audio file
            reference: Reference transcription
            use_adaptation: Whether to use speaker adaptation
            user_id: User ID for adaptation
            
        Returns:
            Dictionary with comparison results
        """
        try:
            # Load audio
            audio_data = self.load_audio(audio_path)
            
            # Get audio parameters
            sample_rate = 16000  # Default sample rate
            
            # Transcribe audio
            transcription = self.transcribe_audio(
                audio_data, sample_rate, use_adaptation, user_id
            )
            
            # Calculate WER
            results = self.calculate_wer(reference, transcription)
            
            # Add file information
            results["audio_file"] = audio_path
            results["audio_duration"] = len(audio_data) / sample_rate
            results["use_adaptation"] = use_adaptation
            results["user_id"] = user_id
            
            return results
            
        except Exception as e:
            logger.error(f"Error comparing transcriptions: {str(e)}")
            return {
                "error": str(e),
                "wer": 1.0,  # Maximum error
                "audio_file": audio_path,
                "reference": reference
            }

def load_reference_from_file(file_path: str) -> str:
    """
    Load reference transcript from a file
    
    Args:
        file_path: Path to reference file
        
    Returns:
        Reference text
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Error reading reference file: {str(e)}")
        return ""

# Directory paths
AUDIO_DIRECTORY = os.path.join("D:\\", "medsync", "primock57", "output", "mixed_audio")
TRANSCRIPT_DIRECTORY = os.path.join("D:\\", "medsync", "primock57", "output", "joined_transcripts")

# Results directory
RESULTS_DIRECTORY = os.path.join("tests", "transcription", "results")

def find_matching_files() -> List[Tuple[str, str]]:
    """
    Find matching audio and transcript files from the directories
    
    Returns:
        List of tuples (audio_path, transcript_path)
    """
    matches = []
    
    # Ensure directories exist
    if not os.path.exists(AUDIO_DIRECTORY):
        logger.error(f"Audio directory not found: {AUDIO_DIRECTORY}")
        return []
    
    if not os.path.exists(TRANSCRIPT_DIRECTORY):
        logger.error(f"Transcript directory not found: {TRANSCRIPT_DIRECTORY}")
        return []
    
    # Get all WAV files in the audio directory
    audio_files = [f for f in os.listdir(AUDIO_DIRECTORY) if f.lower().endswith('.wav')]
    logger.info(f"Found {len(audio_files)} audio files in {AUDIO_DIRECTORY}")
    
    # Find matching transcript files
    for audio_file in audio_files:
        # Generate the expected transcript filename
        basename = os.path.splitext(audio_file)[0]
        transcript_file = f"{basename}.txt"
        transcript_path = os.path.join(TRANSCRIPT_DIRECTORY, transcript_file)
        
        # Check if transcript file exists
        if os.path.exists(transcript_path):
            audio_path = os.path.join(AUDIO_DIRECTORY, audio_file)
            matches.append((audio_path, transcript_path))
        else:
            logger.warning(f"No matching transcript found for {audio_file}")
    
    logger.info(f"Found {len(matches)} matching audio-transcript pairs")
    return matches

def process_file_pair(calculator: WERCalculator, audio_path: str, 
                     transcript_path: str, use_adaptation: bool = False, 
                     user_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Process a single audio-transcript pair
    
    Args:
        calculator: WER calculator instance
        audio_path: Path to audio file
        transcript_path: Path to transcript file
        use_adaptation: Whether to use speaker adaptation
        user_id: User ID for adaptation
        
    Returns:
        Results dictionary
    """
    try:
        # Load reference transcript
        reference = load_reference_from_file(transcript_path)
        if not reference:
            logger.error(f"Failed to load reference from {transcript_path}")
            return {
                "error": "Empty reference transcript",
                "audio_file": audio_path,
                "transcript_file": transcript_path,
                "wer": 1.0,
                "success": False
            }
        
        # Run comparison
        start_time = time.time()
        results = calculator.compare_transcriptions(
            audio_path=audio_path,
            reference=reference,
            use_adaptation=use_adaptation,
            user_id=user_id
        )
        processing_time = time.time() - start_time
        
        # Add additional info to results
        results["transcript_file"] = transcript_path
        results["processing_time"] = processing_time
        results["success"] = "error" not in results
        
        return results
        
    except Exception as e:
        logger.error(f"Error processing {audio_path}: {str(e)}")
        return {
            "error": str(e),
            "audio_file": audio_path,
            "transcript_file": transcript_path,
            "wer": 1.0,
            "success": False
        }

def generate_summary_report(results: List[Dict[str, Any]], output_dir: str) -> Dict[str, Any]:
    """
    Generate summary report from test results
    
    Args:
        results: List of results dictionaries
        output_dir: Directory for report output
        
    Returns:
        Summary metrics
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Filter out failures
        successful_results = [r for r in results if r.get("success", False)]
        
        # Basic statistics
        if not successful_results:
            logger.error("No successful results to generate report from")
            return {"error": "No successful results"}
        
        # Calculate summary metrics
        summary = {
            "total_files": len(results),
            "successful_files": len(successful_results),
            "failed_files": len(results) - len(successful_results),
            "total_audio_duration": sum(r.get("audio_duration", 0) for r in successful_results),
            "total_processing_time": sum(r.get("processing_time", 0) for r in successful_results),
            "average_wer": sum(r.get("wer", 0) for r in successful_results) / len(successful_results),
            "average_cer": sum(r.get("cer", 0) for r in successful_results) / len(successful_results),
            "best_wer": min(r.get("wer", 1.0) for r in successful_results),
            "worst_wer": max(r.get("wer", 0.0) for r in successful_results),
            "total_words": sum(r.get("reference_length", 0) for r in successful_results),
            "total_correct_words": sum(r.get("hits", 0) for r in successful_results),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Calculate accuracy percentage
        if summary["total_words"] > 0:
            summary["overall_accuracy"] = (summary["total_correct_words"] / summary["total_words"]) * 100
        else:
            summary["overall_accuracy"] = 0
        
        # Save summary to JSON
        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved to {summary_path}")
        
        # Create detailed results table
        results_df = pd.DataFrame([
            {
                "file": os.path.basename(r.get("audio_file", "")),
                "wer": r.get("wer", 1.0),
                "cer": r.get("cer", 1.0),
                "substitutions": r.get("substitutions", 0),
                "deletions": r.get("deletions", 0), 
                "insertions": r.get("insertions", 0),
                "correct_words": r.get("hits", 0),
                "reference_length": r.get("reference_length", 0),
                "duration": r.get("audio_duration", 0)
            }
            for r in successful_results
        ])
        
        # Sort by WER (best to worst)
        results_df = results_df.sort_values(by=["wer"])
        
        # Save to CSV
        csv_path = os.path.join(output_dir, "detailed_results.csv")
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Detailed results saved to {csv_path}")
        
        # Generate visualization of WER distribution
        plt.figure(figsize=(12, 6))
        
        # WER distribution histogram
        plt.subplot(1, 2, 1)
        plt.hist(results_df["wer"], bins=20, alpha=0.7, color='blue')
        plt.axvline(summary["average_wer"], color='red', linestyle='dashed', linewidth=2)
        plt.xlabel('Word Error Rate (WER)')
        plt.ylabel('Number of Files')
        plt.title('WER Distribution')
        
        # Top 10 best and worst files
        plt.subplot(1, 2, 2)
        
        # Get top 5 best and worst
        best_5 = results_df.head(5)
        worst_5 = results_df.tail(5)
        
        # Combine and plot
        comparison_df = pd.concat([best_5, worst_5])
        bars = plt.barh(comparison_df["file"], comparison_df["wer"], color=['green']*5 + ['red']*5)
        plt.xlabel('Word Error Rate (WER)')
        plt.title('Best vs Worst Files')
        plt.tight_layout()
        
        # Save figure
        chart_path = os.path.join(output_dir, "wer_distribution.png")
        plt.savefig(chart_path)
        logger.info(f"Visualization saved to {chart_path}")
        
        return summary
        
    except Exception as e:
        logger.error(f"Error generating summary report: {str(e)}")
        return {"error": str(e)}

def main():
    """Main function for command-line execution"""
    parser = argparse.ArgumentParser(description="Test transcription accuracy across multiple files")
    parser.add_argument('--adaptation', action='store_true', help='Use speaker adaptation')
    parser.add_argument('--user_id', type=int, help='User ID for adaptation')
    parser.add_argument('--output_dir', default=RESULTS_DIRECTORY, help='Directory for test results')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--limit', type=int, help='Limit the number of files to process')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Find matching files
    file_pairs = find_matching_files()
    
    if not file_pairs:
        logger.error("No matching audio-transcript pairs found")
        sys.exit(1)
    
    # Apply limit if specified
    if args.limit and args.limit > 0:
        file_pairs = file_pairs[:args.limit]
        logger.info(f"Limited to processing {args.limit} files")
    
    # Create WER calculator
    calculator = WERCalculator()
    
    # Process all file pairs
    all_results = []
    for audio_path, transcript_path in tqdm(file_pairs, desc="Processing files"):
        logger.info(f"Processing: {os.path.basename(audio_path)}")
        
        result = process_file_pair(
            calculator=calculator,
            audio_path=audio_path,
            transcript_path=transcript_path,
            use_adaptation=args.adaptation,
            user_id=args.user_id
        )
        
        all_results.append(result)
        
        # Log result for this file
        if result.get("success", False):
            logger.info(f"WER: {result['wer']:.4f}, CER: {result['cer']:.4f}, " +
                       f"Correct words: {result['hits']}/{result['reference_length']}")
        else:
            logger.error(f"Failed: {result.get('error', 'Unknown error')}")
        break

    # Generate summary report
    summary = generate_summary_report(all_results, args.output_dir)
    
    # Print overall results
    if "error" not in summary:
        logger.info("=" * 50)
        logger.info(f"TRANSCRIPTION TEST SUMMARY:")
        logger.info(f"Files processed: {summary['successful_files']}/{summary['total_files']}")
        logger.info(f"Average WER: {summary['average_wer']:.4f}")
        logger.info(f"Average CER: {summary['average_cer']:.4f}")
        logger.info(f"Overall word accuracy: {summary['overall_accuracy']:.2f}%")
        logger.info(f"Best WER: {summary['best_wer']:.4f}")
        logger.info(f"Worst WER: {summary['worst_wer']:.4f}")
        logger.info(f"Total audio duration: {summary['total_audio_duration']:.2f} seconds")
        logger.info(f"Total processing time: {summary['total_processing_time']:.2f} seconds")
        logger.info("=" * 50)
        logger.info(f"Detailed results saved to: {args.output_dir}")
    else:
        logger.error(f"Failed to generate summary: {summary['error']}")

if __name__ == "__main__":
    main()
