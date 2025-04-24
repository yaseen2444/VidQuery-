import logging
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from youtube_transcript_api.formatters import TextFormatter
from langdetect import detect, LangDetectException

logger = logging.getLogger(__name__)

def fetch_transcript(video_id):
    """
    Fetch transcript from YouTube video
    Returns tuple of (transcript_text, language_code)
    """
    logger.info(f"Fetching transcript for video ID: {video_id}")
    
    try:
        # Get all available transcripts
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = None
        
        # Try to get English transcript first
        try:
            # Try to find an English transcript
            transcript = transcript_list.find_transcript(['en', 'en-US', 'en-GB'])
            logger.info("Found English transcript")
        except NoTranscriptFound:
            logger.info("No English transcript found, looking for any available transcript")
            
            # If no English transcript, get the first available and translate it to English
            try:
                # Get the first available transcript (manual or auto-generated)
                transcript = list(transcript_list)[0]
                
                # If it's not English, translate it
                if transcript.language_code not in ['en', 'en-US', 'en-GB']:
                    logger.info(f"Translating transcript from {transcript.language_code} to English")
                    transcript = transcript.translate('en')
            except IndexError:
                raise NoTranscriptFound("No transcripts available for this video")
        
        # Get the transcript data
        transcript_data = transcript.fetch()
        
        # Format transcript to text
        formatter = TextFormatter()
        transcript_text = formatter.format_transcript(transcript_data)
        
        # Detect language
        try:
            detected_lang = detect(transcript_text[:1000])  # Use first 1000 chars for detection
        except LangDetectException:
            detected_lang = transcript.language_code  # Use transcript language if detection fails
            
        logger.info(f"Transcript fetched successfully, language detected: {detected_lang}")
        
        return (transcript_text, detected_lang)
        
    except TranscriptsDisabled:
        logger.error(f"Transcripts are disabled for video {video_id}")
        raise Exception("Transcripts are disabled for this video.")
        
    except NoTranscriptFound:
        logger.error(f"No transcript found for video {video_id}")
        raise Exception("No transcript found for this video.")
        
    except Exception as e:
        logger.exception(f"Error fetching transcript: {e}")
        raise Exception(f"Error fetching transcript: {str(e)}")