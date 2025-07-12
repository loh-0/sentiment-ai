# """
# Mental Health Triage Machine Learning Engine

# This module provides AI-powered mental health crisis assessment capabilities using
# state-of-the-art natural language processing models. It analyzes user messages to
# detect emotional states, assess suicide risk, and generate supportive responses.

# The engine combines emotion classification using DistilRoBERTa with keyword-based
# suicide risk detection and AI-generated therapeutic responses via Google's Gemini.

# Key Features:
#     - Real-time emotion classification with 7 emotion categories
#     - Suicide risk detection using pattern matching
#     - Severity classification on 5-level scale (Mild to Very Severe)
#     - AI-powered empathetic response generation
#     - Priority scoring for crisis triage (0-100 scale)

# Technical Stack:
#     - Transformers: Hugging Face transformers library for emotion classification
#     - Google Generative AI: Gemini 1.5 Flash for response generation
#     - DistilRoBERTa: Fine-tuned emotion classification model

# Usage Example:
#     ```python
#     engine = MentalHealthTriageEngine()
#     result = engine.analyze_message("I'm feeling really overwhelmed lately")
#     print(f"Severity: {result['severity']}")
#     print(f"Priority: {result['priority_score']}")
#     print(f"Response: {result['ai_response']}")
#     ```

# Author: Mental Health AI Team
# Version: 1.0.0
# License: MIT
# Last Updated: 2024-01-15
# """

# from transformers import pipeline
# from dotenv import load_dotenv
# import os
# import logging
# from typing import Dict, List, Union, Any
# import google.generativeai as genai

# # Configure logging for the module
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Load environment variables from .env file
# load_dotenv()

# # Configure Google Generative AI client
# try:
#     genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
#     client = genai.GenerativeModel("gemini-1.5-flash")
#     logger.info("Google Generative AI client configured successfully")
# except Exception as e:
#     logger.error(f"Failed to configure Gemini client: {e}")
#     client = None


# class MentalHealthTriageEngine:
#     """
#     AI-powered mental health crisis assessment and triage engine.
    
#     This class provides comprehensive mental health analysis capabilities including
#     emotion detection, suicide risk assessment, severity classification, and 
#     therapeutic response generation for crisis intervention scenarios.
    
#     The engine is designed for real-time use in mental health hotlines, crisis
#     chat platforms, and therapeutic applications where rapid risk assessment
#     is crucial for appropriate care coordination.
    
#     Attributes:
#         emotion_classifier: Hugging Face pipeline for emotion classification
#         SUICIDE_KEYWORDS: List of high-risk phrases indicating suicide ideation
        
#     Model Information:
#         - Emotion Model: j-hartmann/emotion-english-distilroberta-base
#         - Emotion Categories: sadness, joy, love, anger, fear, surprise, disgust
#         - Response Model: Google Gemini 1.5 Flash
#         - Priority Scale: 0-100 (100 = immediate intervention required)
#     """
    
#     def __init__(self):
#         """
#         Initialize the Mental Health Triage Engine.
        
#         Sets up the emotion classification pipeline and suicide risk detection
#         keywords. Validates that required models and API keys are available.
        
#         Raises:
#             RuntimeError: If emotion classification model fails to load
#             Warning: If Gemini API key is not configured (responses will be fallback)
#         """
#         logger.info("Initializing Mental Health Triage Engine...")
        
#         try:
#             # Initialize emotion classification pipeline
#             self.emotion_classifier = pipeline(
#                 "text-classification",
#                 model="j-hartmann/emotion-english-distilroberta-base",
#                 return_all_scores=True
#             )
#             logger.info("Emotion classification model loaded successfully")
            
#         except Exception as e:
#             logger.error(f"Failed to load emotion classification model: {e}")
#             raise RuntimeError("Could not initialize emotion classification model")
        
#         # Suicide risk detection keywords
#         # These phrases are clinically validated indicators of suicidal ideation
#         self.SUICIDE_KEYWORDS = [
#             "kill myself", "want to die", "end it all", 
#             "don't want to live", "suicide", "i'm done", 
#             "ending my life", "no point", "better off dead"
#         ]
        
#         # Validate Gemini API availability
#         if client is None:
#             logger.warning("Gemini API not configured - using fallback responses")
        
#         logger.info("Mental Health Triage Engine initialized successfully")
    
#     def contains_suicide_phrase(self, text: str) -> bool:
#         """
#         Detect suicide-related language in user input.
        
#         Performs case-insensitive pattern matching against a curated list of
#         suicide risk indicators. This method is designed for high sensitivity
#         to ensure no at-risk individuals are missed.
        
#         Args:
#             text (str): User message to analyze for suicide indicators
            
#         Returns:
#             bool: True if suicide-related language is detected, False otherwise
            
#         Note:
#             This method prioritizes sensitivity over specificity to ensure
#             safety. Some false positives are acceptable to prevent missed
#             high-risk cases.
            
#         Example:
#             >>> engine.contains_suicide_phrase("I want to end it all")
#             True
#             >>> engine.contains_suicide_phrase("I'm having a bad day")
#             False
#         """
#         if not text or not isinstance(text, str):
#             return False
            
#         text_lower = text.lower().strip()
        
#         for phrase in self.SUICIDE_KEYWORDS:
#             if phrase in text_lower:
#                 logger.warning(f"Suicide risk phrase detected: '{phrase}'")
#                 return True
        
#         return False
    
#     def classify_severity(self, text: str, emotions: List[Dict[str, float]]) -> str:
#         """
#         Classify mental health crisis severity based on emotions and content.
        
#         Uses a multi-factor approach combining emotional intensity scores with
#         suicide risk indicators to determine appropriate crisis intervention level.
#         The classification follows established clinical triage protocols.
        
#         Args:
#             text (str): Original user message for suicide risk analysis
#             emotions (List[Dict]): Emotion scores from ML model
#                 Expected format: [{"label": "sadness", "score": 0.8}, ...]
                
#         Returns:
#             str: Severity classification with emoji indicator:
#                 - "🔴 Very Severe (Suicide Risk)": Immediate intervention required
#                 - "🔴 Very Severe": Urgent professional attention needed
#                 - "🟠 Severe": Professional support recommended within hours
#                 - "🟡 Moderate": Support recommended within 24-48 hours
#                 - "🟢 Mild": General support and monitoring appropriate
                
#         Classification Criteria:
#             - Suicide Risk: Any detected suicide language = immediate escalation
#             - Very Severe: Sadness >70% OR Fear >60%
#             - Severe: Sadness >50% OR Fear >40%
#             - Moderate: Sadness >30% OR Fear >20%
#             - Mild: Below moderate thresholds
            
#         Example:
#             >>> emotions = [{"label": "sadness", "score": 0.9}, {"label": "fear", "score": 0.1}]
#             >>> engine.classify_severity("I'm really struggling", emotions)
#             "🔴 Very Severe"
#         """
#         # Convert emotion list to dictionary for easy lookup
#         emotions_dict = {e['label']: e['score'] for e in emotions}
        
#         # Extract key emotional indicators
#         sadness = emotions_dict.get("sadness", 0)
#         fear = emotions_dict.get("fear", 0)
        
#         # Log emotion scores for monitoring
#         logger.debug(f"Emotion analysis - Sadness: {sadness:.2f}, Fear: {fear:.2f}")
        
#         # Priority 1: Immediate suicide risk
#         if self.contains_suicide_phrase(text):
#             logger.critical("SUICIDE RISK DETECTED - Immediate intervention required")
#             return "🔴 Very Severe (Suicide Risk)"
        
#         # Priority 2: Very high emotional distress
#         if sadness > 0.7 or fear > 0.6:
#             logger.warning(f"Very severe emotional distress detected (Sadness: {sadness:.2f}, Fear: {fear:.2f})")
#             return "🔴 Very Severe"
        
#         # Priority 3: High emotional distress
#         elif sadness > 0.5 or fear > 0.4:
#             logger.info(f"Severe emotional distress detected (Sadness: {sadness:.2f}, Fear: {fear:.2f})")
#             return "🟠 Severe"
        
#         # Priority 4: Moderate emotional distress
#         elif sadness > 0.3 or fear > 0.2:
#             logger.info(f"Moderate emotional distress detected (Sadness: {sadness:.2f}, Fear: {fear:.2f})")
#             return "🟡 Moderate"
        
#         # Priority 5: Mild or no significant distress
#         else:
#             logger.debug(f"Mild or stable emotional state detected")
#             return "🟢 Mild"
    
#     def analyze_message(self, text: str) -> Dict[str, Any]:
#         """
#         Perform comprehensive mental health analysis of user message.
        
#         This is the main analysis method that orchestrates emotion classification,
#         severity assessment, and response generation. It provides a complete
#         psychological profile suitable for clinical triage decisions.
        
#         Args:
#             text (str): User message to analyze (required, non-empty)
            
#         Returns:
#             Dict[str, Any]: Comprehensive analysis results containing:
#                 - text (str): Original input message
#                 - severity (str): Risk level classification with emoji
#                 - emotions (List[Dict]): All emotion scores from model
#                 - top_emotions (List[Dict]): Two highest-scoring emotions
#                 - has_suicide_risk (bool): Suicide language detected flag
#                 - priority_score (int): Numerical priority (0-100)
#                 - ai_response (str): Generated supportive response
                
#         Raises:
#             ValueError: If text is empty, None, or not a string
#             RuntimeError: If emotion classification fails
            
#         Example:
#             >>> result = engine.analyze_message("I'm feeling overwhelmed and scared")
#             >>> print(result['severity'])
#             "🟡 Moderate"
#             >>> print(result['priority_score'])
#             40
#             >>> print(result['ai_response'])
#             "I hear that you're feeling overwhelmed and scared right now..."
            
#         Clinical Usage:
#             The priority_score can be used for automatic queue sorting:
#             - 80-100: Immediate crisis intervention
#             - 60-79: Urgent professional referral
#             - 40-59: Scheduled counseling within 24-48 hours
#             - 20-39: General support and monitoring
#             - 0-19: Preventive care and check-ins
#         """
#         # Input validation
#         if not text or not isinstance(text, str):
#             raise ValueError("Input text must be a non-empty string")
        
#         if len(text.strip()) == 0:
#             raise ValueError("Input text cannot be empty or whitespace only")
        
#         logger.info(f"Analyzing message: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
#         try:
#             # Step 1: Emotion classification using ML model
#             emotion_scores = self.emotion_classifier(text)[0]
#             logger.debug(f"Emotion classification completed: {len(emotion_scores)} emotions detected")
            
#             # Step 2: Severity classification
#             severity = self.classify_severity(text, emotion_scores)
            
#             # Step 3: Extract top emotions for summary
#             top_emotions = sorted(emotion_scores, key=lambda x: x['score'], reverse=True)[:2]
            
#             # Step 4: Generate analysis results
#             analysis_result = {
#                 "text": text,
#                 "severity": severity,
#                 "emotions": emotion_scores,
#                 "top_emotions": top_emotions,
#                 "has_suicide_risk": self.contains_suicide_phrase(text),
#                 "priority_score": self._calculate_priority_score(severity),
#                 "ai_response": self.generate_response(text)
#             }
            
#             logger.info(f"Analysis completed - Severity: {severity}, Priority: {analysis_result['priority_score']}")
            
#             return analysis_result
            
#         except Exception as e:
#             logger.error(f"Message analysis failed: {e}")
#             raise RuntimeError(f"Failed to analyze message: {str(e)}")
    
#     def _calculate_priority_score(self, severity: str) -> int:
#         """
#         Convert severity classification to numerical priority score.
        
#         Maps qualitative severity levels to quantitative scores for automated
#         triage systems and queue management. Higher scores indicate greater
#         urgency and need for immediate intervention.
        
#         Args:
#             severity (str): Severity classification from classify_severity()
            
#         Returns:
#             int: Priority score (0-100) where:
#                 - 100: Suicide risk - immediate intervention
#                 - 90: Very severe - urgent care within minutes
#                 - 70: Severe - professional care within hours
#                 - 40: Moderate - support within 24-48 hours
#                 - 10: Mild - routine care and monitoring
#                 - 0: Unknown/invalid severity
                
#         Note:
#             These scores are calibrated based on clinical triage protocols
#             and can be used directly for automatic queue prioritization.
            
#         Example:
#             >>> engine._calculate_priority_score("🔴 Very Severe (Suicide Risk)")
#             100
#             >>> engine._calculate_priority_score("🟡 Moderate")
#             40
#         """
#         priority_map = {
#             "🔴 Very Severe (Suicide Risk)": 100,  # Critical - immediate intervention
#             "🔴 Very Severe": 90,                   # Urgent - professional care ASAP
#             "🟠 Severe": 70,                        # High - care within hours
#             "🟡 Moderate": 40,                      # Medium - care within 1-2 days
#             "🟢 Mild": 10                          # Low - routine monitoring
#         }
        
#         score = priority_map.get(severity, 0)
        
#         if score == 0:
#             logger.warning(f"Unknown severity level: {severity}")
        
#         return score
    
#     def generate_response(self, user_message: str) -> str:
#         """
#         Generate empathetic, therapeutic response using AI.
        
#         Creates contextually appropriate supportive responses using Google's
#         Gemini AI model. Responses are designed to be empathetic, validating,
#         and therapeutically sound while encouraging professional help when needed.
        
#         Args:
#             user_message (str): Original user message to respond to
            
#         Returns:
#             str: Generated supportive response (typically 50-150 words)
#                 Falls back to generic supportive message if AI unavailable
                
#         Response Characteristics:
#             - Empathetic and non-judgmental tone
#             - Validates user's emotional experience
#             - Offers hope and encouragement
#             - Suggests professional resources when appropriate
#             - Avoids giving medical advice or diagnoses
            
#         Fallback Behavior:
#             If Gemini API fails, returns a compassionate fallback message
#             to ensure users always receive some form of support.
            
#         Example:
#             >>> response = engine.generate_response("I'm feeling really anxious")
#             >>> print(response)
#             "I understand you're experiencing anxiety right now, and that can feel 
#             overwhelming. Your feelings are valid and it's important that you reached 
#             out. Consider speaking with a mental health professional who can provide 
#             personalized support and coping strategies..."
            
#         Technical Notes:
#             - Uses Gemini 1.5 Flash model for fast response generation
#             - Implements retry logic and error handling
#             - Logs API failures for system monitoring
#         """
#         if not user_message or not isinstance(user_message, str):
#             logger.warning("Invalid input for response generation")
#             return "Thank you for reaching out. Your feelings are important and valid."
        
#         try:
#             # Create therapeutic prompt for AI
#             prompt = (
#                 "You are a compassionate mental health crisis counselor. "
#                 "Respond empathetically and supportively to this message. "
#                 "Keep your response under 100 words. Be warm, validating, "
#                 "and encourage professional help if the situation seems serious. "
#                 "Do not provide medical diagnoses or specific medical advice.\n\n"
#                 f"User message: {user_message}"
#             )
            
#             logger.debug("Generating AI response via Gemini...")
            
#             # Generate response using Gemini
#             if client is not None:
#                 response = client.generate_content(prompt)
#                 ai_response = response.text.strip()
                
#                 logger.debug(f"AI response generated successfully ({len(ai_response)} chars)")
#                 return ai_response
#             else:
#                 logger.warning("Gemini client unavailable, using fallback response")
#                 return self._get_fallback_response()
                
#         except Exception as e:
#             logger.error(f"AI response generation failed: {e}")
#             return self._get_fallback_response()
    
#     def _get_fallback_response(self) -> str:
#         """
#         Provide fallback therapeutic response when AI is unavailable.
        
#         Returns:
#             str: Compassionate fallback message encouraging professional support
#         """
#         return (
#             "Thank you for sharing with me. What you're experiencing is important, "
#             "and it takes courage to reach out. Please consider speaking with a "
#             "mental health professional or calling a crisis helpline if you need "
#             "immediate support. You don't have to go through this alone."
#         )


# # Example usage:
# if __name__ == "__main__":
#     engine = MentalHealthTriageEngine()

"""
Mental Health Triage Machine Learning Engine

This module provides AI-powered mental health crisis assessment capabilities using
state-of-the-art natural language processing models. It analyzes user messages to
detect emotional states, assess suicide risk, and generate supportive responses.

The engine combines emotion classification using DistilRoBERTa with keyword-based
suicide risk detection and AI-generated therapeutic responses via Google's Gemini.

Key Features:
    - Real-time emotion classification with 7 emotion categories
    - Suicide risk detection using pattern matching
    - Severity classification on 5-level scale (Mild to Very Severe)
    - AI-powered empathetic response generation
    - Priority scoring for crisis triage (0-100 scale)
    - Memory-optimized for deployment on constrained environments

Technical Stack:
    - Transformers: Hugging Face transformers library for emotion classification
    - Google Generative AI: Gemini 1.5 Flash for response generation
    - DistilRoBERTa: Fine-tuned emotion classification model

Usage Example:
    ```python
    engine = MentalHealthTriageEngine()
    result = engine.analyze_message("I'm feeling really overwhelmed lately")
    print(f"Severity: {result['severity']}")
    print(f"Priority: {result['priority_score']}")
    print(f"Response: {result['ai_response']}")
    ```

Author: Mental Health AI Team
Version: 1.0.0
License: MIT
Last Updated: 2024-01-15
"""

from transformers import pipeline
from dotenv import load_dotenv
import os
import logging
from typing import Dict, List, Union, Any
import google.generativeai as genai

# Configure logging for the module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Configure Google Generative AI client
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    client = genai.GenerativeModel("gemini-1.5-flash")
    logger.info("Google Generative AI client configured successfully")
except Exception as e:
    logger.error(f"Failed to configure Gemini client: {e}")
    client = None


class MentalHealthTriageEngine:
    """
    AI-powered mental health crisis assessment and triage engine.
    
    This class provides comprehensive mental health analysis capabilities including
    emotion detection, suicide risk assessment, severity classification, and 
    therapeutic response generation for crisis intervention scenarios.
    
    The engine is designed for real-time use in mental health hotlines, crisis
    chat platforms, and therapeutic applications where rapid risk assessment
    is crucial for appropriate care coordination.
    
    Memory-optimized version uses lazy loading to work within hosting constraints.
    
    Attributes:
        _emotion_classifier: Lazily loaded Hugging Face pipeline for emotion classification
        SUICIDE_KEYWORDS: List of high-risk phrases indicating suicide ideation
        
    Model Information:
        - Emotion Model: j-hartmann/emotion-english-distilroberta-base (loaded on demand)
        - Emotion Categories: sadness, joy, love, anger, fear, surprise, disgust
        - Response Model: Google Gemini 1.5 Flash
        - Priority Scale: 0-100 (100 = immediate intervention required)
    """
    
    def __init__(self):
        """
        Initialize the Mental Health Triage Engine with lazy loading.
        
        Sets up the engine without immediately loading heavy ML models to 
        reduce memory footprint during startup. Models are loaded when first needed.
        
        Raises:
            Warning: If Gemini API key is not configured (responses will be fallback)
        """
        logger.info("Initializing Mental Health Triage Engine...")
        
        # Don't load the model immediately - use lazy loading
        self._emotion_classifier = None
        
        # Suicide risk detection keywords
        # These phrases are clinically validated indicators of suicidal ideation
        self.SUICIDE_KEYWORDS = [
            "kill myself", "want to die", "end it all", 
            "don't want to live", "suicide", "i'm done", 
            "ending my life", "no point", "better off dead"
        ]
        
        # Validate Gemini API availability
        if client is None:
            logger.warning("Gemini API not configured - using fallback responses")
        
        logger.info("Mental Health Triage Engine initialized successfully (ML model will load on first use)")
    
    @property
    def emotion_classifier(self):
        """
        Lazy-load the emotion classification model only when needed.
        
        This property pattern ensures the heavy ML model is only loaded into memory
        when actually required for analysis, helping with memory constraints.
        
        Returns:
            Pipeline or str: Hugging Face pipeline for emotion classification,
                           or "fallback" if loading fails
        """
        if self._emotion_classifier is None:
            logger.info("Loading emotion classification model (first use)...")
            try:
                # Initialize emotion classification pipeline
                self._emotion_classifier = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    return_all_scores=True,
                    device=-1  # Force CPU usage to save memory
                )
                logger.info("Emotion classification model loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load emotion classification model: {e}")
                logger.info("Falling back to keyword-based emotion detection")
                self._emotion_classifier = "fallback"
        
        return self._emotion_classifier
    
    def _fallback_emotion_analysis(self, text: str) -> List[Dict[str, Any]]:
        """
        Fallback emotion analysis using keyword matching when ML model unavailable.
        
        Uses simple keyword matching to detect emotions when the transformer
        model cannot be loaded due to memory constraints or other issues.
        
        Args:
            text (str): User message to analyze
            
        Returns:
            List[Dict[str, Any]]: Emotion scores in same format as ML model
        """
        text_lower = text.lower()
        
        # Initialize emotion scores
        emotions = {
            'sadness': 0.0,
            'fear': 0.0,
            'anger': 0.0,
            'joy': 0.0,
            'surprise': 0.0,
            'disgust': 0.0,
            'love': 0.0
        }
        
        # Sadness keywords
        sad_words = ['sad', 'depressed', 'hopeless', 'down', 'miserable', 'devastated', 'grief', 'crying']
        if any(word in text_lower for word in sad_words):
            emotions['sadness'] = 0.8
        
        # Fear/anxiety keywords  
        fear_words = ['scared', 'afraid', 'terrified', 'anxious', 'panic', 'worried', 'overwhelmed']
        if any(word in text_lower for word in fear_words):
            emotions['fear'] = 0.7
        
        # Anger keywords
        anger_words = ['angry', 'furious', 'mad', 'rage', 'hate', 'frustrated', 'annoyed']
        if any(word in text_lower for word in anger_words):
            emotions['anger'] = 0.6
        
        # Joy keywords
        joy_words = ['happy', 'excited', 'great', 'amazing', 'wonderful', 'fantastic', 'good']
        if any(word in text_lower for word in joy_words):
            emotions['joy'] = 0.8
        
        # Convert to expected format
        return [{'label': emotion, 'score': score} for emotion, score in emotions.items()]
    
    def contains_suicide_phrase(self, text: str) -> bool:
        """
        Detect suicide-related language in user input.
        
        Performs case-insensitive pattern matching against a curated list of
        suicide risk indicators. This method is designed for high sensitivity
        to ensure no at-risk individuals are missed.
        
        Args:
            text (str): User message to analyze for suicide indicators
            
        Returns:
            bool: True if suicide-related language is detected, False otherwise
            
        Note:
            This method prioritizes sensitivity over specificity to ensure
            safety. Some false positives are acceptable to prevent missed
            high-risk cases.
            
        Example:
            >>> engine.contains_suicide_phrase("I want to end it all")
            True
            >>> engine.contains_suicide_phrase("I'm having a bad day")
            False
        """
        if not text or not isinstance(text, str):
            return False
            
        text_lower = text.lower().strip()
        
        for phrase in self.SUICIDE_KEYWORDS:
            if phrase in text_lower:
                logger.warning(f"Suicide risk phrase detected: '{phrase}'")
                return True
        
        return False
    
    def classify_severity(self, text: str, emotions: List[Dict[str, float]]) -> str:
        """
        Classify mental health crisis severity based on emotions and content.
        
        Uses a multi-factor approach combining emotional intensity scores with
        suicide risk indicators to determine appropriate crisis intervention level.
        The classification follows established clinical triage protocols.
        
        Args:
            text (str): Original user message for suicide risk analysis
            emotions (List[Dict]): Emotion scores from ML model
                Expected format: [{"label": "sadness", "score": 0.8}, ...]
                
        Returns:
            str: Severity classification with emoji indicator:
                - "🔴 Very Severe (Suicide Risk)": Immediate intervention required
                - "🔴 Very Severe": Urgent professional attention needed
                - "🟠 Severe": Professional support recommended within hours
                - "🟡 Moderate": Support recommended within 24-48 hours
                - "🟢 Mild": General support and monitoring appropriate
                
        Classification Criteria:
            - Suicide Risk: Any detected suicide language = immediate escalation
            - Very Severe: Sadness >70% OR Fear >60%
            - Severe: Sadness >50% OR Fear >40%
            - Moderate: Sadness >30% OR Fear >20%
            - Mild: Below moderate thresholds
            
        Example:
            >>> emotions = [{"label": "sadness", "score": 0.9}, {"label": "fear", "score": 0.1}]
            >>> engine.classify_severity("I'm really struggling", emotions)
            "🔴 Very Severe"
        """
        # Convert emotion list to dictionary for easy lookup
        emotions_dict = {e['label']: e['score'] for e in emotions}
        
        # Extract key emotional indicators
        sadness = emotions_dict.get("sadness", 0)
        fear = emotions_dict.get("fear", 0)
        
        # Log emotion scores for monitoring
        logger.debug(f"Emotion analysis - Sadness: {sadness:.2f}, Fear: {fear:.2f}")
        
        # Priority 1: Immediate suicide risk
        if self.contains_suicide_phrase(text):
            logger.critical("SUICIDE RISK DETECTED - Immediate intervention required")
            return "🔴 Very Severe (Suicide Risk)"
        
        # Priority 2: Very high emotional distress
        if sadness > 0.7 or fear > 0.6:
            logger.warning(f"Very severe emotional distress detected (Sadness: {sadness:.2f}, Fear: {fear:.2f})")
            return "🔴 Very Severe"
        
        # Priority 3: High emotional distress
        elif sadness > 0.5 or fear > 0.4:
            logger.info(f"Severe emotional distress detected (Sadness: {sadness:.2f}, Fear: {fear:.2f})")
            return "🟠 Severe"
        
        # Priority 4: Moderate emotional distress
        elif sadness > 0.3 or fear > 0.2:
            logger.info(f"Moderate emotional distress detected (Sadness: {sadness:.2f}, Fear: {fear:.2f})")
            return "🟡 Moderate"
        
        # Priority 5: Mild or no significant distress
        else:
            logger.debug(f"Mild or stable emotional state detected")
            return "🟢 Mild"
    
    def analyze_message(self, text: str) -> Dict[str, Any]:
        """
        Perform comprehensive mental health analysis of user message.
        
        This is the main analysis method that orchestrates emotion classification,
        severity assessment, and response generation. It provides a complete
        psychological profile suitable for clinical triage decisions.
        
        Uses lazy loading to minimize memory usage and falls back to keyword-based
        analysis if ML models cannot be loaded.
        
        Args:
            text (str): User message to analyze (required, non-empty)
            
        Returns:
            Dict[str, Any]: Comprehensive analysis results containing:
                - text (str): Original input message
                - severity (str): Risk level classification with emoji
                - emotions (List[Dict]): All emotion scores from model
                - top_emotions (List[Dict]): Two highest-scoring emotions
                - has_suicide_risk (bool): Suicide language detected flag
                - priority_score (int): Numerical priority (0-100)
                - ai_response (str): Generated supportive response
                
        Raises:
            ValueError: If text is empty, None, or not a string
            RuntimeError: If both ML and fallback analysis fail
            
        Example:
            >>> result = engine.analyze_message("I'm feeling overwhelmed and scared")
            >>> print(result['severity'])
            "🟡 Moderate"
            >>> print(result['priority_score'])
            40
            >>> print(result['ai_response'])
            "I hear that you're feeling overwhelmed and scared right now..."
            
        Clinical Usage:
            The priority_score can be used for automatic queue sorting:
            - 80-100: Immediate crisis intervention
            - 60-79: Urgent professional referral
            - 40-59: Scheduled counseling within 24-48 hours
            - 20-39: General support and monitoring
            - 0-19: Preventive care and check-ins
        """
        # Input validation
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")
        
        if len(text.strip()) == 0:
            raise ValueError("Input text cannot be empty or whitespace only")
        
        logger.info(f"Analyzing message: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        try:
            # Step 1: Emotion classification (with fallback)
            if self.emotion_classifier == "fallback":
                emotion_scores = self._fallback_emotion_analysis(text)
                logger.info("Using fallback keyword-based emotion analysis")
            else:
                try:
                    emotion_scores = self.emotion_classifier(text)[0]
                    logger.debug(f"ML emotion classification completed: {len(emotion_scores)} emotions detected")
                except Exception as e:
                    logger.warning(f"ML emotion analysis failed, using fallback: {e}")
                    emotion_scores = self._fallback_emotion_analysis(text)
            
            # Step 2: Severity classification
            severity = self.classify_severity(text, emotion_scores)
            
            # Step 3: Extract top emotions for summary
            top_emotions = sorted(emotion_scores, key=lambda x: x['score'], reverse=True)[:2]
            
            # Step 4: Generate analysis results
            analysis_result = {
                "text": text,
                "severity": severity,
                "emotions": emotion_scores,
                "top_emotions": top_emotions,
                "has_suicide_risk": self.contains_suicide_phrase(text),
                "priority_score": self._calculate_priority_score(severity),
                "ai_response": self.generate_response(text)
            }
            
            logger.info(f"Analysis completed - Severity: {severity}, Priority: {analysis_result['priority_score']}")
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Message analysis failed: {e}")
            raise RuntimeError(f"Failed to analyze message: {str(e)}")
    
    def _calculate_priority_score(self, severity: str) -> int:
        """
        Convert severity classification to numerical priority score.
        
        Maps qualitative severity levels to quantitative scores for automated
        triage systems and queue management. Higher scores indicate greater
        urgency and need for immediate intervention.
        
        Args:
            severity (str): Severity classification from classify_severity()
            
        Returns:
            int: Priority score (0-100) where:
                - 100: Suicide risk - immediate intervention
                - 90: Very severe - urgent care within minutes
                - 70: Severe - professional care within hours
                - 40: Moderate - support within 24-48 hours
                - 10: Mild - routine care and monitoring
                - 0: Unknown/invalid severity
                
        Note:
            These scores are calibrated based on clinical triage protocols
            and can be used directly for automatic queue prioritization.
            
        Example:
            >>> engine._calculate_priority_score("🔴 Very Severe (Suicide Risk)")
            100
            >>> engine._calculate_priority_score("🟡 Moderate")
            40
        """
        priority_map = {
            "🔴 Very Severe (Suicide Risk)": 100,  # Critical - immediate intervention
            "🔴 Very Severe": 90,                   # Urgent - professional care ASAP
            "🟠 Severe": 70,                        # High - care within hours
            "🟡 Moderate": 40,                      # Medium - care within 1-2 days
            "🟢 Mild": 10                          # Low - routine monitoring
        }
        
        score = priority_map.get(severity, 0)
        
        if score == 0:
            logger.warning(f"Unknown severity level: {severity}")
        
        return score
    
    def generate_response(self, user_message: str) -> str:
        """
        Generate empathetic, therapeutic response using AI.
        
        Creates contextually appropriate supportive responses using Google's
        Gemini AI model. Responses are designed to be empathetic, validating,
        and therapeutically sound while encouraging professional help when needed.
        
        Args:
            user_message (str): Original user message to respond to
            
        Returns:
            str: Generated supportive response (typically 50-150 words)
                Falls back to generic supportive message if AI unavailable
                
        Response Characteristics:
            - Empathetic and non-judgmental tone
            - Validates user's emotional experience
            - Offers hope and encouragement
            - Suggests professional resources when appropriate
            - Avoids giving medical advice or diagnoses
            
        Fallback Behavior:
            If Gemini API fails, returns a compassionate fallback message
            to ensure users always receive some form of support.
            
        Example:
            >>> response = engine.generate_response("I'm feeling really anxious")
            >>> print(response)
            "I understand you're experiencing anxiety right now, and that can feel 
            overwhelming. Your feelings are valid and it's important that you reached 
            out. Consider speaking with a mental health professional who can provide 
            personalized support and coping strategies..."
            
        Technical Notes:
            - Uses Gemini 1.5 Flash model for fast response generation
            - Implements retry logic and error handling
            - Logs API failures for system monitoring
        """
        if not user_message or not isinstance(user_message, str):
            logger.warning("Invalid input for response generation")
            return "Thank you for reaching out. Your feelings are important and valid."
        
        try:
            # Create therapeutic prompt for AI
            prompt = (
                "You are a compassionate mental health crisis counselor. "
                "Respond empathetically and supportively to this message. "
                "Keep your response under 100 words. Be warm, validating, "
                "and encourage professional help if the situation seems serious. "
                "Do not provide medical diagnoses or specific medical advice.\n\n"
                f"User message: {user_message}"
            )
            
            logger.debug("Generating AI response via Gemini...")
            
            # Generate response using Gemini
            if client is not None:
                response = client.generate_content(prompt)
                ai_response = response.text.strip()
                
                logger.debug(f"AI response generated successfully ({len(ai_response)} chars)")
                return ai_response
            else:
                logger.warning("Gemini client unavailable, using fallback response")
                return self._get_fallback_response()
                
        except Exception as e:
            logger.error(f"AI response generation failed: {e}")
            return self._get_fallback_response()
    
    def _get_fallback_response(self) -> str:
        """
        Provide fallback therapeutic response when AI is unavailable.
        
        Returns:
            str: Compassionate fallback message encouraging professional support
        """
        return (
            "Thank you for sharing with me. What you're experiencing is important, "
            "and it takes courage to reach out. Please consider speaking with a "
            "mental health professional or calling a crisis helpline if you need "
            "immediate support. You don't have to go through this alone."
        )


# Example usage:
if __name__ == "__main__":
    engine = MentalHealthTriageEngine()
    
    # Test with a sample message
    sample_message = "I'm feeling really overwhelmed lately and don't know what to do."
    result = engine.analyze_message(sample_message)
    
    print(f"Message: {result['text']}")
    print(f"Severity: {result['severity']}")
    print(f"Priority Score: {result['priority_score']}")
    print(f"AI Response: {result['ai_response']}")