"""
demo.py

A simple mental health triage demo using emotion classification via a Hugging Face transformer.
It identifies emotional severity and flags suicide risk to help prioritise urgent support cases.

Author: [Your Name]
Generated partially with assistance from LLMs
"""

from transformers import pipeline

# Suicide-related phrases to flag high-risk messages
SUICIDE_KEYWORDS = [
    "kill myself", "want to die", "end it all",
    "don't want to live", "suicide", "i'm done", "ending my life"
]

def contains_suicide_phrase(text: str) -> bool:
    """
    # This function is very basic. It simple scans to see if there's any direct matches with the key words.
    # Slight typos ie, kll myslf something like will not get detected so this approach is limited.
    """
    text = text.lower()
    for keyword in SUICIDE_KEYWORDS:
        if keyword in text:
            return True
    return False


def classify_severity(text: str, emotions: list) -> str:
    """
    Classify the severity of emotional distress based on emotion scores
    and whether suicide phrases are detected.
    """
    print(emotions)
    emotions = {e['label']: e['score'] for e in emotions}
    sadness = emotions.get("sadness", 0)
    fear = emotions.get("fear", 0)

    if contains_suicide_phrase(text):
        return "🔴 Immediate: Self Harm Mentioned"
    if sadness > 0.7 or fear > 0.6:
        return "🔴 Very Severe"
    elif sadness > 0.5 or fear > 0.4:
        return "🟠 Severe"
    elif sadness > 0.3 or fear > 0.2:
        return "🟡 Moderate"
    else:
        return "🟢 Mild"

def analyse_emotions_and_severity(texts: list):
    """
    Perform emotion classification and severity scoring on each text input.
    """
    print("\n------ Sentinment Analysis Demo -------")
    emotion_classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True
    )

    for i, text in enumerate(texts, 1):
        emotion_scores = emotion_classifier(text)[0]
        severity = classify_severity(text, emotion_scores)

        # Get top 2 emotions
        top_emotions = sorted(emotion_scores, key=lambda x: x['score'], reverse=True)[:2]
        emo_str = ", ".join([f"{e['label']} ({e['score']:.2f})" for e in top_emotions])

        print(f"{i}. {text}")
        print(f"   → Severity: {severity}")
        print(f"   → Top Emotions: {emo_str}\n")

if __name__ == "__main__":
    SAMPLE_TEXTS = [
        "I'm feeling okay today, just tired.",
        "Everything is falling apart. I can’t cope anymore.",
        "I think I'm getting better slowly, thanks for checking in.",
        "Nobody listens. I feel invisible.",
        "I want to disappear. What's the point of anything?",
        "I’m doing great! Therapy has really helped.",
        "I’m not sure what I need, but I’m overwhelmed.",
        "This week was tough, but I made it through.",
        "I can't stop crying. Everything feels heavy.",
        "I'm just numb. Not sad, not happy, just... nothing.",
        "I want to die. Please help me.",
        "I'm so tired of life. I’m done.",
        "Therapy is helping me slowly regain control.",
        "Just a rough day. I’ll be fine."
    ]

    analyse_emotions_and_severity(SAMPLE_TEXTS)
