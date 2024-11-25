from typing import List

from pydantic import BaseModel, Extra, validator

emocause_emotions = set(
    [
        "afraid",
        "angry",
        "annoyed",
        "anticipating",
        "anxious",
        "apprehensive",
        "ashamed",
        "caring",
        "confident",
        "content",
        "devastated",
        "disappointed",
        "disgusted",
        "embarrassed",
        "excited",
        "faithful",
        "furious",
        "grateful",
        "guilty",
        "hopeful",
        "impressed",
        "jealous",
        "joyful",
        "lonely",
        "nostalgic",
        "prepared",
        "proud",
        "sad",
        "sentimental",
        "surprised",
        "terrified",
        "trusting",
    ]
)

gne_emotions = set(
    [
        "anger",
        "annoyance",
        "disgust",
        "fear",
        "guilt",
        "joy",
        "love_including_like",
        "negative_anticipation_including_pessimism",
        "negative_surprise",
        "positive_anticipation_including_optimism",
        "positive_surprise",
        "pride",
        "sadness",
        "shame",
        "trust",
    ]
)


class ModelOutput(BaseModel, extra=Extra.forbid):
    predicted_emotion: str
    predicted_cause: str

    # Enumerate type enforcement on the emotion sets
    @validator("predicted_emotion")
    def latitude_in_range(emotion_str):
        if emotion_str not in emocause_emotions and emotion_str not in gne_emotions:
            raise ValueError(
                f"predicted_emotion {emotion_str} is not a valid GNE or EmoCause emotion"
            )
        return emotion_str


# Request structure
class PredictInput(BaseModel, extra=Extra.forbid):
    queries: List[str]


# Response structure
class PredictOutput(BaseModel, extra=Extra.forbid):
    predictions: List[ModelOutput]
