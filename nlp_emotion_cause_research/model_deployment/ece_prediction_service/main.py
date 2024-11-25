import logging
import os
from datetime import datetime

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from tensorflow import argmax
from tensorflow.keras.models import load_model
from tensorflow.python.client import device_lib
from transformers import BertTokenizer, TFBertModel

from service_types.types import *

# ==================== Setup config ====================

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",  # Define the log message format
    handlers=[logging.StreamHandler()],  # Display logs in the console
)

# Environment should strictly use tensorflow cpu
logging.debug("Querying available devices:")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
logging.debug(device_lib.list_local_devices())

# Configure local CORS
app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global definitions
logging.info("=== Beginning initialization ===")
MODEL_DIR = "./model"
MAX_SEQUENCE_LENGTH = 124
beam_size = (
    4  # Beam size for predicting end index given start index for emotion cause spans
)

# ==================== Load prediction assets ====================

# Load BERT assets (reused between requests)
logging.info("Loading BERT assets")

logging.debug("Loading BERT model")
spanbert_model = TFBertModel.from_pretrained(f"{MODEL_DIR}/BERT/spanbert_model.h5")
logging.debug("Finished loading BERT model")

logging.debug("Loading BERT tokenizer")
spanbert_tokenizer = BertTokenizer.from_pretrained(
    f"{MODEL_DIR}/BERT/spanbert_tokenizer.h5"
)
logging.debug("Finished loading BERT model")

# Maps EmoCause class labels to its emotion string
emocause_emo_map = {
    29: "surprised",
    18: "guilty",
    30: "terrified",
    26: "proud",
    17: "grateful",
    0: "afraid",
    16: "furious",
    12: "disgusted",
    22: "joyful",
    25: "prepared",
    6: "ashamed",
    4: "anxious",
    31: "trusting",
    13: "embarrassed",
    2: "annoyed",
    14: "excited",
    24: "nostalgic",
    11: "disappointed",
    8: "confident",
    15: "faithful",
    27: "sad",
    7: "caring",
    10: "devastated",
    5: "apprehensive",
    28: "sentimental",
    21: "jealous",
    9: "content",
    23: "lonely",
    20: "impressed",
    1: "angry",
    3: "anticipating",
    19: "hopeful",
}

# Maps GNE class labels to its emotion string
gne_emo_map = {
    2: "disgust",
    5: "joy",
    8: "negative_surprise",
    7: "negative_anticipation_including_pessimism",
    3: "fear",
    1: "annoyance",
    0: "anger",
    9: "positive_anticipation_including_optimism",
    14: "trust",
    12: "sadness",
    10: "positive_surprise",
    13: "shame",
    11: "pride",
    6: "love_including_like",
    4: "guilt",
}

# Load trained models for inferece
logging.info("Loading ECE model assets")
bert_model_flag = {"TFBertModel": spanbert_model}

logging.debug("Loading EmoCause ECE model")
emocause_model = load_model(
    f"{MODEL_DIR}/EmoCause/emocause_insert_delete.keras", custom_objects=bert_model_flag
)
logging.debug("Finished loading EmoCause ECE model")

logging.debug("Loading GNE ECE model")
gne_model = load_model(
    f"{MODEL_DIR}/GNE/gne_sr_only.keras", custom_objects=bert_model_flag
)
logging.debug("Finished loading GNE ECE model")

logging.info("=== Service is ready to handle requests ===")

# ==================== Util functions ====================


def get_query_encodings(queries):
    logging.debug(f"Generating query encodings")
    return spanbert_tokenizer(
        queries,
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQUENCE_LENGTH,
        return_tensors="tf",
    )


def parse_emocause_emotion_strings(ec_pred_classes):
    logging.debug(f"Parsing EmoCause emotion strings")
    return [emocause_emo_map[pred_label] for pred_label in ec_pred_classes.numpy()]


def parse_gne_emotion_strings(ec_pred_classes):
    logging.debug(f"Parsing GNE emotion strings")
    return [gne_emo_map[pred_label] for pred_label in ec_pred_classes.numpy()]


def parse_predicted_spans(queries, si_preds, ei_preds):
    logging.debug(f"Parsing predicted spans")
    num_queries = len(queries)
    query_pred_spans = [None] * num_queries

    for i in range(num_queries):
        # We offset index by one since model was trained with 1-based indexing labels (instead of 0-based)
        curr_pred_si = si_preds[i] - 1
        curr_pred_ei = ei_preds[i] - 1
        curr_tokenized_query = spanbert_tokenizer.tokenize(queries[i])
        curr_pred_span_tokens = curr_tokenized_query[curr_pred_si : curr_pred_ei + 1]

        # Parse list of tokens back into a string
        curr_pred_span = spanbert_tokenizer.convert_tokens_to_string(
            curr_pred_span_tokens
        )
        query_pred_spans[i] = curr_pred_span

    return query_pred_spans


def get_predicted_emotions(ece_model, query_encodings):
    logging.debug(f"Generating emotion predictions")
    num_queries = len(query_encodings.input_ids)

    # Run emotion prediction
    ec_preds = ece_model.predict(
        [
            query_encodings.input_ids,
            query_encodings.token_type_ids,
            query_encodings.attention_mask,
            np.zeros(num_queries),
        ]
    )
    ec_pred_classes = argmax(ec_preds[0], axis=-1)

    return ec_pred_classes, ec_preds


def get_predicted_indices(ece_model, query_encodings, ec_preds):
    logging.debug(f"Generating span index predictions")
    num_queries = len(query_encodings.input_ids)

    for i in range(beam_size):
        # Grab the top (i + 1)-th start indices and probabilities for evaluation examples
        top_i_s_prob = -np.sort(-ec_preds[1], axis=-1)[:, i]
        top_i_si = np.argsort(-ec_preds[1], axis=-1)[:, i]

        # Pass those indices into predict, and grab the second output (which is end indices)
        ec_preds_ei = ece_model.predict(
            [
                query_encodings.input_ids,
                query_encodings.token_type_ids,
                query_encodings.attention_mask,
                top_i_si,
            ]
        )[2]

        # Grab predicted end index and its probability (just the max and argmax at index 0!)
        top_i_e_prob = -np.sort(-ec_preds_ei, axis=-1)[:, 0]
        top_i_ei = np.argsort(-ec_preds_ei, axis=-1)[:, 0]

        # Multiply the end index probability by the respective probability of the start index to get joint probability
        top_i_joint = np.multiply(top_i_s_prob, top_i_e_prob)

        # Log the joint and start-end pair
        top_i_ind_pair = np.stack((top_i_si, top_i_ei), axis=-1)

        if i == 0:
            beam_i_joint = top_i_joint
            beam_i_ind_pair = top_i_ind_pair
        elif i == 1:
            beam_i_joint = np.stack((beam_i_joint, top_i_joint), axis=1)
            beam_i_ind_pair = np.stack((beam_i_ind_pair, top_i_ind_pair), axis=1)
        else:
            beam_i_joint = np.hstack(
                (beam_i_joint, top_i_joint.reshape(len(top_i_joint), 1))
            )
            beam_i_ind_pair = np.hstack(
                (beam_i_ind_pair, top_i_ind_pair.reshape(top_i_ind_pair.shape[0], 1, 2))
            )

    # Highest joint probability results in final choice of start and end index for evaluation
    top_joints = np.argsort(-beam_i_joint, axis=-1)[:, 0]

    si_preds = [None] * num_queries
    ei_preds = [None] * num_queries
    for i in range(len(top_joints)):
        si_preds[i] = beam_i_ind_pair[i][top_joints[i]][0]
        ei_preds[i] = beam_i_ind_pair[i][top_joints[i]][1]

    return si_preds, ei_preds


# === Emocause inference handler ===
def get_emocause_predictions(queries):
    # Run inference
    query_encodings = get_query_encodings(queries)
    predicted_emotion_labels, ec_preds = get_predicted_emotions(
        emocause_model, query_encodings
    )
    si_preds, ei_preds = get_predicted_indices(
        emocause_model, query_encodings, ec_preds
    )

    # Parse predictions
    predicted_emotions = parse_emocause_emotion_strings(predicted_emotion_labels)
    predicted_spans = parse_predicted_spans(queries, si_preds, ei_preds)

    return [
        ModelOutput(predicted_emotion=predicted_emotion, predicted_cause=predicted_span)
        for predicted_emotion, predicted_span in zip(
            predicted_emotions, predicted_spans
        )
    ]


# === GNE inference handler ===
def get_gne_predictions(queries):
    # Run inference
    query_encodings = get_query_encodings(queries)
    predicted_emotion_labels, ec_preds = get_predicted_emotions(
        gne_model, query_encodings
    )
    si_preds, ei_preds = get_predicted_indices(gne_model, query_encodings, ec_preds)

    # Parse predictions
    predicted_emotions = parse_gne_emotion_strings(predicted_emotion_labels)
    predicted_spans = parse_predicted_spans(queries, si_preds, ei_preds)

    return [
        ModelOutput(predicted_emotion=predicted_emotion, predicted_cause=predicted_span)
        for predicted_emotion, predicted_span in zip(
            predicted_emotions, predicted_spans
        )
    ]


# ==================== Service route definitions ====================


@app.get("/api/health")
async def health():
    logging.info("Service is healthy")
    return {"message": str(datetime.now())}


@app.post("/api/predict/emocause", response_model=PredictOutput)
async def predict_emocause(input: PredictInput):
    logging.info(f"Generating EmoCause predictions for queries: {input.queries}")
    predictions = get_emocause_predictions(input.queries)

    return PredictOutput(predictions=predictions)


@app.post("/api/predict/gne", response_model=PredictOutput)
async def predict_gne(input: PredictInput):
    logging.info(f"Generating GNE predictions for queries: {input.queries}")
    predictions = get_gne_predictions(input.queries)

    return PredictOutput(predictions=predictions)
