"""
Bhashini (ULCA) API client.

Bhashini is used here for two things Sarvam cannot currently do for this
project:
  1. Hindi -> Santali (Ol Chiki) neural machine translation, as a second
     translation engine / fallback next to the local Excel dataset and
     Sarvam.
  2. Santali text-to-speech. Sarvam's Bulbul TTS does not have a Santali
     voice, so the app currently only plays back Hindi audio. Bhashini's
     TTS pipeline is used to generate the Santali voice output.

Auth
----
Bhashini issues two keys when you request API access on bhashini.gov.in:
  - a "User ID"        -> also called the "udyat" key in your notes
  - an "Ulca-Api-Key"  -> also called the "inference" key in your notes

Put them in .streamlit/secrets.toml (never commit this file):

    BHASHINI_USER_ID = "your-user-id"
    BHASHINI_API_KEY = "your-ulca-api-key"

Both calls below follow the standard two-step ULCA pattern:
  1. POST to the pipeline-config endpoint with your two keys to discover
     which model/callback URL + auth token to use for the task you want
     (translation or TTS).
  2. POST to the callback URL returned in step 1, using the short-lived
     inference token it hands back, to actually run the task.

NOTE: Bhashini's exact model IDs for Santali (sat) can change as new
models are published on the platform. If a call below returns a 4xx/5xx
error, log in to the Bhashini dashboard, open your pipeline, and confirm
the source/target language codes and task types are still supported for
"sat" — then adjust LANGUAGE codes / task configs here if needed.
"""

import base64
import requests
import streamlit as st

PIPELINE_CONFIG_URL = (
    "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline"
)


def _get_keys():
    user_id = st.secrets.get("BHASHINI_USER_ID")
    api_key = st.secrets.get("BHASHINI_API_KEY")
    if not user_id or not api_key:
        return None, None
    return user_id, api_key


def _get_pipeline(task_type, source_lang, target_lang=None):
    """Ask Bhashini which service/endpoint to use for a task."""

    user_id, api_key = _get_keys()
    if not user_id:
        raise RuntimeError("Bhashini keys not found in Streamlit secrets.")

    config = {"language": {"sourceLanguage": source_lang}}
    if target_lang:
        config["language"]["targetLanguage"] = target_lang

    payload = {
        "pipelineTasks": [{"taskType": task_type, "config": config}],
        "pipelineRequestConfig": {"pipelineId": "64392f96daac500b55c543cd"},
    }

    headers = {
        "userID": user_id,
        "ulcaApiKey": api_key,
        "Content-Type": "application/json",
    }

    resp = requests.post(PIPELINE_CONFIG_URL, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def translate_hi_to_santali(text: str) -> str:
    """Translate Hindi text to Santali using Bhashini NMT."""

    pipeline = _get_pipeline("translation", "hi", "sat")

    endpoint = pipeline["pipelineInferenceAPIEndPoint"]
    compute_url = endpoint["callbackUrl"]
    inference_key = endpoint["inferenceApiKey"]["value"]
    service_id = pipeline["pipelineResponseConfig"][0]["config"][0]["serviceId"]

    body = {
        "pipelineTasks": [
            {
                "taskType": "translation",
                "config": {
                    "language": {"sourceLanguage": "hi", "targetLanguage": "sat"},
                    "serviceId": service_id,
                },
            }
        ],
        "inputData": {"input": [{"source": text}]},
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": inference_key,
    }

    resp = requests.post(compute_url, json=body, headers=headers, timeout=30)
    resp.raise_for_status()
    result = resp.json()

    return result["pipelineResponse"][0]["output"][0]["target"].strip()


def santali_text_to_speech(text: str) -> bytes:
    """Generate Santali speech audio (wav bytes) from Santali text."""

    pipeline = _get_pipeline("tts", "sat")

    endpoint = pipeline["pipelineInferenceAPIEndPoint"]
    compute_url = endpoint["callbackUrl"]
    inference_key = endpoint["inferenceApiKey"]["value"]
    service_id = pipeline["pipelineResponseConfig"][0]["config"][0]["serviceId"]

    body = {
        "pipelineTasks": [
            {
                "taskType": "tts",
                "config": {
                    "language": {"sourceLanguage": "sat"},
                    "serviceId": service_id,
                    "gender": "female",
                },
            }
        ],
        "inputData": {"input": [{"source": text}]},
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": inference_key,
    }

    resp = requests.post(compute_url, json=body, headers=headers, timeout=30)
    resp.raise_for_status()
    result = resp.json()

    audio_b64 = result["pipelineResponse"][0]["audio"][0]["audioContent"]
    return base64.b64decode(audio_b64)
