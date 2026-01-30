"""OpenAI-compatible image generation endpoint using ChatGPT conversation API."""
from __future__ import annotations

import base64
import json
import random
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import requests
from flask import Blueprint, Response, current_app, jsonify, make_response, request

from .http import build_cors_headers
from .proof_of_work import get_answer_token, get_config, get_requirements_token
from .utils import get_effective_chatgpt_auth

images_bp = Blueprint("images", __name__)

CHATGPT_BASE_URL = "https://chatgpt.com"
CHATGPT_CONVERSATION_URL = f"{CHATGPT_BASE_URL}/backend-api/f/conversation"
CHATGPT_REQUIREMENTS_URL = f"{CHATGPT_BASE_URL}/backend-api/sentinel/chat-requirements"
CHATGPT_FILES_DOWNLOAD_URL = f"{CHATGPT_BASE_URL}/backend-api/files"

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36"
OAI_DEVICE_ID = str(uuid.uuid4())  # Generate a persistent device ID
OAI_CLIENT_BUILD = "4323075"
OAI_CLIENT_VERSION = "prod-f700ea221d7179d2ee3ed2b022f6972f77a5a7a1"


def _log_json(prefix: str, payload: Any) -> None:
    try:
        print(f"{prefix}\n{json.dumps(payload, indent=2, ensure_ascii=False)}")
    except Exception:
        try:
            print(f"{prefix}\n{payload}")
        except Exception:
            pass


def _get_base_headers(access_token: str, account_id: str) -> Dict[str, str]:
    """Build base headers for ChatGPT API requests."""
    headers = {
        "accept": "*/*",
        "accept-encoding": "gzip, deflate, br, zstd",
        "accept-language": "en-US,en;q=0.9",
        "content-type": "application/json",
        "dnt": "1",
        "oai-client-build-number": OAI_CLIENT_BUILD,
        "oai-client-version": OAI_CLIENT_VERSION,
        "oai-device-id": OAI_DEVICE_ID,
        "oai-language": "en-US",
        "origin": CHATGPT_BASE_URL,
        "priority": "u=1, i",
        "referer": f"{CHATGPT_BASE_URL}/",
        "sec-ch-ua": '"Not(A:Brand";v="8", "Chromium";v="144", "Google Chrome";v="144"',
        "sec-ch-ua-arch": '"arm"',
        "sec-ch-ua-bitness": '"64"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-model": '""',
        "sec-ch-ua-platform": '"macOS"',
        "sec-ch-ua-platform-version": '"15.0.0"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "user-agent": USER_AGENT,
        "authorization": f"Bearer {access_token}",
        "chatgpt-account-id": account_id,
    }
    return headers


def _get_chat_requirements(
    access_token: str,
    account_id: str,
    verbose: bool = False,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Get chat requirements token and proof-of-work token."""
    headers = _get_base_headers(access_token, account_id)

    config = get_config(USER_AGENT)
    p_token = get_requirements_token(config)

    try:
        resp = requests.post(
            CHATGPT_REQUIREMENTS_URL,
            headers=headers,
            json={"p": p_token},
            timeout=30,
        )

        if verbose:
            _log_json("Chat requirements response", resp.text[:500] if resp.text else "")

        if resp.status_code != 200:
            return None, None, f"Failed to get chat requirements: {resp.status_code} - {resp.text[:200]}"

        try:
            data = resp.json()
        except Exception as e:
            return None, None, f"Failed to parse chat requirements response: {e}"

        if not isinstance(data, dict):
            return None, None, f"Unexpected chat requirements response type: {type(data)}"

        chat_token = data.get("token")

        # Handle proof-of-work if required
        proof_token = None
        proofofwork = data.get("proofofwork", {})
        if isinstance(proofofwork, dict) and proofofwork.get("required"):
            seed = proofofwork.get("seed")
            diff = proofofwork.get("difficulty")
            if seed and diff:
                # Convert difficulty to string if it's not already
                diff_str = str(diff) if not isinstance(diff, str) else diff
                proof_token, solved = get_answer_token(seed, diff_str, config)
                if verbose:
                    print(f"  Proof-of-work: diff={diff_str}, solved={solved}")

        return chat_token, proof_token, None

    except requests.RequestException as e:
        return None, None, f"Failed to get chat requirements: {e}"


def _poll_async_status(
    conversation_id: str,
    access_token: str,
    account_id: str,
    max_attempts: int = 30,
    delay: float = 2.0,
    verbose: bool = False,
    wait_for_stable: bool = True,
) -> List[str]:
    """Poll the conversation endpoint for image file IDs.

    If wait_for_stable=True, keeps polling until no new file IDs appear
    for 2 consecutive attempts (to ensure we get the final image).
    """
    headers = _get_base_headers(access_token, account_id)

    all_file_ids: List[str] = []
    last_file_id: Optional[str] = None
    stable_count = 0
    required_stable = 15 if wait_for_stable else 1

    for attempt in range(max_attempts):
        # Try fetching conversation details
        url = f"{CHATGPT_BASE_URL}/backend-api/conversation/{conversation_id}"
        try:
            resp = requests.get(
                url,
                headers=headers,
                timeout=30,
            )

            if resp.status_code == 200:
                data = resp.json()
                if verbose:
                    print(f"  Conversation fetch (attempt {attempt + 1}): found {len(str(data))} chars...")

                # Look for file IDs in the response
                content_str = json.dumps(data)

                # Pattern for file IDs (file_xxxxx format)
                file_pattern = r'file_[0-9a-f]{20,40}'
                found_files = re.findall(file_pattern, content_str)

                new_files_found = False
                for f in found_files:
                    if f not in all_file_ids:
                        all_file_ids.append(f)
                        last_file_id = f
                        new_files_found = True
                        if verbose:
                            print(f"  Found new file ID: {f}")

                # Also look for asset_pointer patterns
                asset_pattern = r'(?:file-service|sediment)://([^"\']+)'
                found_assets = re.findall(asset_pattern, content_str)
                for a in found_assets:
                    if a not in all_file_ids:
                        all_file_ids.append(a)
                        last_file_id = a
                        new_files_found = True
                        if verbose:
                            print(f"  Found new asset: {a}")

                # Track stability
                if new_files_found:
                    stable_count = 0
                elif all_file_ids:
                    stable_count += 1
                    if verbose:
                        print(f"  No new files, stable count: {stable_count}/{required_stable}")

                # Return when stable (no new files for required_stable attempts)
                if all_file_ids and stable_count >= required_stable:
                    if verbose:
                        print(f"  File IDs stable, returning last: {last_file_id}")
                    # Return only the last file ID (most recent = final image)
                    return [last_file_id] if last_file_id else all_file_ids[-1:]

            elif verbose:
                print(f"  Conversation fetch error ({resp.status_code}): {resp.text[:100]}...")

        except Exception as e:
            if verbose:
                print(f"  Conversation fetch error: {e}")

        time.sleep(delay)

    # Return the last file ID found (most likely to be final)
    if verbose:
        print(f"  Max attempts reached, returning last file ID: {last_file_id}")
    return [last_file_id] if last_file_id else all_file_ids[-1:] if all_file_ids else []


def _delete_conversation(
    conversation_id: str,
    access_token: str,
    account_id: str,
    verbose: bool = False,
) -> bool:
    """Delete a conversation to clean up after image generation."""
    if not conversation_id:
        return False

    headers = _get_base_headers(access_token, account_id)
    url = f"{CHATGPT_BASE_URL}/backend-api/conversation/{conversation_id}"

    try:
        # Use PATCH with is_visible: false to "delete" (hide) the conversation
        resp = requests.patch(
            url,
            headers=headers,
            json={"is_visible": False},
            timeout=30,
        )
        if verbose:
            print(f"  Delete conversation {conversation_id}: {resp.status_code}")
        return resp.status_code == 200
    except Exception as e:
        if verbose:
            print(f"  Delete conversation error: {e}")
        return False


def _generate_image_via_conversation(
    prompt: str,
    access_token: str,
    account_id: str,
    size: str = "1024x1024",
    quality: str = "auto",
    style: Optional[str] = None,
    n: int = 1,
    verbose: bool = False,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Use ChatGPT conversation API to generate images.
    Returns a list of image info dicts and any error message.
    """
    # Get chat requirements first
    chat_token, proof_token, error = _get_chat_requirements(access_token, account_id, verbose)
    if error:
        return [], error

    # Build the message
    enhanced_prompt = prompt
    if style and style.lower() != "vivid":
        enhanced_prompt = f"[Style: {style}] {prompt}"
    if quality and quality.lower() == "hd":
        enhanced_prompt = f"[High quality, detailed] {enhanced_prompt}"

    parent_message_id = str(uuid.uuid4())
    message_id = str(uuid.uuid4())

    # Build conversation request with image generation hints
    chat_request = {
        "action": "next",
        "messages": [
            {
                "id": message_id,
                "author": {"role": "user"},
                "create_time": time.time(),
                "content": {
                    "content_type": "text",
                    "parts": [f"Generate an image: {enhanced_prompt}"]
                },
                "metadata": {
                    "developer_mode_connector_ids": [],
                    "selected_github_repos": [],
                    "selected_all_github_repos": False,
                    "system_hints": ["picture_v2"],
                    "serialization_metadata": {"custom_symbol_offsets": []}
                }
            }
        ],
        "parent_message_id": parent_message_id,
        "model": "gpt-4o",
        "timezone_offset_min": 480,
        "timezone": "America/Los_Angeles",
        "conversation_mode": {"kind": "primary_assistant"},
        "enable_message_followups": True,
        "system_hints": ["picture_v2"],  # Enable image generation
        "supports_buffering": True,
        "supported_encodings": ["v1"],
        "client_contextual_info": {
            "is_dark_mode": True,
            "time_since_loaded": random.randint(100, 2000),
            "page_height": random.randint(500, 1000),
            "page_width": random.randint(1000, 2500),
            "pixel_ratio": 2,
            "screen_height": random.randint(800, 1500),
            "screen_width": random.randint(1000, 2500),
            "app_name": "chatgpt.com"
        },
        "paragen_cot_summary_display_override": "allow",
        "force_parallel_switch": "auto",
        # Note: history_and_training_disabled breaks polling, so we delete the conversation after instead
    }

    headers = _get_base_headers(access_token, account_id)
    headers.update({
        "accept": "text/event-stream",
        "openai-sentinel-chat-requirements-token": chat_token or "",
    })
    if proof_token:
        headers["openai-sentinel-proof-token"] = proof_token

    if verbose:
        _log_json("OUTBOUND >> ChatGPT Conversation API (image gen)", chat_request)

    try:
        resp = requests.post(
            CHATGPT_CONVERSATION_URL,
            headers=headers,
            json=chat_request,
            stream=True,
            timeout=300,
        )
    except requests.RequestException as e:
        return [], f"Failed to connect to ChatGPT: {e}"

    if resp.status_code >= 400:
        try:
            err_text = resp.text[:500]
        except Exception:
            err_text = "Unknown error"
        return [], f"ChatGPT API error ({resp.status_code}): {err_text}"

    # Parse the SSE stream for image data
    file_ids: List[str] = []
    image_urls: List[str] = []
    conversation_id: Optional[str] = None

    try:
        for raw in resp.iter_lines(decode_unicode=False):
            if not raw:
                continue
            line = raw.decode("utf-8", errors="ignore") if isinstance(raw, (bytes, bytearray)) else raw

            if verbose:
                print(f"  SSE: {line[:300]}...")

            if not line.startswith("data: "):
                continue
            data = line[6:].strip()
            if not data or data == "[DONE]":
                continue

            try:
                evt = json.loads(data)
            except Exception:
                continue

            # Skip non-dict events (like delta_encoding "v1")
            if not isinstance(evt, dict):
                continue

            # Extract conversation_id from the event or nested data
            if not conversation_id:
                conversation_id = evt.get("conversation_id")
                # Also check in nested data for delta events
                if not conversation_id and "v" in evt and isinstance(evt.get("v"), dict):
                    conversation_id = evt["v"].get("conversation_id")

            # Handle delta-encoded events (v1 format)
            if "v" in evt and isinstance(evt.get("v"), dict):
                evt = evt["v"]  # Use the nested value

            message = evt.get("message", {})
            if not isinstance(message, dict):
                message = {}
            content = message.get("content", {})
            if not isinstance(content, dict):
                content = {}
            content_type = content.get("content_type", "")

            # Check message status
            message_status = message.get("status", "")

            # ONLY collect file IDs from finished_successfully messages with image_asset_pointer
            if message_status == "finished_successfully" and content_type == "multimodal_text":
                parts = content.get("parts", [])
                for part in parts:
                    if isinstance(part, dict):
                        inner_content_type = part.get("content_type", "")
                        if inner_content_type == "image_asset_pointer":
                            asset_pointer = part.get("asset_pointer", "")
                            if asset_pointer.startswith("file-service://"):
                                fid = asset_pointer.replace("file-service://", "")
                                if fid and fid not in file_ids:
                                    if verbose:
                                        print(f"    Final image_asset_pointer (file-service): {fid}")
                                    file_ids.append(fid)
                            elif asset_pointer.startswith("sediment://"):
                                fid = asset_pointer.replace("sediment://", "")
                                if fid and fid not in file_ids:
                                    if verbose:
                                        print(f"    Final image_asset_pointer (sediment): {fid}")
                                    file_ids.append(fid)

            # Serialize event for regex searches
            content_str = json.dumps(evt)

            # Only use regex fallback for finished_successfully messages
            if message_status == "finished_successfully":
                # Pattern for file IDs: file-xxxxxxxx or file_xxxxxxxx
                file_pattern = r'file[-_][0-9a-zA-Z]{20,40}'
                found_files = re.findall(file_pattern, content_str)
                for f in found_files:
                    if f not in file_ids:
                        if verbose:
                            print(f"    Regex fallback found: {f}")
                        file_ids.append(f)

            # Look for direct image URLs (always search)
            url_pattern = r'https?://[^\s"\'<>]+\.(?:png|jpg|jpeg|webp|gif)[^\s"\'<>]*'
            found_urls = re.findall(url_pattern, content_str, re.IGNORECASE)
            for url in found_urls:
                # Clean up URL
                url = url.rstrip('\\').rstrip('"').rstrip("'")
                if url not in image_urls and "oaiusercontent" in url:
                    image_urls.append(url)

    finally:
        resp.close()

    if verbose:
        print(f"  Stream found file_ids: {file_ids}, image_urls: {image_urls}, conversation_id: {conversation_id}")

    # ALWAYS poll for stable file IDs (to get the final image, not intermediate)
    # The stream often returns intermediate file IDs before the final one is ready
    if conversation_id:
        if verbose:
            print(f"  Polling for FINAL image (waiting for stable file IDs)...")
        polled_files = _poll_async_status(
            conversation_id, access_token, account_id,
            max_attempts=60, delay=2.0, verbose=verbose,
            wait_for_stable=True
        )
        # Use polled files instead of stream files (polled = final)
        if polled_files:
            file_ids = polled_files
            if verbose:
                print(f"  Using polled file IDs (final): {file_ids}")

    if verbose:
        print(f"  Final file_ids: {file_ids}, image_urls: {image_urls}")

    # Build results
    results: List[Dict[str, Any]] = []

    # Prefer direct URLs
    for url in image_urls[:n]:
        results.append({"type": "url", "value": url, "conversation_id": conversation_id})

    # Fall back to file IDs
    if len(results) < n:
        for fid in file_ids[:n - len(results)]:
            results.append({"type": "file_id", "value": fid, "conversation_id": conversation_id})

    return results, None


def _get_download_url_and_bytes(
    file_id: str,
    conversation_id: Optional[str],
    access_token: str,
    account_id: str,
    verbose: bool = False,
) -> Tuple[Optional[str], Optional[bytes], Optional[str]]:
    """Get the actual download URL and image bytes for a file ID."""
    headers = _get_base_headers(access_token, account_id)

    # Try /files/{file_id}/download first
    url = f"{CHATGPT_FILES_DOWNLOAD_URL}/{file_id}/download"
    if conversation_id:
        url += f"?conversation_id={conversation_id}&inline=false"

    if verbose:
        print(f"  Getting download URL: {url}")

    download_url = None
    file_name = None
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            download_url = data.get("download_url")
            file_name = data.get("file_name", "")
            if verbose:
                print(f"  Got download_url: {download_url[:100] if download_url else None}...")
                print(f"  File name: {file_name}")
            # Check if this is a partial/intermediate image
            if file_name and ".part" in file_name:
                if verbose:
                    print(f"  SKIPPING: This is a partial/intermediate image ({file_name})")
                return None, None, "Partial image - still generating"
    except Exception as e:
        if verbose:
            print(f"  /files download error: {e}")

    # If we didn't get a download URL, try attachment endpoint
    if not download_url and conversation_id:
        url = f"{CHATGPT_BASE_URL}/backend-api/conversation/{conversation_id}/attachment/{file_id}/download"
        if verbose:
            print(f"  Trying attachment URL: {url}")

        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                download_url = data.get("download_url")
        except Exception as e:
            if verbose:
                print(f"  Attachment download error: {e}")

    if not download_url:
        return None, None, "Failed to get download URL"

    # Now download the actual image using the authenticated session
    if verbose:
        print(f"  Downloading image from: {download_url[:100]}...")

    try:
        # The estuary URLs require authentication - use our headers
        img_resp = requests.get(download_url, headers=headers, timeout=60)
        if img_resp.status_code == 200:
            content_type = img_resp.headers.get("Content-Type", "")
            if "image" in content_type or len(img_resp.content) > 1000:
                return download_url, img_resp.content, None
            else:
                if verbose:
                    print(f"  Unexpected content type: {content_type}, content: {img_resp.text[:100]}")
                return download_url, None, f"Unexpected content type: {content_type}"
        else:
            return download_url, None, f"Image download failed: {img_resp.status_code}"
    except Exception as e:
        return download_url, None, f"Image download error: {e}"


def _download_from_url(url: str, verbose: bool = False) -> Tuple[Optional[bytes], Optional[str]]:
    """Download an image from a direct URL."""
    if verbose:
        print(f"  Downloading image from URL: {url}")

    try:
        resp = requests.get(url, timeout=60, allow_redirects=True)
        if resp.status_code == 200:
            return resp.content, None
        return None, f"Download failed with status {resp.status_code}"
    except requests.RequestException as e:
        return None, f"Download error: {e}"


@images_bp.route("/v1/images/generations", methods=["POST"])
def create_image() -> Response:
    """
    OpenAI-compatible image generation endpoint.

    Uses ChatGPT's conversation API with system_hints=["picture_v2"] to generate images.

    Accepts:
    - prompt: string (required)
    - model: string (optional, default "gpt-image-1")
    - n: int (optional, default 1, max 4)
    - size: string (optional, "1024x1024", "1024x1792", "1792x1024")
    - quality: string (optional, "auto", "hd")
    - style: string (optional, "vivid", "natural")
    - response_format: string (optional, "url" or "b64_json")

    Returns OpenAI-compatible response with generated images.
    """
    verbose = bool(current_app.config.get("VERBOSE"))

    raw = request.get_data(cache=True, as_text=True) or ""
    if verbose:
        try:
            print("IN POST /v1/images/generations\n" + raw)
        except Exception:
            pass

    try:
        payload = json.loads(raw) if raw else {}
    except Exception:
        err = {"error": {"message": "Invalid JSON body"}}
        return jsonify(err), 400

    # Extract parameters
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        err = {"error": {"message": "prompt is required and must be a non-empty string"}}
        return jsonify(err), 400

    n = min(max(int(payload.get("n", 1)), 1), 4)  # 1-4 images
    size = payload.get("size", "1024x1024")
    quality = payload.get("quality", "auto")
    style = payload.get("style")
    response_format = payload.get("response_format", "url")

    # Get authentication
    access_token, account_id = get_effective_chatgpt_auth()
    if not access_token or not account_id:
        err = {
            "error": {
                "message": "Missing ChatGPT credentials. Run 'python3 chatmock.py login' first."
            }
        }
        resp = make_response(jsonify(err), 401)
        for k, v in build_cors_headers().items():
            resp.headers.setdefault(k, v)
        return resp

    # Generate images
    image_infos, error = _generate_image_via_conversation(
        prompt=prompt,
        access_token=access_token,
        account_id=account_id,
        size=size,
        quality=quality,
        style=style,
        n=n,
        verbose=verbose,
    )

    if error:
        err = {"error": {"message": error}}
        resp = make_response(jsonify(err), 502)
        for k, v in build_cors_headers().items():
            resp.headers.setdefault(k, v)
        return resp

    if not image_infos:
        err = {"error": {"message": "No images were generated. The prompt may have been rejected or image generation is not available."}}
        resp = make_response(jsonify(err), 500)
        for k, v in build_cors_headers().items():
            resp.headers.setdefault(k, v)
        return resp

    # Build response data - with retry logic for partial images
    created = int(time.time())
    data: List[Dict[str, Any]] = []
    tried_file_ids: set = set()
    conversation_id = image_infos[0].get("conversation_id") if image_infos else None
    max_poll_retries = 30
    poll_delay = 2.0

    for poll_attempt in range(max_poll_retries):
        for info in image_infos:
            info_type = info.get("type")
            info_value = info.get("value")
            conv_id = info.get("conversation_id")

            # Skip already tried file IDs
            if info_value in tried_file_ids:
                continue
            tried_file_ids.add(info_value)

            if info_type == "url":
                # Direct URL - try to download
                img_bytes, dl_error = _download_from_url(info_value, verbose=verbose)
            else:
                # File ID - get download URL and bytes
                dl_url, img_bytes, dl_error = _get_download_url_and_bytes(
                    info_value, conv_id, access_token, account_id, verbose=verbose
                )

            if response_format == "b64_json":
                # Return as base64
                if img_bytes:
                    b64_data = base64.b64encode(img_bytes).decode("utf-8")
                    data.append({"b64_json": b64_data})
                elif verbose:
                    print(f"  Failed to download {info_value}: {dl_error}")
            else:
                # Return URL - but since ChatGPT URLs require auth, we should use b64_json
                # or return a proxy URL. For now, if we have bytes, return b64_json as fallback
                if img_bytes:
                    # We have the image, return as b64_json even if url was requested
                    # because ChatGPT URLs require authentication
                    b64_data = base64.b64encode(img_bytes).decode("utf-8")
                    data.append({"b64_json": b64_data})
                elif info_type == "url":
                    data.append({"url": info_value})
                else:
                    if verbose:
                        print(f"  Could not get image for {info_value}: {dl_error}")

        # If we have enough images, we're done
        if len(data) >= n:
            break

        # If no images yet, poll for more file IDs
        if not data and conversation_id:
            if verbose:
                print(f"  No final images yet, polling for more (attempt {poll_attempt + 1}/{max_poll_retries})...")
            time.sleep(poll_delay)
            # Poll for new file IDs
            new_file_ids = _poll_async_status(
                conversation_id, access_token, account_id,
                max_attempts=1, delay=0, verbose=verbose
            )
            for fid in new_file_ids:
                if fid not in tried_file_ids:
                    image_infos.append({"type": "file_id", "value": fid, "conversation_id": conversation_id})
        else:
            break

    response = {
        "created": created,
        "data": data,
    }

    # Clean up: delete the conversation to avoid cluttering history
    if conversation_id and data:
        if verbose:
            print(f"  Cleaning up conversation {conversation_id}...")
        _delete_conversation(conversation_id, access_token, account_id, verbose=verbose)

    if verbose:
        _log_json("OUT POST /v1/images/generations", response)

    resp = make_response(jsonify(response), 200)
    for k, v in build_cors_headers().items():
        resp.headers.setdefault(k, v)
    return resp


@images_bp.route("/v1/images/generations", methods=["OPTIONS"])
def create_image_options() -> Response:
    """Handle CORS preflight for image generation."""
    resp = make_response("", 200)
    for k, v in build_cors_headers().items():
        resp.headers.setdefault(k, v)
    return resp
