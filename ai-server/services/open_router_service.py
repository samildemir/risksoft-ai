import base64
import logging
import json
import mimetypes
import time
from pathlib import Path
from typing import List, Dict, Optional, Union, Any, Sequence

import requests
from utils.helper import get_env
from constants.config import (
    OPENROUTER_GPT_4O,
    OPENROUTER_GPT_4O_MINI,
    OPENROUTER_CLAUDE_3_5_SONNET,
    OPENROUTER_GEMINI_FLASH,
)
from utils.download import download_and_order_files
from models.schemas import DocumentSource, AgentResponse, ModelUsage, ChatbotUsageLog
from utils.s3Handler import S3Handler


class OpenRouterService:
    def __init__(self):
        """
        OpenRouter API servisi için istemci sınıfını başlatır
        """
        self.api_key = get_env("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

        # S3 handler for document operations
        self.s3 = S3Handler()
        self.BUCKET_NAME = self.s3.get_bucket_name()

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")

    @staticmethod
    def _mime_to_format(mime_type: Optional[str], fallback: str = "bin") -> str:
        """Map MIME type to simple format extension."""
        if not mime_type:
            return fallback
        return mime_type.split("/")[-1]

    def _prepare_file_payload(
        self,
        source: Union[str, Path, Dict[str, Any], bytes, bytearray],
        *,
        default_mime: str,
        default_name_prefix: str,
    ) -> Dict[str, str]:
        """
        Read file-like inputs and convert them into the payload shape expected by OpenRouter.
        Supports local paths, raw bytes, and dictionaries with explicit metadata.
        """
        try:
            if isinstance(source, (str, Path)):
                path = Path(source)
                data_bytes = path.read_bytes()
                name = path.name or f"{default_name_prefix}.bin"
                mime_type = mimetypes.guess_type(path.name)[0] or default_mime
            elif isinstance(source, (bytes, bytearray)):
                data_bytes = bytes(source)
                mime_type = default_mime
                name = f"{default_name_prefix}.bin"
            elif isinstance(source, dict):
                data = source.get("data")
                if data is None:
                    raise ValueError("File source dictionary must include 'data'")
                name = (
                    source.get("name")
                    or source.get("filename")
                    or f"{default_name_prefix}.bin"
                )
                mime_type = source.get("mime_type") or default_mime
                if isinstance(data, str):
                    if source.get("is_base64"):
                        return {
                            "name": name,
                            "mime_type": mime_type,
                            "data": data,
                        }
                    data_bytes = data.encode("utf-8")
                else:
                    data_bytes = bytes(data)
            else:
                raise TypeError("Unsupported file source type for multimodal payloads")

            encoded = base64.b64encode(data_bytes).decode("utf-8")
            return {
                "name": name,
                "mime_type": mime_type,
                "data": encoded,
            }
        except Exception as exc:
            logging.error(f"Failed to prepare file payload: {exc}")
            raise

    def _build_multimodal_content(
        self,
        *,
        text: Optional[str] = None,
        image_urls: Optional[Sequence[str]] = None,
        pdf_files: Optional[
            Sequence[Union[str, Path, Dict[str, Any], bytes, bytearray]]
        ] = None,
        audio_files: Optional[
            Sequence[Union[str, Path, Dict[str, Any], bytes, bytearray]]
        ] = None,
        video_files: Optional[
            Sequence[Union[str, Path, Dict[str, Any], bytes, bytearray]]
        ] = None,
        extra_content: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Build the `content` array for multimodal messages following
        https://openrouter.ai/docs/features/multimodal/overview and the image specific
        guidance outlined in https://openrouter.ai/docs/features/multimodal/images
        (text first, then `image_url`/file entries, each image in its own block).
        """
        contents: List[Dict[str, Any]] = []

        if text:
            contents.append({"type": "text", "text": text})

        if image_urls:
            for url in image_urls:
                if not url:
                    continue
                contents.append({"type": "image_url", "image_url": {"url": url}})

        if pdf_files:
            for entry in pdf_files:
                payload = self._prepare_file_payload(
                    entry,
                    default_mime="application/pdf",
                    default_name_prefix="document",
                )
                contents.append({"type": "file", "file": payload})

        if audio_files:
            for entry in audio_files:
                payload = self._prepare_file_payload(
                    entry,
                    default_mime="audio/mpeg",
                    default_name_prefix="audio",
                )
                contents.append(
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": payload["data"],
                            "format": self._mime_to_format(
                                payload["mime_type"], fallback="mp3"
                            ),
                        },
                    }
                )

        if video_files:
            for entry in video_files:
                payload = self._prepare_file_payload(
                    entry,
                    default_mime="video/mp4",
                    default_name_prefix="video",
                )
                contents.append(
                    {
                        "type": "input_video",
                        "input_video": {
                            "data": payload["data"],
                            "format": self._mime_to_format(
                                payload["mime_type"], fallback="mp4"
                            ),
                        },
                    }
                )

        if extra_content:
            contents.extend(extra_content)

        if not contents:
            raise ValueError(
                "At least one content block must be provided for multimodal requests"
            )

        return contents

    def _make_request(
        self,
        messages: List[Dict],
        model: str = OPENROUTER_GPT_4O,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """
        OpenRouter API'sine istek gönderir

        Args:
            messages: Chat mesajları listesi
            model: Kullanılacak model adı
            temperature: Yaratıcılık seviyesi (0.0-1.0)
            max_tokens: Maksimum token sayısı

        Returns:
            API yanıtı
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens:
            data["max_tokens"] = max_tokens

        if response_format:
            data["response_format"] = response_format

        if extra_params:
            data.update(extra_params)

        try:
            start_time = time.time()
            response = requests.post(
                self.base_url, headers=headers, data=json.dumps(data), timeout=120
            )
            end_time = time.time()
            response_time_ms = int((end_time - start_time) * 1000)

            response.raise_for_status()
            result = response.json()

            # Add response time to result
            result["response_time_ms"] = response_time_ms
            return result

        except requests.exceptions.RequestException as e:
            logging.error(f"OpenRouter API request failed: {str(e)}")
            raise Exception(f"OpenRouter API request failed: {str(e)}")

    def chat_completion(
        self,
        messages: List[Dict],
        model: str = OPENROUTER_GPT_4O,
        temperature: float = 0.7,
        usage_log: Optional[ChatbotUsageLog] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> AgentResponse:
        """
        Chat completion API'si kullanarak metin üretir

        Args:
            messages: Chat mesajları listesi [{"role": "user", "content": "message"}]
            model: Kullanılacak model
            temperature: Yaratıcılık seviyesi
            usage_log: Kullanım logları için

        Returns:
            AgentResponse objesi
        """
        response_payload: Optional[Dict[str, Any]] = None
        try:
            response_payload = self._make_request(
                messages,
                model,
                temperature,
                max_tokens,
                response_format,
                extra_params,
            )

            if not isinstance(response_payload, dict):
                raise ValueError("OpenRouter API response is empty or malformed")

            # Response metadata
            response_id = response_payload.get("id", "")
            provider = response_payload.get("provider", "")

            # Choice data
            choices = response_payload.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ValueError(
                    f"OpenRouter response missing 'choices': {response_payload}"
                )

            choice = choices[0] or {}
            if not isinstance(choice, dict):
                raise ValueError(
                    f"OpenRouter response choice is not an object: {choice}"
                )

            content = (choice.get("message") or {}).get("content", "")
            finish_reason = choice.get("finish_reason", "")

            # Usage data
            usage = response_payload.get("usage") or {}
            if not isinstance(usage, dict):
                usage = {}
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            response_time_ms = response_payload.get("response_time_ms", 0)

            # Additional usage details
            prompt_tokens_details = usage.get("prompt_tokens_details") or {}
            completion_tokens_details = usage.get("completion_tokens_details") or {}
            cached_tokens = prompt_tokens_details.get("cached_tokens", 0)
            reasoning_tokens = completion_tokens_details.get("reasoning_tokens", 0)

            # Log detailed usage if usage_log is provided
            if usage_log:
                usage_log.add_usage(
                    model=f"{provider}/{model}" if provider else model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    response_time_ms=response_time_ms,
                )

            # Log additional metadata
            logging.info(
                f"OpenRouter Response - ID: {response_id}, Provider: {provider}, "
                f"Finish Reason: {finish_reason}, Cached Tokens: {cached_tokens}, "
                f"Reasoning Tokens: {reasoning_tokens}"
            )

            return AgentResponse(
                content=content,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                cost=0.0,
                model=f"{provider}/{model}" if provider else model,
                response_time_ms=response_time_ms,
            )

        except Exception as e:
            logging.error(
                f"Chat completion failed: {str(e)} | response={response_payload}"
            )
            return AgentResponse(
                content=f"Bir hata oluştu: {str(e)}",
                prompt_tokens=0,
                completion_tokens=0,
                input_tokens=0,
                output_tokens=0,
                cost=0.0,
                model=model,
                response_time_ms=0,
            )

    def generate_text(
        self,
        prompt: str,
        model: str = OPENROUTER_GPT_4O,
        temperature: float = 0.7,
        system_message: Optional[str] = None,
        usage_log: Optional[ChatbotUsageLog] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> AgentResponse:
        """
        Tek bir prompt ile metin üretir

        Args:
            prompt: Kullanıcı mesajı
            model: Kullanılacak model
            temperature: Yaratıcılık seviyesi
            system_message: Sistem mesajı (opsiyonel)
            usage_log: Kullanım logları için

        Returns:
            AgentResponse objesi
        """
        messages = []

        if system_message:
            messages.append({"role": "system", "content": system_message})

        messages.append({"role": "user", "content": prompt})

        return self.chat_completion(
            messages,
            model,
            temperature,
            usage_log,
            max_tokens=max_tokens,
            response_format=response_format,
            extra_params=extra_params,
        )

    def multimodal_completion(
        self,
        *,
        text: Optional[str] = None,
        image_urls: Optional[Sequence[str]] = None,
        pdf_files: Optional[
            Sequence[Union[str, Path, Dict[str, Any], bytes, bytearray]]
        ] = None,
        audio_files: Optional[
            Sequence[Union[str, Path, Dict[str, Any], bytes, bytearray]]
        ] = None,
        video_files: Optional[
            Sequence[Union[str, Path, Dict[str, Any], bytes, bytearray]]
        ] = None,
        extra_content: Optional[List[Dict[str, Any]]] = None,
        system_message: Optional[str] = None,
        model: str = OPENROUTER_GPT_4O,
        temperature: float = 0.7,
        usage_log: Optional[ChatbotUsageLog] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> AgentResponse:
        """
        Multimodal içerikleri tek çağrıda işlemek için yardımcı metot.
        OpenRouter'ın multimodal image kılavuzuna göre önce metin, ardından her görseli
        ayrı `image_url` girdisi olarak gönderir.
        """
        content_blocks = self._build_multimodal_content(
            text=text,
            image_urls=image_urls,
            pdf_files=pdf_files,
            audio_files=audio_files,
            video_files=video_files,
            extra_content=extra_content,
        )

        messages: List[Dict[str, Any]] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": content_blocks})

        return self.chat_completion(
            messages,
            model,
            temperature,
            usage_log,
            max_tokens=max_tokens,
            response_format=response_format,
            extra_params=extra_params,
        )

    def image_to_text(
        self,
        image_url: Union[str, List[str]],
        prompt: str,
        model: str = OPENROUTER_GPT_4O,
        usage_log: Optional[ChatbotUsageLog] = None,
    ) -> AgentResponse:
        """
        Görsel analizi yapar

        Args:
            image_url: Görsel URL'si veya URL listesi
            prompt: Analiz için prompt
            model: Kullanılacak model (vision destekleyen model olmalı)
            usage_log: Kullanım logları için

        Returns:
            AgentResponse objesi
        """
        try:
            images: List[str]
            if isinstance(image_url, str):
                images = [image_url]
            elif isinstance(image_url, list):
                images = image_url
            else:
                raise ValueError("image_url must be a string or list of strings")

            return self.multimodal_completion(
                text=prompt,
                image_urls=images,
                model=model,
                usage_log=usage_log,
            )

        except Exception as e:
            logging.error(f"Image to text failed: {str(e)}")
            return AgentResponse(
                content=f"Görsel analizi sırasında hata oluştu: {str(e)}",
                prompt_tokens=0,
                completion_tokens=0,
                input_tokens=0,
                output_tokens=0,
                cost=0.0,
                model=model,
                response_time_ms=0,
            )

    async def create_merged_text_file(
        self, paths: List[DocumentSource], bucket_id: str
    ) -> bool:
        """
        Belgeleri birleştirerek S3'e kaydeder

        Args:
            paths: Belge kaynaklarının listesi
            bucket_id: S3 bucket ID'si

        Returns:
            İşlem başarılı ise True
        """
        try:
            # Clear existing index files
            response = self.s3.s3_get_object_list(prefix=f"{bucket_id}/index-files/")
            for obj in response:
                self.s3.delete_object(obj.get("Key"))

            # Download and merge documents
            docs = download_and_order_files(paths)
            documents = [str(doc) for doc in docs if doc and str(doc).strip()]
            merged_content = "\n".join(documents)

            return self.s3.upload_file(
                path=f"{bucket_id}/index-files/merged.txt", data=merged_content
            )

        except Exception as e:
            logging.error(f"Error in create_merged_text_file: {e}")
            return False

    def chat_completion_detailed(
        self,
        messages: List[Dict],
        model: str = OPENROUTER_GPT_4O,
        temperature: float = 0.7,
        usage_log: Optional[ChatbotUsageLog] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """
        Chat completion API'si kullanarak detaylı response döndürür (tüm metadata ile)

        Args:
            messages: Chat mesajları listesi [{"role": "user", "content": "message"}]
            model: Kullanılacak model
            temperature: Yaratıcılık seviyesi
            usage_log: Kullanım logları için

        Returns:
            OpenRouter'dan gelen tam response
        """
        try:
            response = self._make_request(
                messages,
                model,
                temperature,
                max_tokens,
                response_format,
                extra_params,
            )

            # Log usage if usage_log is provided
            if usage_log:
                usage = response.get("usage", {})
                provider = response.get("provider", "")
                usage_log.add_usage(
                    model=f"{provider}/{model}" if provider else model,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                    response_time_ms=response.get("response_time_ms", 0),
                )

            return response

        except Exception as e:
            logging.error(f"Detailed chat completion failed: {str(e)}")
            return {
                "error": str(e),
                "choices": [{"message": {"content": f"Bir hata oluştu: {str(e)}"}}],
                "model": model,
            }

    def get_available_models(self) -> List[str]:
        """
        Kullanılabilir modellerin listesini döndürür

        Returns:
            Model adları listesi
        """
        return [
            OPENROUTER_GPT_4O,
            OPENROUTER_GPT_4O_MINI,
            OPENROUTER_CLAUDE_3_5_SONNET,
            OPENROUTER_GEMINI_FLASH,
        ]

    def get_response_metadata(self, response: Dict) -> Dict:
        """
        OpenRouter response'ından metadata bilgilerini çıkarır

        Args:
            response: OpenRouter API response'ı

        Returns:
            Metadata bilgileri
        """
        usage = response.get("usage", {})
        choice = response.get("choices", [{}])[0]

        return {
            "response_id": response.get("id", ""),
            "provider": response.get("provider", ""),
            "model": response.get("model", ""),
            "created": response.get("created", 0),
            "system_fingerprint": response.get("system_fingerprint", ""),
            "finish_reason": choice.get("finish_reason", ""),
            "native_finish_reason": choice.get("native_finish_reason", ""),
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "cached_tokens": usage.get("prompt_tokens_details", {}).get(
                    "cached_tokens", 0
                ),
                "reasoning_tokens": usage.get("completion_tokens_details", {}).get(
                    "reasoning_tokens", 0
                ),
            },
            "response_time_ms": response.get("response_time_ms", 0),
        }
