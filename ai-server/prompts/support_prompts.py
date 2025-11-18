import json
from typing import Any, Dict, List, Optional


class SupportPrompts:
    """Helper utilities for building/processing support assistant prompts."""

    def build_support_prompt(
        self,
        *,
        user_message: str,
        improved_question: str,
        agent_answer: str,
        sources: List[str],
        context_snapshot: Dict[str, Any],
    ) -> str:
        profile_block = self._format_user_profile(context_snapshot.get("user_profile"))
        extra_block = self._format_generic_context(context_snapshot)
        sources_block = ", ".join(sources) if sources else "casual"

        return f"""
Sen Risksoft platformunun teknik destek ajanısın.
Kullanıcıya uygulanabilir çözüm ve yönlendirmeler sunmak için aşağıdaki bağlamı kullan:
- Kullanıcı mesajı: {user_message}
- Düzenlenmiş soru: {improved_question}
- Ajan cevabı: {agent_answer}
- Kullanılan kaynak türleri: {sources_block}
- Kullanıcı profili: {profile_block}
- Destek bağlamı:
{extra_block}

Kurallar:
1. Yanıt tamamen Türkçe olmalı, sakin ve profesyonel bir ton kullan.
2. Teknik çözüm adımlarını olabildiğince sırayla ve maddeler halinde aktar; gerekiyorsa Risksoft içindeki menü yollarını belirt.
3. Emin olmadığın durumlarda varsayım yapma; kullanıcıdan hangi ek bilgiyi istediğini açıkça söyle.
4. Gerektiğinde kullanıcıyı Risksoft içindeki ilgili sayfalara veya ekiplere yönlendir (ör. "Ayarlar > Bildirimler").
5. Yanıt sonunda kullanıcıdan beklenen aksiyonları madde madde listele.
6. Sorun canlı destek müdahalesi gerektiriyorsa nedeniyle birlikte belirt ve ilgili öneriyi ekle.
7. Yanıtın sonunda aşağıdaki JSON şemasını kullanarak rapor üret.

ÇIKTI ŞEMASI (kesinlikle bu anahtarları kullan):
{{
  "answer": "<kullanıcıya verilecek tam yanıt>",
  "confidence": 0.0 - 1.0 arasında ondalık sayı,
  "needs_human_support": true veya false,
  "intent": "general" | "escalate" | "error" | "follow_up",
  "support_actions": ["<öneri 1>", "<öneri 2>"],
  "escalation_reason": "<destek gerekiyorsa kısa açıklama>"
}}

ÖNEMLİ:
- Eğer yanıt kesin değilse confidence 0.5'in altında olmalı ve needs_human_support true olmalı.
- Teknik müdahale gerektiğini düşünüyorsan support_actions içinde "Canlı destek talebi oluştur" maddesini ekle.
- JSON dışında ekstra metin üretme.

Kullanıcı mesajı: {user_message}
"""

    def parse_support_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM output into structured dict."""
        cleaned = (
            response_text.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

        # Fallback if parsing fails
        return {
            "answer": response_text.strip(),
            "confidence": 0.3,
            "needs_human_support": True,
            "intent": "error",
            "support_actions": ["Canlı destek talebi oluştur"],
            "escalation_reason": "LLM yanıtı beklenen formatta değil",
        }

    @staticmethod
    def _format_user_profile(profile: Optional[Dict[str, Any]]) -> str:
        if not profile:
            return "Bilgi bulunamadı"

        parts = []
        full_name = " ".join(
            filter(None, [profile.get("name"), profile.get("surname")])
        ).strip()
        if full_name:
            parts.append(full_name)
        if profile.get("task"):
            parts.append(profile["task"])
        if profile.get("account_id"):
            parts.append(f"Account #{profile['account_id']}")
        return " - ".join(parts) if parts else "Bilgi bulunamadı"

    def _format_generic_context(self, context: Dict[str, Any]) -> str:
        if not context:
            return "- Ek bilgi bulunamadı"

        ignored_keys = {"user_profile"}
        lines: List[str] = []
        for key, value in context.items():
            if key in ignored_keys or value in (None, [], {}):
                continue

            if isinstance(value, list):
                preview_items = value[:3]
                rendered = ", ".join(
                    self._summarize_dict(item) if isinstance(item, dict) else str(item)
                    for item in preview_items
                )
                more = "" if len(value) <= 3 else f" (+{len(value)-3} kayıt)"
                lines.append(f"- {key}: {rendered}{more}")
            elif isinstance(value, dict):
                lines.append(f"- {key}: {self._summarize_dict(value)}")
            else:
                lines.append(f"- {key}: {value}")

        return "\n".join(lines) if lines else "- Ek bilgi bulunamadı"

    @staticmethod
    def _summarize_dict(payload: Dict[str, Any]) -> str:
        if not payload:
            return "—"
        parts = []
        for field in ("title", "name", "status", "id"):
            if payload.get(field):
                parts.append(str(payload[field]))
        if not parts:
            parts = [f"{k}: {v}" for k, v in list(payload.items())[:3]]
        return " | ".join(parts)
