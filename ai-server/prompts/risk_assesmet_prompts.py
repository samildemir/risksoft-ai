from typing import Any, Dict, List, Optional, Tuple


LANGUAGE_LABELS = {"tr": "Turkish", "en": "English"}

AFFECTED_PEOPLE_OPTIONS = {
    "tr": [
        "Maruz kalan kişi",
        "Yakın mesafede bulunan kişi/kişiler",
        "Uzak mesafede bulunan kişi/kişiler",
        "Tüm tesis çalışanları",
        "Komşu tesiste bulunan kişiler",
        "Sokak/Caddede bulunan yerleşim alanları",
        "Mahallede bulunan yerleşim alanları",
        "Birden fazla mahallede bulunan yerleşim alanları",
        "İlçe/Şehrin önemli büyüklüğünü kapsayan yerleşim alanları",
    ],
    "en": [
        "Exposed person",
        "People in close proximity",
        "People at a distance",
        "All facility workers",
        "People in neighboring facilities",
        "Residential areas on street/avenue",
        "Residential areas in neighborhood",
        "Residential areas across multiple neighborhoods",
        "Residential areas covering significant portion of district/city",
    ],
}

METHOD_REQUIREMENTS = {
    "5X5": {
        "label": "5x5",
        "instructions": [
            "Set `possibility` as an integer between 1 (rare) and 5 (very likely).",
            "Set `intensity` as an integer between 1 (negligible) and 5 (catastrophic).",
            "Justify every numeric rating with explicit evidence from the images.",
        ],
        "fields": ["possibility", "intensity"],
    },
    "FINE_KINNEY": {
        "label": "Fine-Kinney",
        "instructions": [
            "Set `possibility` using one of these values: 0.1, 0.2, 0.5, 1, 3, 6, or 10.",
            "Set `intensity` using one of these values: 1, 3, 7, 15, 40, or 100.",
            "Set `frequency` using one of these values: 0.5, 1, 2, 3, 6, or 10.",
            "Link each numeric rating to specific observations in the images.",
        ],
        "fields": ["possibility", "intensity", "frequency"],
    },
}

FALLBACK_METHOD_INSTRUCTIONS = [
    "Set `possibility` as an integer between 1 and 5 and justify the rating with image evidence.",
    "Set `intensity` as an integer between 1 and 5 and justify the rating with image evidence.",
    "If your methodology requires additional factors (e.g., `frequency`), include them and explain the supporting evidence.",
]

SCORING_REFERENCES = {
    "5X5": {
        "possibility": [
            {
                "score": "1",
                "description": "Rare – Occurs only in exceptional or unforeseen circumstances.",
            },
            {
                "score": "2",
                "description": "Unlikely – Could happen but not expected during normal operations.",
            },
            {
                "score": "3",
                "description": "Possible – Has occurred before or could happen with some regularity.",
            },
            {
                "score": "4",
                "description": "Likely – Expected to happen several times during operations.",
            },
            {
                "score": "5",
                "description": "Almost certain – Will occur frequently unless conditions change.",
            },
        ],
        "intensity": [
            {
                "score": "1",
                "description": "Negligible – No injury or very minor first-aid case.",
            },
            {
                "score": "2",
                "description": "Minor – Minor injury requiring basic treatment, no lost time.",
            },
            {
                "score": "3",
                "description": "Moderate – Injury causing lost time or temporary disability.",
            },
            {
                "score": "4",
                "description": "Major – Serious injury or permanent partial disability.",
            },
            {
                "score": "5",
                "description": "Catastrophic – Fatality or multiple severe injuries.",
            },
        ],
    },
    "FINE_KINNEY": {
        "possibility": [
            {
                "score": "0.1",
                "description": "Almost impossible – Conceivable but highly improbable.",
            },
            {
                "score": "0.2",
                "description": "Practically impossible – Would require a rare combination of events.",
            },
            {
                "score": "0.5",
                "description": "Very unlikely – Could happen, but not expected in the foreseeable future.",
            },
            {
                "score": "1",
                "description": "Unlikely – Known to occur but not in typical operations.",
            },
            {
                "score": "3",
                "description": "Possible – Has occurred or is likely under certain conditions.",
            },
            {
                "score": "6",
                "description": "Likely – Expected to occur repeatedly during operations.",
            },
            {
                "score": "10",
                "description": "Almost certain – Occurs frequently or continuously.",
            },
        ],
        "intensity": [
            {
                "score": "1",
                "description": "Very slight – Reversible injury requiring minimal treatment.",
            },
            {
                "score": "3",
                "description": "Slight – Medical attention required, short-term absence.",
            },
            {
                "score": "7",
                "description": "Noticeable – Serious injury causing significant absence.",
            },
            {
                "score": "15",
                "description": "Serious – Severe injury or permanent partial disability.",
            },
            {
                "score": "40",
                "description": "Very serious – Life-threatening injury or multiple casualties.",
            },
            {
                "score": "100",
                "description": "Catastrophic – Fatality or multiple fatalities.",
            },
        ],
        "frequency": [
            {
                "score": "0.5",
                "description": "Very rare exposure – Less than once in 10 years.",
            },
            {
                "score": "1",
                "description": "Rare exposure – About once in several years.",
            },
            {
                "score": "2",
                "description": "Occasional exposure – A few times per year.",
            },
            {"score": "3", "description": "Frequent exposure – Monthly or more often."},
            {
                "score": "6",
                "description": "Very frequent – Weekly or several times per month.",
            },
            {"score": "10", "description": "Continuous – Daily or constant exposure."},
        ],
    },
}


class RiskAssessmentPrompts:
    def __init__(self):
        pass

    def _normalize_language(self, language: Optional[str]) -> str:
        if not language:
            return "tr"
        normalized = language.lower()
        return normalized if normalized in AFFECTED_PEOPLE_OPTIONS else "tr"

    def _get_output_language_label(self, language_code: str) -> str:
        return LANGUAGE_LABELS.get(language_code, LANGUAGE_LABELS["tr"])

    def _format_affected_people_options(self, language_code: str) -> str:
        options = AFFECTED_PEOPLE_OPTIONS.get(
            language_code, AFFECTED_PEOPLE_OPTIONS["tr"]
        )
        return "\n".join([f"- {option}" for option in options])

    def _build_scoring_reference(self, method_key: str, output_language: str) -> str:
        reference = SCORING_REFERENCES.get(method_key)
        if not reference:
            return (
                "Scoring reference: Provide detailed, evidence-based justifications for every "
                "numeric value and translate descriptors into the requested output language."
            )

        lines: List[str] = [
            f"Use the following descriptors (translate them into {output_language}) when explaining why each score was selected:",
        ]
        for dimension, entries in reference.items():
            title = dimension.replace("_", " ").title()
            lines.append(f"{title}:")
            for entry in entries:
                lines.append(f"- Score {entry['score']}: {entry['description']}")
        return "\n".join(lines)

    def _get_method_instruction_block(
        self, analysis_method: str, output_language: str
    ) -> Tuple[str, bool, str, str]:
        method_key = (analysis_method or "").upper()
        method_info = METHOD_REQUIREMENTS.get(method_key)
        if method_info:
            instruction_lines = "\n".join(
                [f"- {instruction}" for instruction in method_info["instructions"]]
            )
            include_frequency = "frequency" in method_info["fields"]
            scoring_reference = self._build_scoring_reference(
                method_key, output_language
            )
            return (
                instruction_lines,
                include_frequency,
                method_info["label"],
                scoring_reference,
            )

        fallback_lines = "\n".join(
            [f"- {instruction}" for instruction in FALLBACK_METHOD_INSTRUCTIONS]
        )
        scoring_reference = self._build_scoring_reference(method_key, output_language)
        return fallback_lines, True, analysis_method or "selected", scoring_reference

    def _split_uploaded_resources(
        self, uploaded_documents: Optional[List[Dict[str, Any]]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        image_exts = {"jpg", "jpeg", "png", "gif", "bmp", "webp", "tif", "tiff"}
        image_resources: List[Dict[str, Any]] = []
        other_resources: List[Dict[str, Any]] = []

        if not uploaded_documents:
            return image_resources, other_resources

        for doc in uploaded_documents:
            if not isinstance(doc, dict):
                continue

            entry = {
                "url": doc.get("url") or doc.get("path"),
                "name": doc.get("name"),
                "size": doc.get("size"),
                "key": doc.get("key"),
                "mime_type": doc.get("mime_type"),
            }

            base_name = entry["url"] or ""
            if "?" in base_name:
                base_name = base_name.split("?")[0]
            if not base_name and entry["name"]:
                base_name = entry["name"]

            extension = ""
            if base_name and "." in base_name:
                extension = base_name.rsplit(".", 1)[-1].lower()

            mime_type = (entry.get("mime_type") or "").lower()
            is_image = extension in image_exts or (
                mime_type.startswith("image/") if mime_type else False
            )

            if is_image:
                image_resources.append(entry)
            else:
                other_resources.append(entry)

        return image_resources, other_resources

    def _format_resource_lines(self, resources: Optional[List[Any]], label: str) -> str:
        if not resources:
            return f"- (No {label.lower()}s provided)"

        lines: List[str] = []
        for idx, resource in enumerate(resources):
            name = None
            size = None
            key = None
            url = None

            if isinstance(resource, dict):
                name = resource.get("name")
                size = resource.get("size")
                key = resource.get("key")
                url = (
                    resource.get("url")
                    or resource.get("path")
                    or resource.get("source")
                )
            elif isinstance(resource, str):
                url = resource
            else:
                continue

            descriptor_parts = [f"{label} {idx + 1}"]
            detail_parts = []
            if name:
                detail_parts.append(str(name))
            if size:
                detail_parts.append(f"{size} MB")
            if detail_parts:
                descriptor_parts.append(f"({', '.join(detail_parts)})")

            descriptor = " ".join(descriptor_parts)
            line = f"- {descriptor}"
            if url:
                line += f": {url}"
            lines.append(line)
            if key:
                lines.append(f"  storage_key: {key}")

        return "\n".join(lines)

    def _build_json_template(self, language_code: str, include_frequency: bool) -> str:
        output_language = self._get_output_language_label(language_code)
        lines = [
            "{",
            f'  "legal_basis": "<List at least two relevant legal references in {output_language}, linking each citation to the observed hazards and control list focus>",',
            f'  "affected_people": ["<Select every impacted group in {output_language}>"],',
            f'  "risks": "<Describe all observable controls or confirm their absence in {output_language}; note verification steps if visibility is limited>",',
            f'  "cautions": "<Use newline-separated bullet points (e.g., - ...) in {output_language} to cover immediate, cascading, and long-term risks; cite image evidence, question context, and user context explicitly>",',
            f'  "current_cautions": "<Provide layered preventive and corrective actions in {output_language}, grouping engineering, administrative, and PPE controls with responsible roles and target timelines>",',
            '  "possibility": 3,',
        ]
        if include_frequency:
            lines.append('  "intensity": 4,')
            lines.append('  "frequency": 2')
        else:
            lines.append('  "intensity": 4')
        lines.append("}")
        return "\n".join(lines)

    def generate_risk_assessment_questions(self, title: str, description: str) -> str:
        return f"""
        You are an occupational health and safety expert. Based on the given title and description, create specific risk assessment questions that would help identify potential hazards and safety measures.

        Title: {title}
        Description: {description}

        Generate practical, actionable questions that a safety inspector would ask during a risk assessment. Focus on:
        - Physical hazards and conditions
        - Safety measures and controls
        - Compliance with safety standards
        - Equipment and environmental factors
        - Emergency preparedness

        Return your response as a JSON array of strings, where each string is a specific question. Each question should be clear, direct, and focused on a specific safety aspect.

        Examples of good questions:
        - "Are ground deformations, collapses, and bumps eliminated?"
        - "Are stair widths and step heights appropriate?"
        - "Is personal protective equipment properly maintained and accessible?"
        - "Are emergency exits clearly marked and unobstructed?"

        Important: Return the questions in the same language as the title and description provided. If the title and description are in Turkish, return questions in Turkish. If they are in English, return questions in English.

        Format your response as a clean JSON array:
        ["Question 1", "Question 2", "Question 3", ...]
        """

    def merge_risk_assessments(
        self,
        image_paths: List[Any],
        analysis_method: str,
        language: str,
        additional_context: Optional[str] = None,
        question_context: Optional[Dict[str, Any]] = None,
        supporting_documents: Optional[List[Any]] = None,
    ) -> str:
        language_code = self._normalize_language(language)
        output_language = self._get_output_language_label(language_code)
        affected_people_options = self._format_affected_people_options(language_code)
        (
            method_instructions,
            include_frequency,
            method_label,
            scoring_reference,
        ) = self._get_method_instruction_block(analysis_method, output_language)
        json_template = self._build_json_template(language_code, include_frequency)

        images_section = self._format_resource_lines(image_paths, "Image")

        supporting_documents_section = ""
        if supporting_documents:
            supporting_documents_section = (
                "\nSupporting documents:\n"
                + self._format_resource_lines(supporting_documents, "Document")
            )

        context_section = ""
        if additional_context:
            context_section = f"\nUser context:\n{additional_context}\n"

        question_section = ""
        if question_context:
            question_section = "\nQuestion context:\n" + "\n".join(
                [
                    f"- {key}: {value}"
                    for key, value in question_context.items()
                    if value is not None
                ]
            )

        return f"""
You are an experienced occupational health and safety expert. Analyze the provided images directly to produce a risk assessment using the {method_label} methodology.

Images to analyze:
{images_section}
{supporting_documents_section}
{context_section}{question_section}
Grounding rules:
- Base every conclusion strictly on visual evidence from the images and on the content of the supporting documents listed above.
- Confirm whether each image URL is reachable; if you cannot load an image or the visual content is unclear, state "Image <index> unavailable" in your findings and avoid speculation.
- Reference image numbers (and document identifiers when applicable) when citing evidence, e.g., "Image 1" or "Document 2".
- If no images are available, state this clearly in the JSON output and avoid inventing details.
- Do not rely on previously generated descriptions; inspect the images and documents directly each time.

Context integration requirements:
- Explicitly weave the provided user context into your analysis; quote critical phrases or data points when relevant.
- Tie your assessment to the control list or question context (e.g., reference the question text, checklist name, IDs) and explain how the observed hazards affect compliance.
- Call out any assumptions you cannot verify due to limited visibility and recommend follow-up inspection steps or document reviews.

Analytical expectations:
- In `risks`, evaluate what is visibly compliant, what is missing, and how to confirm uncertain controls (include verification methods when visibility is limited).
- In `cautions`, cover immediate, cascading, and long-term effects; articulate severity reasoning aligned with the {method_label} ratings.
- In `current_cautions`, combine engineering, administrative, and PPE measures, assign responsible roles, and suggest implementation timelines.

Method requirements:
{method_instructions}

Scoring guidance:
{scoring_reference}

Output requirements:
- Return a single JSON object only; do not include additional prose.
- Write every field value in {output_language} with full sentences or bullet lists as instructed above.
- Select `affected_people` entries exclusively from this list ({output_language}):
{affected_people_options}
- When citing numeric scores, explicitly mention (within your narrative text) the descriptor that matches each selected value and explain how the visual evidence supports it.
- Provide at least two legal citations (e.g., Law 6331, Law 4857, OSHA/ISO standards) and connect them to the findings.
- Ensure every risk, control, and rating is explicitly supported by visual evidence or clearly stated limitations.

Use the following JSON structure and replace the placeholders with your findings:
{json_template}
"""

    def merge_risk_assessments_ai_help(
        self,
        question: str,
        control_list_name: str,
        keywords: Optional[str] = None,
        uploaded_documents: Optional[List[Dict[str, Any]]] = None,
        language: str = "tr",
        question_id: Optional[int] = None,
        analysis_method: str = "FINE_KINNEY",
    ) -> str:
        payload = self.build_ai_help_prompt_payload(
            question=question,
            control_list_name=control_list_name,
            keywords=keywords,
            uploaded_documents=uploaded_documents,
            language=language,
            question_id=question_id,
            analysis_method=analysis_method,
        )
        return payload["prompt"]

    def build_ai_help_prompt_payload(
        self,
        question: str,
        control_list_name: str,
        keywords: Optional[str] = None,
        uploaded_documents: Optional[List[Dict[str, Any]]] = None,
        language: str = "tr",
        question_id: Optional[int] = None,
        analysis_method: str = "FINE_KINNEY",
    ) -> Dict[str, Any]:
        """
        Helper that mirrors merge_risk_assessments_ai_help but also returns the structured
        media metadata so callers can supply actual image inputs to the LLM.
        """
        images, supporting_documents = self._split_uploaded_resources(uploaded_documents)

        question_context: Dict[str, Any] = {
            "Control list": control_list_name,
            "Question": question,
        }
        if question_id is not None:
            question_context["Question ID"] = question_id
        if keywords and keywords.strip():
            question_context["Keywords"] = keywords.strip()

        additional_context_lines: List[str] = []
        if keywords and keywords.strip():
            additional_context_lines.append(
                "Keywords provided by the user must be addressed explicitly in the findings."
            )
        if supporting_documents:
            additional_context_lines.append(
                "Supporting documents are listed below; cite them explicitly alongside relevant observations."
            )

        additional_context = (
            "\n".join(additional_context_lines) if additional_context_lines else None
        )

        prompt_text = self.merge_risk_assessments(
            image_paths=images,
            analysis_method=analysis_method,
            language=language,
            additional_context=additional_context,
            question_context=question_context,
            supporting_documents=supporting_documents,
        )

        image_urls = [
            entry.get("url")
            for entry in images
            if isinstance(entry, dict) and entry.get("url")
        ]

        return {
            "prompt": prompt_text,
            "image_resources": images,
            "supporting_documents": supporting_documents,
            "image_urls": image_urls,
        }
