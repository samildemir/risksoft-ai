import asyncio
from typing import List, Dict, Optional, Tuple

# Güncel LangChain modüllerine göre importlar:
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain
from langchain_core.prompts import ChatPromptTemplate
from constants.config import (
    GPT_4o,
    GPT_4o_mini,
    OPENROUTER_GPT_4O,
    OPENROUTER_GPT_4O_MINI,
)
from langchain import hub
from constants.env_variables import OPENAI_API_KEY
from services.open_router_service import OpenRouterService
from core.database import database_engine, db_session, get_database_uri
import logging
from operator import itemgetter

logger = logging.getLogger(__name__)
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain import hub
from typing_extensions import Annotated
from pydantic import ValidationError
from models.schemas import (
    State,
    ChatbotUsageLog,
    ChatbotSqlTemplate,
    SQLQueryResponse,
    SQL_QUERY_RESPONSE_SCHEMA,
)
from prompts.sql_prompts import (
    build_sql_generation_system_prompt,
    build_sql_generation_prompt,
    SQL_ANSWER_SYSTEM_MESSAGE,
    build_sql_answer_prompt,
    build_category_check_prompt,
    build_category_refine_prompt,
    build_answer_verification_prompt,
    build_answer_improvement_prompt,
    ADVANCED_SQL_SYSTEM_MESSAGE,
    build_advanced_sql_prompt,
    build_visualization_prompt,
)
from typing_extensions import TypedDict
import time
import json


class QueryOutput(TypedDict):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid PostgreSQL SQL query."]


class SQLQueryAgentService:
    """
    Gelişmiş SQL Agent servisi:
    - Doğal dil sorgusunu SQL'e dönüştürür
    - Few-shot learning ve semantic similarity desteği
    - Token kullanım takibi, asenkron destek ve hata yönetimi
    - Kontekst için chat history entegrasyonu
    - Hesap bazlı veri erişimi ve güvenliği
    """

    def __init__(self):
        """Database bağlantısı ve LLM kurulumu ile SQL Agent servisini başlatır."""
        self._initialize_database()
        self._initialize_openrouter()
        self._initialize_chatbot_sql_templates()

    def _initialize_database(self) -> None:
        """Belirli tablolar için database bağlantısını başlatır."""
        self.db = SQLDatabase.from_uri(
            get_database_uri(),  # Engine yerine string URI kullanıyoruz
            include_tables=[
                "users",
                "operational_audit_report",
                "incident_report",
                "secg_report",
                "security_tour_control_list",
                "risk_analysis",
                "dfis",
                "workplaces",
                "companies",
                "units",
                "account_types",
            ],
            sample_rows_in_table_info=3,
        )
        self.db_session = db_session()

    def _initialize_chatbot_sql_templates(self) -> None:
        """Chatbot SQL query templates'ını başlatır."""
        self.chatbot_sql_templates = self.db_session.query(
            ChatbotSqlTemplate.input_text,
            ChatbotSqlTemplate.query,
            ChatbotSqlTemplate.description,
        ).all()

    def _initialize_openrouter(self) -> None:
        """OpenRouter servisini başlatır."""
        self.openrouter_service = OpenRouterService()

    def write_query(self, state: State):
        """Generate SQL query to fetch information using OpenRouter."""
        try:
            # Template'leri formatla
            formatted_templates = "\n\n".join(
                [
                    f"Input: {template[0]}\nQuery: {template[1]}\nDescription: {template[2]}"
                    for template in self.chatbot_sql_templates
                ]
            )

            db_schema = json.loads(open("constants/db_schema.json").read())
            db_schema_pretty = json.dumps(db_schema, indent=2)
            system_message = build_sql_generation_system_prompt(
                account_id=state["account_id"],
                usable_tables=list(self.db.get_usable_table_names()),
                db_schema_pretty=db_schema_pretty,
                formatted_templates=formatted_templates,
            )

            prompt = build_sql_generation_prompt(state["question"])

            # Use OpenRouter with Claude for superior SQL generation
            try:
                # Claude 3.5 Sonnet is excellent for SQL - use it as primary choice
                response_obj = self.openrouter_service.generate_text(
                    prompt=prompt,
                    model="anthropic/claude-sonnet-4",  # Claude excels at structured reasoning and SQL
                    temperature=0.05,  # Very low temperature for precise, deterministic SQL
                    system_message=system_message,
                    usage_log=state.get("usage_log"),
                    response_format=SQL_QUERY_RESPONSE_SCHEMA,
                )
                parsed_response = self._parse_sql_query_response(response_obj.content)
                logger.info(f"SQL generated using Claude 3.5 Sonnet")

            except Exception as claude_error:
                logger.warning(f"Claude failed, falling back to GPT-4o: {claude_error}")
                # Fallback to GPT-4o for SQL generation
                response_obj = self.openrouter_service.generate_text(
                    prompt=prompt,
                    model=OPENROUTER_GPT_4O,  # Use full GPT-4o for better SQL reasoning
                    temperature=0.1,
                    system_message=system_message,
                    usage_log=state.get("usage_log"),
                    response_format=SQL_QUERY_RESPONSE_SCHEMA,
                )
                parsed_response = self._parse_sql_query_response(response_obj.content)

            # Extract SQL query from response (remove any markdown formatting)
            query = parsed_response.sql_query.strip()
            if query.startswith("```sql"):
                query = query.replace("```sql", "").replace("```", "").strip()
            elif query.startswith("```"):
                query = query.replace("```", "").strip()

            # Clean up the query
            query = query.split(";")[0]  # Remove semicolon and anything after
            query = query.strip()

            logger.info(f"Generated SQL query: {query}")

            # Basic validation - let LLM handle the complex logic
            query_lower = query.lower()

            # Simple check: if no account_id filtering is present anywhere in the query, flag it
            if "account_id" not in query_lower:
                logger.warning(
                    f"Query appears to be missing account_id filtering: {query}"
                )
                # Log for monitoring, but don't modify - trust the LLM's decision

            state["query"] = query

        except Exception as e:
            logger.error(f"Error in write_query: {str(e)}")
            # Fallback to simple query
            state["query"] = (
                f"SELECT 'Error generating query: {str(e)}' as error_message"
            )

    def execute_query(self, state: State):
        """Execute SQL query."""
        execute_query_tool = QuerySQLDatabaseTool(db=self.db)
        cleaned_query = state["query"].strip().replace("\n", " ")
        temp = execute_query_tool.invoke(cleaned_query)
        state["result"] = temp

    def generate_answer(self, state: State, account_id: int):
        """Answer question using retrieved information as context."""
        # Get account information for context
        account_info_query = f"""
        SELECT a.id, a.bucket_id, at.name as account_type_name
        FROM accounts a
        JOIN account_types at ON a.account_type_id = at.id
        WHERE a.id = {account_id}
        """

        execute_query_tool = QuerySQLDatabaseTool(db=self.db)
        account_info = execute_query_tool.invoke(account_info_query)

        # Format account info for the prompt
        account_context = ""
        try:
            # Parse account info - it could be a string representation of tuples or actual data
            if isinstance(account_info, str):
                # Try to evaluate if it's a string representation of tuples
                import ast

                try:
                    parsed_info = ast.literal_eval(account_info)
                    if isinstance(parsed_info, list) and len(parsed_info) > 0:
                        if (
                            isinstance(parsed_info[0], tuple)
                            and len(parsed_info[0]) >= 3
                        ):
                            acc_id, bucket_id, acc_type = parsed_info[0][:3]
                            account_context = f"""
                    Account Information:
                    - Account ID: {acc_id}
                    - Account Type: {acc_type}
                    - Bucket ID: {bucket_id}
                    """
                        else:
                            logger.warning(
                                f"Unexpected tuple format in account info: {parsed_info[0]}"
                            )
                            account_context = "\nAccount Information: Available but format unexpected."
                    else:
                        account_context = "\nAccount Information: No data available."
                except (ValueError, SyntaxError):
                    # If it's just a regular string error message
                    if "error" in account_info.lower():
                        account_context = f"\nAccount Information: {account_info}"
                    else:
                        account_context = (
                            "\nAccount Information: Could not parse account data."
                        )

            elif isinstance(account_info, list) and len(account_info) > 0:
                # Handle list/tuple format directly
                if (
                    isinstance(account_info[0], (tuple, list))
                    and len(account_info[0]) >= 3
                ):
                    acc_id, bucket_id, acc_type = account_info[0][:3]
                    account_context = f"""
                    Account Information:
                    - Account ID: {acc_id}
                    - Account Type: {acc_type}
                    - Bucket ID: {bucket_id}
                    """
                elif isinstance(account_info[0], dict):
                    account = account_info[0]
                    acc_id = account.get("id", "N/A")
                    acc_type = account.get("account_type_name", "N/A")
                    bucket_id = account.get("bucket_id", "N/A")
                    account_context = f"""
                    Account Information:
                    - Account ID: {acc_id}
                    - Account Type: {acc_type}
                    - Bucket ID: {bucket_id}
                    """
                else:
                    logger.warning(f"Unexpected account info format: {account_info[0]}")
                    account_context = (
                        "\nAccount Information: Available but format unexpected."
                    )
            else:
                account_context = "\nAccount Information: Not found or empty."

        except Exception as e:
            logger.error(
                f"Error parsing account context: {e}. Raw data: {account_info}"
            )
            account_context = "\nAccount Information: Error retrieving details."

        response_obj = self.openrouter_service.generate_text(
            prompt=build_sql_answer_prompt(
                question=state["question"],
                query=state["query"],
                result=state["result"],
                account_context=account_context,
            ),
            model=OPENROUTER_GPT_4O_MINI,
            temperature=0.3,
            system_message=SQL_ANSWER_SYSTEM_MESSAGE,
            usage_log=state.get("usage_log"),
        )
        state["answer"] = response_obj.content

        # Check if the answer needs refinement (contains category codes)
        self.refine_answer(state)

    def refine_answer(self, state: State):
        """
        Refine the answer by replacing category codes with their proper names.
        This is especially useful for category codes that should be translated to human-readable names.
        """
        answer = state["answer"]
        result = state["result"]

        # Check if the answer contains category codes that might need translation
        if any(
            code_indicator in answer.lower()
            for code_indicator in ["kategori", "category"]
        ):
            # Create a query to check if we need to translate any category codes
            category_check_obj = self.openrouter_service.generate_text(
                prompt=build_category_check_prompt(answer, result),
                model=OPENROUTER_GPT_4O_MINI,
                temperature=0.1,
                usage_log=state.get("usage_log"),
            )
            category_check_response_content = category_check_obj.content

            # If the model identifies codes that need translation
            if "yes" in category_check_response_content.lower():
                # Query the database to get the proper category names
                try:
                    # Create a query to get category translations
                    translation_query = """
                    SELECT key, title_tr, title_en 
                    FROM incident_report_categories
                    """

                    # Execute the query
                    execute_query_tool = QuerySQLDatabaseTool(db=self.db)
                    category_translations = execute_query_tool.invoke(translation_query)

                    # Create a prompt to refine the answer with proper category names
                    refined_obj = self.openrouter_service.generate_text(
                        prompt=build_category_refine_prompt(
                            answer=answer,
                            translations=category_translations,
                        ),
                        model=OPENROUTER_GPT_4O_MINI,
                        temperature=0.2,
                        usage_log=state.get("usage_log"),
                    )
                    state["answer"] = refined_obj.content

                except Exception as e:
                    logger.error(
                        f"Error refining answer with category translations: {str(e)}"
                    )
                    # Keep the original answer if refinement fails
                    pass

        # After refining, verify and improve the answer if needed
        self.verify_and_improve_answer(state)

    def verify_and_improve_answer(self, state: State):
        """
        Verify if the answer is satisfactory and try to improve it if needed.
        This is a second-pass check to ensure high-quality responses.
        """
        answer = state["answer"]
        question = state["question"]
        result = state["result"]
        query = state["query"]

        try:
            verification_obj = self.openrouter_service.generate_text(
                prompt=build_answer_verification_prompt(
                    question=question,
                    query=query,
                    result=result,
                    answer=answer,
                ),
                model=OPENROUTER_GPT_4O_MINI,
                temperature=0.1,
                usage_log=state.get("usage_log"),
            )
            verification_response_content = verification_obj.content

            # Check if the answer needs improvement
            if any(
                indicator in verification_response_content.lower()
                for indicator in ["below 8", "improve", "should be", "could be better"]
            ):
                # Create an improvement prompt
                improved_obj = self.openrouter_service.generate_text(
                    prompt=build_answer_improvement_prompt(
                        question=question,
                        query=query,
                        result=result,
                        answer=answer,
                        evaluation=verification_response_content,
                    ),
                    model=OPENROUTER_GPT_4O_MINI,
                    temperature=0.3,
                    usage_log=state.get("usage_log"),
                )
                state["answer"] = improved_obj.content

        except Exception as e:
            logger.error(f"Error verifying and improving answer: {str(e)}")
            # Keep the original answer if improvement fails
            pass

    @staticmethod
    def _parse_sql_query_response(response_text: str) -> SQLQueryResponse:
        """
        Validate the structured SQL response. Falls back to raw text when parsing fails.
        """
        payload = (response_text or "").strip()
        try:
            return SQLQueryResponse.model_validate_json(payload)
        except (ValidationError, ValueError) as exc:
            logger.warning("SQL query response parse failed: %s", exc)
            return SQLQueryResponse(sql_query=payload, reasoning=None)

    async def chat_with_database(
        self, question: str, account_id: int
    ) -> Tuple[str, ChatbotUsageLog]:
        """Process database queries and track token usage."""
        try:
            usage_log = ChatbotUsageLog()
            state = State(question=question)

            state["usage_log"] = usage_log
            state["account_id"] = account_id  # Add account_id to state

            # Execute write_query (usage tracking handled internally in OpenRouter calls)
            self.write_query(state)

            logger.info(
                f"SQL Agent State: Question='{state['question']}', Account ID={state.get('account_id', 'N/A')}"
            )
            # Execute query doesn't use LLM
            self.execute_query(state)

            # Execute answer generation (usage tracking handled internally in OpenRouter calls)
            self.generate_answer(state, account_id)

            return state["answer"], usage_log

        except Exception as e:
            error_msg = f"SQL agent hatası: {str(e)}"
            logger.error(error_msg)
            error_log = ChatbotUsageLog.create_error_log(error_msg)
            raise Exception(error_msg)

    async def advanced_database_chat(
        self,
        question: str,
        account_id: int,
        model: str = OPENROUTER_GPT_4O_MINI,
        temperature: float = 0.1,
    ) -> Tuple[str, ChatbotUsageLog]:
        """
        Advanced database chat with customizable OpenRouter model selection.

        Args:
            question: User question
            account_id: Account identifier
            model: OpenRouter model to use
            temperature: Response creativity level

        Returns:
            Tuple containing (response content, ChatbotUsageLog)
        """
        try:

            usage_log = ChatbotUsageLog()
            state = State(question=question)

            state["usage_log"] = usage_log
            state["account_id"] = account_id

            # Execute SQL query generation and execution
            self.write_query(state)
            self.execute_query(state)

            # Enhanced answer generation with specified model
            system_message = ADVANCED_SQL_SYSTEM_MESSAGE
            prompt = build_advanced_sql_prompt(
                question=state["question"],
                query=state["query"],
                result=state["result"],
                account_id=account_id,
            )

            logger.info(
                f"Advanced Database Chat - Model: {model}, Temperature: {temperature}"
            )
            # Use specified OpenRouter model
            response_obj = self.openrouter_service.generate_text(
                prompt=prompt,
                model="google/gemini-2.5-flash",
                temperature=temperature,
                system_message=system_message,
                usage_log=usage_log,
            )

            final_answer = response_obj.content

            # Enhanced post-processing for advanced model
            if "gpt-4o" in model.lower() or "claude" in model.lower():
                # Add data visualization suggestions for advanced models
                viz_obj = self.openrouter_service.generate_text(
                    prompt=build_visualization_prompt(state["result"]),
                    model=OPENROUTER_GPT_4O_MINI,
                    temperature=0.2,
                    usage_log=usage_log,
                )

                final_answer += f"\n\n📊 Görselleştirme Önerisi: {viz_obj.content}"

            return final_answer, usage_log

        except Exception as e:
            error_msg = f"Advanced SQL agent error: {str(e)}"
            logger.error(error_msg)
            error_log = ChatbotUsageLog.create_error_log(error_msg)
            raise Exception(error_msg)

    def get_available_models(self) -> List[str]:
        """Get list of available OpenRouter models for SQL processing."""
        return self.openrouter_service.get_available_models()
