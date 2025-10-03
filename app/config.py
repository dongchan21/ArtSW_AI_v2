# backend/app/config.py

from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    애플리케이션의 모든 환경 변수를 관리하는 Pydantic BaseSettings 클래스.
    .env 파일을 사용하여 설정값을 로드합니다 (pydantic-settings 필요).
    """

    # --- 서버 설정 ---
    API_V1_STR: str = "/api/v1"
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 8000
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")

    # --- LLM/임베딩 설정 ---
    # LLM API 키
    LLM_API_KEY: Optional[str] = Field(default=None, env="LLM_API_KEY")
    # RAG 답변 생성에 사용할 LLM 모델(Llama)
    LLM_MODEL_NAME: str = Field(default="meta-llama/Meta-Llama-3-8B-Instruct", env="LLM_MODEL_NAME")
    # 임베딩 벡터 생성에 사용할 모델 이름 (예: text-embedding-ada-002)
    LLM_API_BASE: str = Field(default="http://localhost:11434/v1", env="LLM_API_BASE")
    # LoRA 질문 추천 등에 사용할 경량 LLM 모델 경로 (Hugging Face)
    EMBEDDING_MODEL_NAME: str = Field(default="Bge-m3", env="EMBEDDING_MODEL_NAME")
    # 주석: 경량 LLM 분류 모델 경로 (LoRA용)
    CLASSIFIER_MODEL_PATH: str = Field(default="local/llama3-classifier", env="CLASSIFIER_MODEL_PATH")

    # --- 벡터 DB 설정 (Pinecone 예시) ---
    # 주석: 벡터 DB 서비스 제공자 이름
    VECTOR_DB_PROVIDER: str = Field(default="PINECONE", env="VECTOR_DB_PROVIDER")
    # 주석: 벡터 DB API 키
    PINECONE_API_KEY: str = Field(..., env="PINECONE_API_KEY")
    # 주석: 벡터 DB 환경/리전
    PINECONE_ENVIRONMENT: str = Field(..., env="PINECONE_ENVIRONMENT")
    # 주석: RAG 검색에 사용할 벡터 DB 인덱스 이름
    PINECONE_INDEX_NAME: str = Field(default="rag-tutorial-index", env="PINECONE_INDEX_NAME")

    # --- LoRA 설정 ---
    # 주석: LoRA 질문 추천 모델 저장 경로
    LORA_MODEL_PATH: str = Field(default="models/lora_suggest.pth", env="LORA_MODEL_PATH")
    # 주석: 질문 추천 시 검색할 갯수
    LORA_SUGGESTION_COUNT: int = Field(default=5, env="LORA_SUGGESTION_COUNT")

    class Config:
        # .env 파일에서 환경 변수를 로드하도록 설정
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

    @field_validator('PINECONE_API_KEY', 'PINECONE_ENVIRONMENT', mode="before")
    def check_required_env(cls, v, field):
        """필수 환경 변수가 설정되었는지 확인합니다."""
        if not v:
            raise ValueError(f"'{field.name}' 환경 변수는 필수입니다.")
        return v

# -----------------------------------------------------------------------------
# 2. 인스턴스 및 로딩 함수
# -----------------------------------------------------------------------------
_settings: Optional[Settings] = None

def get_settings() -> Settings:
    """
    애플리케이션 전체에서 사용할 설정 객체 (Settings 인스턴스)를 반환합니다.
    
    주석: FastAPI의 Dependency Injection(DI)에 사용되어 설정값을 쉽게 주입합니다.
    
    Input:
        없음 (환경 변수 파일에서 자동 로드)
        
    Output:
        Settings: 애플리케이션 설정값이 담긴 Pydantic 객체.
        
    예시 Input:
        (시스템 환경 변수 또는 .env 파일에 아래 내용이 존재)
        OPENAI_API_KEY="sk-..."
        PINECONE_API_KEY="abc-123"
        PINECONE_ENVIRONMENT="gcp-starter"
        
    예시 Output:
        Settings(
            API_V1_STR='/api/v1', 
            SERVER_PORT=8000, 
            OPENAI_API_KEY='sk-...',
            PINECONE_INDEX_NAME='rag-tutorial-index',
            ...
        )
    """
    global _settings

    if _settings is None:
        _settings = Settings()
    
    return _settings

# -----------------------------------------------------------------------------
# 3. DB 연결 정보 확인 함수 (선택적)
# -----------------------------------------------------------------------------

def get_vector_db_config() -> dict:
    """
    벡터 DB 연결에 필요한 설정값들을 딕셔너리 형태로 반환합니다.
    
    주석: vectorstore.py 모듈에서 이 정보를 사용하여 연결을 초기화합니다.
    
    Input:
        없음 (Settings 객체에서 가져옴)
        
    Output:
        dict: 벡터 DB 연결 정보.
        
    예시 Input:
        (Settings 객체가 위에서 로드된 상태)
        
    예시 Output:
        {
            "provider": "PINECONE",
            "api_key": "abc-123",
            "environment": "gcp-starter",
            "index_name": "rag-tutorial-index"
        }
    """
    if get_settings().VECTOR_DB_PROVIDER.upper() == "PINECONE":
        return {
            "provider": get_settings().VECTOR_DB_PROVIDER,
            "api_key": get_settings().PINECONE_API_KEY,
            "environment": get_settings().PINECONE_ENVIRONMENT,
            "index_name": get_settings().PINECONE_INDEX_NAME,
        }
    # 추후 Faiss, Weaviate 등 다른 DB가 추가될 수 있습니다.
    else:
        # 다른 DB의 경우 해당 설정을 추가합니다.
        return {"provider": get_settings().VECTOR_DB_PROVIDER}


# -----------------------------------------------------------------------------
# 4. LLM 모델 정보 확인 함수 (선택적)
# -----------------------------------------------------------------------------

def get_llm_model_names() -> dict:
    """
    각 서비스에 사용될 LLM 모델 이름과 API 정보를 딕셔너리 형태로 반환합니다.
    예시 Output:
    {
    "rag_llm_name": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llm_api_base": "http://localhost:8080/v1",
    "llm_api_key": null, # 또는 "hf_..."
    "embedding_model": "Bge-m3",
    "classifier_model_path": "local/llama3-classifier"
    }
    """
    return {
        "rag_llm_name": get_settings().LLM_MODEL_NAME, # LLaMA 3 이름
        "llm_api_base": get_settings().LLM_API_BASE,   # LLaMA 3 엔드포인트
        "llm_api_key": get_settings().LLM_API_KEY,     # (선택적) LLaMA 3 API 키
        "embedding_model": get_settings().EMBEDDING_MODEL_NAME,
        "classifier_model_path": get_settings().CLASSIFIER_MODEL_PATH
    }



if __name__ == "__main__":
    # 설정 로드 테스트 및 디버깅
    print("--- Settings Loaded ---")
    current_settings = get_settings()
    print(f"Server Port: {current_settings.SERVER_PORT}")
    print(f"RAG LLM: {current_settings.LLM_MODEL_NAME}")
    print(f"Vector DB Provider: {current_settings.VECTOR_DB_PROVIDER}")
    print(f"Vector DB Config (Partial): {get_vector_db_config()}")