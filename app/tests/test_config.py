import pytest
from pydantic import ValidationError
from app.config import Settings, get_settings, get_vector_db_config, get_llm_model_names
import app.config as config

# 테스트 전, 현재 설정된 환경 변수들이 테스트에 영향을 주지 않도록
# pytest의 monkeypatch fixture를 사용하여 임시로 환경 변수를 설정/해제합니다.
# 🌟🌟 Fixture 추가: 각 테스트 시작/종료 시 _settings를 초기화 🌟🌟
@pytest.fixture(autouse=True)
def cleanup_settings():
    """
    각 테스트가 끝난 후 config 모듈의 내부 상태(_settings)를 초기화합니다.
    """
    # 테스트 시작 전에는 특별히 할 일 없음
    yield 
    # 테스트가 끝난 후 _settings를 None으로 설정
    config._settings = None 

# ----------------------------------------------------------------------
# 1. 필수 환경 변수 누락 테스트
# ----------------------------------------------------------------------

def test_settings_raises_error_on_missing_required_vars(monkeypatch):
    """
    필수 환경 변수(예: PINECONE_API_KEY)가 누락되면 ValidationError를 발생시키는지 확인
    """
    # 필수 환경 변수들을 임시로 삭제
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    monkeypatch.delenv("PINECONE_ENVIRONMENT", raising=False)
    # LLaMA 3를 로컬/비공개 API로 사용할 경우 API_KEY는 필수가 아님.
    # 하지만 테스트를 위해 임시로 LLM_MODEL_NAME을 필수라 가정하고 테스트할 수도 있습니다.

    with pytest.raises(ValidationError) as excinfo:
        get_settings() 
    
    # 에러 메시지에 누락된 필드가 포함되어 있는지 확인
    assert any("PINECONE_API_KEY" in str(e) for e in excinfo.value.errors())
    assert any("PINECONE_ENVIRONMENT" in str(e) for e in excinfo.value.errors())
    
# ----------------------------------------------------------------------
# 2. 기본값 및 로드 성공 테스트
# ----------------------------------------------------------------------

def test_settings_loads_successfully_with_mocks(monkeypatch):
    """
    필수 환경 변수를 모의(Mock)로 설정했을 때 설정 객체가 성공적으로 로드되는지 확인
    """
    # 필수 환경 변수 모킹 (LLaMA 3 설정에 맞춤)
    monkeypatch.setenv("LLM_MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
    monkeypatch.setenv("PINECONE_API_KEY", "mock_pinecone_key")
    monkeypatch.setenv("PINECONE_ENVIRONMENT", "mock_env")
    
    settings = get_settings()
    
    # 기본값이 올바르게 로드되었는지 확인
    assert settings.SERVER_PORT == 8000
    assert settings.LOG_LEVEL == "INFO"
    # 모의 값이 올바르게 로드되었는지 확인
    assert settings.PINECONE_API_KEY == "mock_pinecone_key"
    assert settings.LLM_MODEL_NAME == "meta-llama/Meta-Llama-3-8B-Instruct"

# ----------------------------------------------------------------------
# 3. 헬퍼 함수 반환 값 테스트
# ----------------------------------------------------------------------

def test_get_vector_db_config_returns_correct_dict(monkeypatch):
    """
    get_vector_db_config 함수가 벡터 DB 정보를 올바른 딕셔너리 형태로 반환하는지 확인
    """
    # 필수 환경 변수 모킹
    monkeypatch.setenv("LLM_MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
    monkeypatch.setenv("PINECONE_API_KEY", "test_key_123")
    monkeypatch.setenv("PINECONE_ENVIRONMENT", "test_region")

    get_settings()

    config = get_vector_db_config()
    
    assert config['provider'] == "PINECONE"
    assert config['api_key'] == "test_key_123"
    assert config['environment'] == "test_region"
    assert config['index_name'] == "rag-tutorial-index" # 기본값

def test_get_llm_model_names_returns_correct_dict(monkeypatch):
    """
    get_llm_model_names 함수가 LLM 관련 정보를 올바르게 반환하는지 확인
    """
    # 환경 변수 모킹 (LLaMA 3 관련)
    monkeypatch.setenv("LLM_MODEL_NAME", "meta-llama/Meta-Llama-3-70B-Instruct")
    monkeypatch.setenv("LLM_API_BASE", "https://api.groq.com/v1")
    monkeypatch.setenv("PINECONE_API_KEY", "dummy_key")
    monkeypatch.setenv("PINECONE_ENVIRONMENT", "dummy_env")
    
    get_settings() 
    
    model_names = get_llm_model_names()
    
    assert model_names['rag_llm_name'] == "meta-llama/Meta-Llama-3-70B-Instruct"
    assert model_names['llm_api_base'] == "https://api.groq.com/v1"
    assert model_names['embedding_model'] == "Bge-m3" # 기본값
    assert 'llm_api_key' in model_names # 키 존재 확인

# ----------------------------------------------------------------------
# 4. get_settings 테스트 (DI용)
# ----------------------------------------------------------------------

def test_get_settings_returns_singleton(monkeypatch):
    """
    get_settings 함수가 Settings 인스턴스를 반환하는지 확인
    """
    # 환경 변수 모킹
    monkeypatch.setenv("LLM_MODEL_NAME", "llama3")
    monkeypatch.setenv("PINECONE_API_KEY", "single_key")
    monkeypatch.setenv("PINECONE_ENVIRONMENT", "single_env")
    
    s1 = get_settings()
    s2 = get_settings()
    
    assert isinstance(s1, Settings)
    # 두 번 호출해도 동일한 인스턴스를 반환해야 함 (싱글톤 패턴 확인)
    assert s1 is s2 
    assert s1.PINECONE_API_KEY == "single_key"
