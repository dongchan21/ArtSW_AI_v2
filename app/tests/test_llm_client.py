# backend/app/tests/test_llm_client.py

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from openai import OpenAI
from app.core.llm_client import get_llm_client, generate_chat_completion
from app.config import get_settings # config 모듈의 설정 함수 사용

# ----------------------------------------------------------------------
# 1. Fixture: 설정값 초기화 및 목킹 (Mocking)
# ----------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    """
    LLM 클라이언트 테스트에 필요한 필수 환경 변수를 모킹합니다.
    """
    # config.py의 필수 환경 변수 모킹
    monkeypatch.setenv("LLM_MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
    monkeypatch.setenv("LLM_API_BASE", "http://mock-llama-server:8080/v1")
    monkeypatch.setenv("PINECONE_API_KEY", "dummy_key")
    monkeypatch.setenv("PINECONE_ENVIRONMENT", "dummy_env")
    
    # get_settings()를 호출하여 Settings 객체를 로드합니다. (app/config의 _settings 초기화)
    get_settings()
    
    # 테스트가 끝난 후 config 상태를 정리하는 것은 test_config.py에 있는 fixture에 맡깁니다.

# ----------------------------------------------------------------------
# 2. get_llm_client 테스트
# ----------------------------------------------------------------------

@patch('app.core.llm_client.OpenAI')
def test_get_llm_client_initializes_with_correct_config(MockOpenAI):
    """
    get_llm_client가 config.py의 설정값으로 OpenAI 클라이언트를 올바르게 초기화하는지 검증합니다.
    """
    # 1. 함수 호출
    client = get_llm_client()
    
    # 2. 검증 (MockOpenAI가 올바른 인수로 호출되었는지 확인)
    
    # LLM_API_BASE가 base_url로 전달되었는지 확인
    MockOpenAI.assert_called_once_with(
        base_url="http://mock-llama-server:8080/v1",
        api_key="not-needed" # LLM_API_KEY가 None일 때의 기본값
    )
    # 반환된 객체가 Mock 객체인지 확인 (실제 통신이 아닌 목킹된 클라이언트)
    # assert isinstance(client, MockOpenAI)


@patch('app.core.llm_client.OpenAI')
def test_get_llm_client_uses_actual_api_key_if_present(MockOpenAI, monkeypatch):
    """
    LLM_API_KEY가 설정되어 있을 때 해당 키를 사용하는지 검증합니다.
    """
    import app.config as config
    config._settings = None

    test_key = "real-test-key-456"
    monkeypatch.setenv("LLM_API_KEY", test_key)
    # get_settings() 호출로 새로운 설정 로드
    settings = get_settings() 

    get_llm_client()
    
    # 실제 API 키가 api_key 인수에 전달되었는지 확인
    MockOpenAI.assert_called_once_with(
        base_url=settings.LLM_API_BASE,
        api_key=test_key # 🌟 여기서 test_key 사용
    )

# ----------------------------------------------------------------------
# 3. generate_chat_completion 테스트
# ----------------------------------------------------------------------

# 클라이언트 객체와 LLM_MODEL_NAME을 목킹하여 통신 과정을 시뮬레이션
@pytest.mark.asyncio
@patch('app.core.llm_client.get_llm_client')
async def test_generate_chat_completion_calls_api_with_correct_args(mock_get_llm_client):
    """
    generate_chat_completion이 LLM 모델 이름, 메시지, 온도 등을 
    올바르게 LLM 클라이언트에 전달하는지 검증합니다.
    """
    # Mocking: client.chat.completions.create의 응답을 설정
    mock_client_instance = mock_get_llm_client.return_value
    mock_response = MagicMock()
    
    # 동기 함수로 작성되었으므로 .create를 목킹
    mock_client_instance.chat.completions.create.return_value = mock_response

    # 테스트 입력
    test_messages = [{"role": "user", "content": "RAG 설명해줘"}]
    test_temperature = 0.5
    
    # 1. 함수 호출 (비동기이므로 await 사용)
    response = await generate_chat_completion(
        messages=test_messages, 
        temperature=test_temperature,
        stream=False
    )
    
    # 2. 검증
    settings = get_settings()
    
    # 클라이언트의 create 함수가 올바른 인수로 호출되었는지 확인
    mock_client_instance.chat.completions.create.assert_called_once_with(
        model=settings.LLM_MODEL_NAME,  # LLaMA 3 모델 이름
        messages=test_messages,
        temperature=test_temperature,
        stream=False
    )
    
    # 반환된 값이 목킹된 응답 객체인지 확인
    assert response == mock_response

@pytest.mark.asyncio
@patch('app.core.llm_client.get_llm_client')
async def test_generate_chat_completion_handles_exception(mock_get_llm_client):
    """
    LLM API 호출 시 예외가 발생했을 때 이를 처리하고 다시 raise하는지 검증합니다.
    """
    # Mocking: API 호출 시 예외가 발생하도록 설정
    mock_client_instance = mock_get_llm_client.return_value
    mock_client_instance.chat.completions.create.side_effect = Exception("Mock API Error")

    with pytest.raises(Exception, match="Mock API Error"):
        await generate_chat_completion(messages=[{"role": "user", "content": "Test"}])