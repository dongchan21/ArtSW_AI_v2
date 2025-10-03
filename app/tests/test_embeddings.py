# backend/app/tests/test_embeddings.py

import pytest
from unittest.mock import patch, MagicMock
from app.core.embeddings import get_embedding_model, embed_texts
from app.config import get_settings
import app.config as config

# ----------------------------------------------------------------------
# Fixtures 및 Mocking 설정
# ----------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_embedding_settings(monkeypatch):
    """테스트 모듈 전체에 필요한 설정값을 모킹하고 로드합니다."""
    # 필수 환경 변수 모킹
    monkeypatch.setenv("EMBEDDING_MODEL_NAME", "mock/bge-m3-test")
    monkeypatch.setenv("PINECONE_API_KEY", "dummy_key")
    monkeypatch.setenv("PINECONE_ENVIRONMENT", "dummy_env")
    
    # 설정 객체를 초기화하거나 갱신하여 테스트 환경을 준비합니다.
    import app.config as config
    config._settings = None

    # 설정 객체를 초기화하거나 갱신하여 테스트 환경을 준비합니다.
    get_settings()

@pytest.fixture(autouse=True)
def reset_model_state():
    """각 테스트 후 전역 _embedding_model 상태를 초기화합니다."""
    # app.core.embeddings 모듈의 _embedding_model 전역 변수를 import하여 초기화
    import app.core.embeddings as emb
    emb._embedding_model = None

# ----------------------------------------------------------------------
# 1. get_embedding_model 테스트 (싱글톤 및 초기화 검증)
# ----------------------------------------------------------------------

@patch('app.core.embeddings.SentenceTransformer')
def test_get_embedding_model_initializes_once(MockSentenceTransformer):
    """
    get_embedding_model이 SentenceTransformer를 한 번만 초기화하는지 (싱글톤) 검증합니다.
    """
    # 1. 첫 번째 호출: 모델이 초기화되어야 함
    model1 = get_embedding_model()
    
    # 2. 두 번째 호출: 초기화 없이 기존 인스턴스를 반환해야 함
    model2 = get_embedding_model()
    
    settings = get_settings()
    
    # 검증 1: SentenceTransformer가 설정된 이름으로 한 번만 호출되었는지 확인
    MockSentenceTransformer.assert_called_once_with(settings.EMBEDDING_MODEL_NAME)
    
    # 검증 2: 두 호출이 동일한 인스턴스를 반환했는지 확인 (싱글톤)
    assert model1 is model2

    
   
# ----------------------------------------------------------------------
# 2. embed_texts 테스트 (입력/출력 형식 및 로직 검증)
# ----------------------------------------------------------------------

@patch('app.core.embeddings.get_embedding_model')
def test_embed_texts_for_single_text(mock_get_embedding_model):
    
    mock_model = mock_get_embedding_model.return_value
    
    expected_vector = [0.1, 0.2, 0.3, 0.4]
    
    # 🌟🌟 수정: 인덱싱 가능한 Mock 객체를 설정 🌟🌟
    
    # 1. 단일 벡터를 반환할 Mock 객체 설정
    mock_single_vector = MagicMock()
    mock_single_vector.tolist.return_value = expected_vector # 인덱싱 후 .tolist()가 반환할 값

    # 2. encode의 최종 반환 값(2차원 배열 역할)을 설정
    # side_effect를 사용하여 리스트처럼 인덱싱되도록 설정합니다.
    # MagicMock이 리스트처럼 인덱싱될 때 mock_single_vector를 반환하게 합니다.
    mock_encode_result = MagicMock()
    mock_encode_result.__getitem__.return_value = mock_single_vector 
    
    # 3. .tolist() 호출 시 실제 데이터 반환 (리스트 텍스트 테스트와 통일성을 위해)
    mock_encode_result.tolist.return_value = [expected_vector]
    
    mock_model.encode.return_value = mock_encode_result
    
    test_text = "프롬프트 체이닝은 무엇인가?"
    
    # 1. 함수 호출
    result_vector = embed_texts(test_text)
    
    # 2. 검증 (Assert)
    # 🌟 수정: Mock 객체가 아닌 실제 데이터와 비교
    assert result_vector == expected_vector 

    # 기타 검증은 그대로 유지
    mock_model.encode.assert_called_once()
    assert mock_model.encode.call_args[0][0] == [test_text]
    assert isinstance(result_vector, list)
    assert isinstance(result_vector[0], float)



@patch('app.core.embeddings.get_embedding_model')
def test_embed_texts_for_list_of_texts(mock_get_embedding_model):
    """
    텍스트 리스트 입력 시 올바른 형식의 벡터 리스트를 반환하는지 검증합니다.
    """
    # Mocking: encode 메서드가 가짜 numpy 배열을 반환하도록 설정
    mock_model = mock_get_embedding_model.return_value
    
    # 2차원 numpy 배열 형태의 Mock 벡터 (두 개의 텍스트에 대한 두 개의 벡터)
    mock_vectors = [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0]
    ]
    mock_model.encode.return_value = MagicMock(tolist=lambda: mock_vectors)
    
    test_texts = ["첫 번째 텍스트 청크.", "두 번째 텍스트 청크."]
    
    # 1. 함수 호출
    result_vectors = embed_texts(test_texts)
    
    # 2. 검증
    # encode가 입력된 텍스트 리스트 그대로 호출되었는지 확인
    mock_model.encode.assert_called_once()
    assert mock_model.encode.call_args[0][0] == test_texts

    # 반환 값이 리스트 형태이고, 그 안에 리스트(벡터)가 포함되어 있는지 확인
    assert len(result_vectors) == 2
    
    # Mock 벡터 전체가 반환되었는지 확인
    assert result_vectors == mock_vectors