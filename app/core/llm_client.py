# backend/app/core/llm_client.py

from openai import OpenAI
from typing import List, Dict, Any, Union, Optional
from app.config import get_settings

# -----------------------------------------------------------------------------
# 1. LLM 클라이언트 초기화
# -----------------------------------------------------------------------------

def get_llm_client() -> OpenAI:
    """
    설정값을 기반으로 OpenAI API 호환 클라이언트 객체를 초기화하고 반환합니다.
    
    주석: LLaMA 3 모델이 로컬(Ollama 등) 또는 외부 서비스(Groq 등)에서 
          OpenAI API 형식으로 서빙되는 경우 사용됩니다.
          
    Input:
        없음 (app.config.get_settings() 사용)
        
    Output:
        OpenAI: 초기화된 OpenAI 클라이언트 인스턴스.
        
    예시 Input:
        (config.py의 설정값)
        LLM_API_BASE="http://localhost:8080/v1"
        LLM_API_KEY=None 
        
    예시 Output:
        <openai.OpenAI object at ...>
    """
    settings = get_settings()
    
    # LLaMA 3를 로컬/외부 API로 호출하기 위한 설정
    base_url = settings.LLM_API_BASE
    api_key = settings.LLM_API_KEY if settings.LLM_API_KEY else "not-needed" 
    
    # OpenAI 클라이언트는 base_url을 지정하면 해당 엔드포인트로 요청을 보냅니다.
    # LLaMA 3 모델 이름은 호출 시점에 model 인자로 전달됩니다.
    client = OpenAI(
        base_url=base_url,
        api_key=api_key, 
    )
    return client


# -----------------------------------------------------------------------------
# 2. 채팅 완료 (Chat Completion) 호출 함수
# -----------------------------------------------------------------------------

async def generate_chat_completion(
    messages: List[Dict[str, str]], 
    temperature: float = 0.0, 
    stream: bool = False
) -> Union[Dict[str, Any], Any]:
    """
    LLM 클라이언트를 사용하여 채팅 완료(Chat Completion)를 생성합니다.

    ***RAG 답변 생성 등 주요 LLM 호출 로직에 사용됩니다.***

    Input:
        messages (List[Dict[str, str]]): 대화 기록 (프롬프트). 
                                         예: [{"role": "user", "content": "RAG가 뭐야?"}]
        temperature (float): 샘플링 온도 (기본 0.0, 결정적 답변 유도).
        stream (bool): 스트리밍 응답을 받을지 여부.

    Output:
        Dict[str, Any] | Any: LLM API 응답. 스트림이 False면 JSON 형태의 응답, 
                                            True면 응답 제너레이터(Generator).

    예시 Input (messages):
        [
            {"role": "system", "content": "너는 프롬프팅 기법 전문가야."},
            {"role": "user", "content": "프롬프트 체이닝에 대해 설명해줘."}
        ]
        
    예시 Output (stream=False):
        {
            "id": "chatcmpl-...", 
            "choices": [...], 
            "usage": {...} 
        }
    """
    settings = get_settings()
    client = get_llm_client()
    
    # LLaMA 3 모델 이름 사용
    model_name = settings.LLM_MODEL_NAME

    try:
        # 이 함수는 동기 함수이지만, FastAPI는 기본적으로 비동기 환경에서 작동합니다.
        # 실제 운영에서는 클라이언트 라이브러리가 비동기 API(client.chat.completions.create)를 지원하거나, 
        # run_in_executor를 통해 동기 코드를 비동기로 실행해야 합니다.
        
        # 현재는 설명을 위해 동기 함수 호출 형태로 작성합니다.
        # (OpenAI 라이브러리는 .acreate() 형태로 비동기를 지원함)
        
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            stream=stream
        )
        return response
    
    except Exception as e:
        # 실제 애플리케이션에서는 로깅 및 에러 처리가 필요합니다.
        print(f"LLM 호출 에러 발생: {e}")
        raise e


