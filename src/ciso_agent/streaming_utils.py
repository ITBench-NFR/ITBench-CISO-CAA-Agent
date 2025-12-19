# Copyright contributors to the ITBench project. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from typing import List, Dict, Any

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:
    try:
        from langchain.callbacks.base import BaseCallbackHandler
    except ImportError:
        from langchain.callbacks import BaseCallbackHandler


class StreamingMetricsCallback(BaseCallbackHandler):
    """Callback handler to track TTFT and Token Generation Speed during streaming."""
    
    def __init__(self, debug: bool = True):
        super().__init__()
        self.all_ttft_values: List[float] = []
        self.all_tgs_values: List[float] = []
        self.current_request_start_time = None
        self.current_first_token_time = None
        self.current_last_token_time = None
        self.current_token_count = 0
        self.current_tokens_received = False
        self.debug = debug
        self.callback_invocations = {
            "on_llm_start": 0,
            "on_llm_new_token": 0,
            "on_llm_end": 0,
        }
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Called when the LLM starts running."""
        self.callback_invocations["on_llm_start"] += 1
        if self.debug:
            print(f"[DEBUG] StreamingMetricsCallback.on_llm_start called (count: {self.callback_invocations['on_llm_start']})")
        
        self.current_request_start_time = time.time()
        self.current_first_token_time = None
        self.current_last_token_time = None
        self.current_token_count = 0
        self.current_tokens_received = False
    
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Called when a new token is generated."""
        self.callback_invocations["on_llm_new_token"] += 1
        current_time = time.time()
        
        if not self.current_tokens_received:
            # First token received - calculate TTFT
            self.current_first_token_time = current_time
            if self.current_request_start_time is not None:
                ttft = current_time - self.current_request_start_time
                self.all_ttft_values.append(ttft)
                if self.debug:
                    print(f"[DEBUG] First token received! TTFT: {ttft:.4f}s")
            self.current_tokens_received = True
        
        # Update last token time for each token
        self.current_last_token_time = current_time
        self.current_token_count += 1
        
        if self.debug and self.current_token_count <= 3:
            print(f"[DEBUG] Token {self.current_token_count} received: {repr(token[:20])}")
    
    def on_llm_end(self, response, **kwargs: Any) -> None:
        """Called when the LLM finishes running."""
        self.callback_invocations["on_llm_end"] += 1
        if self.debug:
            print(f"[DEBUG] StreamingMetricsCallback.on_llm_end called (count: {self.callback_invocations['on_llm_end']}, tokens: {self.current_token_count})")
        
        # Calculate Token Generation Speed
        if (self.current_first_token_time is not None and 
            self.current_last_token_time is not None and 
            self.current_token_count >= 2):
            
            generation_time = self.current_last_token_time - self.current_first_token_time
            if generation_time > 0:
                # Tokens generated after the first one
                tokens_generated = self.current_token_count - 1
                tokens_per_second = tokens_generated / generation_time
                self.all_tgs_values.append(tokens_per_second)
                if self.debug:
                    print(f"[DEBUG] TGS calculated: {tokens_per_second:.2f} tokens/sec ({tokens_generated} tokens in {generation_time:.4f}s)")
    
    def get_ttft_stats(self) -> Dict[str, Any]:
        """Get TTFT statistics."""
        if not self.all_ttft_values:
            return {"mean": None, "min": None, "max": None, "count": 0}
        
        import statistics
        return {
            "mean": statistics.mean(self.all_ttft_values) if len(self.all_ttft_values) > 0 else None,
            "min": min(self.all_ttft_values),
            "max": max(self.all_ttft_values),
            "count": len(self.all_ttft_values),
            "values": self.all_ttft_values,
        }
    
    def get_tgs_stats(self) -> Dict[str, Any]:
        """Get Token Generation Speed statistics."""
        if not self.all_tgs_values:
            return {"mean": None, "min": None, "max": None, "count": 0}
        
        import statistics
        return {
            "mean": statistics.mean(self.all_tgs_values) if len(self.all_tgs_values) > 0 else None,
            "min": min(self.all_tgs_values),
            "max": max(self.all_tgs_values),
            "count": len(self.all_tgs_values),
            "values": self.all_tgs_values,
        }


class StreamingLLMWrapper:
    """
    Wrapper around LangChain LLM to track streaming metrics by intercepting invoke/stream calls.
    
    This works around the issue where CrewAI/LiteLLM doesn't forward LangChain callbacks.
    We intercept the LLM calls directly to track TTFT and TGS metrics.
    """
    
    def __init__(self, llm, metrics_callback: StreamingMetricsCallback):
        self._llm = llm
        self.metrics_callback = metrics_callback
        # Preserve important attributes that CrewAI might check
        self.streaming = getattr(llm, 'streaming', False)
        self.model_name = getattr(llm, 'model_name', None)
        self.temperature = getattr(llm, 'temperature', None)
    
    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped LLM."""
        return getattr(self._llm, name)
    
    def invoke(self, input, config=None, **kwargs):
        """Intercept invoke calls to track metrics."""
        request_start = time.time()
        first_token_time = None
        last_token_time = None
        token_count = 0
        
        # Try to use streaming if available
        if hasattr(self._llm, 'stream') and self.metrics_callback.debug:
            print(f"[DEBUG] StreamingLLMWrapper: Using stream() to track metrics")
            try:
                # Use stream to get tokens one by one
                chunks = []
                for chunk in self._llm.stream(input, config=config, **kwargs):
                    chunks.append(chunk)
                    current_time = time.time()
                    
                    if first_token_time is None:
                        first_token_time = current_time
                        ttft = first_token_time - request_start
                        self.metrics_callback.all_ttft_values.append(ttft)
                        if self.metrics_callback.debug:
                            print(f"[DEBUG] First token received via stream! TTFT: {ttft:.4f}s")
                    
                    last_token_time = current_time
                    token_count += 1
                
                # Calculate TGS
                if first_token_time and last_token_time and token_count >= 2:
                    generation_time = last_token_time - first_token_time
                    if generation_time > 0:
                        tokens_generated = token_count - 1
                        tokens_per_second = tokens_generated / generation_time
                        self.metrics_callback.all_tgs_values.append(tokens_per_second)
                        if self.metrics_callback.debug:
                            print(f"[DEBUG] TGS calculated via stream: {tokens_per_second:.2f} tokens/sec")
                
                # Combine chunks into final response
                if chunks:
                    # Try to combine chunks - this depends on the chunk type
                    if hasattr(chunks[0], 'content'):
                        combined_content = ''.join(chunk.content for chunk in chunks if hasattr(chunk, 'content'))
                        # Create a response object similar to what invoke would return
                        final_chunk = chunks[-1]
                        if hasattr(final_chunk, 'content'):
                            final_chunk.content = combined_content
                        return final_chunk
                    return chunks[-1] if chunks else None
            except Exception as e:
                if self.metrics_callback.debug:
                    print(f"[DEBUG] Streaming failed, falling back to invoke: {e}")
        
        # Fallback to regular invoke
        response = self._llm.invoke(input, config=config, **kwargs)
        
        # If we couldn't track via streaming, we can't measure TTFT/TGS accurately
        # But we can at least record that a call was made
        if self.metrics_callback.debug:
            print(f"[DEBUG] StreamingLLMWrapper: Used invoke() (non-streaming), cannot track TTFT/TGS")
        
        return response
    
    def stream(self, input, config=None, **kwargs):
        """Intercept stream calls to track metrics."""
        request_start = time.time()
        first_token_time = None
        last_token_time = None
        token_count = 0
        
        if self.metrics_callback.debug:
            print(f"[DEBUG] StreamingLLMWrapper: Intercepting stream() call")
        
        for chunk in self._llm.stream(input, config=config, **kwargs):
            current_time = time.time()
            
            if first_token_time is None:
                first_token_time = current_time
                ttft = first_token_time - request_start
                self.metrics_callback.all_ttft_values.append(ttft)
                if self.metrics_callback.debug:
                    print(f"[DEBUG] First token received via stream! TTFT: {ttft:.4f}s")
            
            last_token_time = current_time
            token_count += 1
            
            yield chunk
        
        # Calculate TGS after streaming completes
        if first_token_time and last_token_time and token_count >= 2:
            generation_time = last_token_time - first_token_time
            if generation_time > 0:
                tokens_generated = token_count - 1
                tokens_per_second = tokens_generated / generation_time
                self.metrics_callback.all_tgs_values.append(tokens_per_second)
                if self.metrics_callback.debug:
                    print(f"[DEBUG] TGS calculated via stream: {tokens_per_second:.2f} tokens/sec ({tokens_generated} tokens in {generation_time:.4f}s)")
    
    def get_ttft_stats(self) -> Dict[str, Any]:
        """Get TTFT statistics."""
        return self.metrics_callback.get_ttft_stats()
    
    def get_tgs_stats(self) -> Dict[str, Any]:
        """Get Token Generation Speed statistics."""
        return self.metrics_callback.get_tgs_stats()

