# Setup Ollama (One-time)

1. Download and install Ollama from https://ollama.com
2. Pull a good model (recommended for agentic work):
```bash
</> bash
# Fast & decent
ollama pull llama3.2
# or for better reasoning:
ollama pull qwen2.5:14b
# or even stronger (if you have enough RAM):
ollama pull llama3.3:70b
```
3. Start Ollama (it runs as a service):
```bash
</> bash
ollama serve
```
!! You can run Ollama from your application (MAC).  

4. How to Switch Models
Just change one line in .env:  

- For zero cost / local → DEFAULT_MODEL=ollama:llama3.2
- For better quality → DEFAULT_MODEL=openai:gpt-4o-mini  

---------------------------------------------------------

## A classic Docker networking problem.
Your Ollama is running on your host machine (macOS) at http://localhost:11434, which you can access from the browser. 
However, inside the Docker container, localhost (or 127.0.0.1) refers to the container itself, not your Mac. 
That's why you get "Connection refused".  
- Quick Fix (Best for macOS)
  - Update your .env file to use Docker's special hostname for the host machine:
  ```bash
    DEFAULT_MODEL=ollama:llama3.2
    OLLAMA_BASE_URL=http://host.docker.internal:11434
    ```
  - Rebuild the image and run the container again.